#!/usr/bin/env python3
"""Parse BVH motion and export compact, browser-ready world joint positions.

The formal GT_calib dataset contains one left-hand BVH (Skeleton_0) and one
right-hand BVH (Skeleton_1) per take.  Each file has a vendor-added all-zero
motion row before the real 120 Hz capture and a final row without a delivered
Human.cma counter/timestamp counterpart.  This exporter validates that dataset
contract, omits both invalid boundary rows, samples to a browser-friendly
rate, evaluates standard BVH forward kinematics, and stores the result as
quantized frame-major positions.

The JSON is deliberately renderer-agnostic.  A browser reconstructs a value
with::

    position_bvh_units = origin_bvh_units + quantized_value * quantum_bvh_units

BVH does not define a physical unit, so the output never silently labels the
vendor coordinates as metres or millimetres.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "同步整理_20260829_三段"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "bvh_web"
DEFAULT_ALIGNMENT_DIR = PROJECT_ROOT / "outputs" / "mocap_video_alignment_review"

EXPORT_SCHEMA = "gt-calib-bvh-web-v1"
MANIFEST_SCHEMA = "gt-calib-bvh-web-manifest-v1"
FORMAL_SEGMENTS = ("01", "02", "03")
FORMAL_TAKES = {
    "01": ("01_210814_Take_000", "Take_000"),
    "02": ("02_210955_Take_001", "Take_001"),
    "03": ("03_211139_Take_002", "Take_002"),
}
EXPECTED_SOURCE_FRAME_COUNTS = {
    "01": 7216,
    "02": 7483,
    "03": 7350,
}
EXPECTED_OVERLAY_FRAME_COUNTS = {
    "01": 1748,
    "02": 1805,
    "03": 1800,
}
EXPECTED_UNAVAILABLE_OVERLAY_FRAMES = {
    "01": [0],
    "02": [],
    "03": [0],
}
EXPECTED_JOINTS_PER_HAND = 21
EXPECTED_END_SITES_PER_HAND = 5
EXPECTED_FRAME_TIME_S = 0.00833333
SUPPORTED_CHANNELS = frozenset(
    {
        "Xposition",
        "Yposition",
        "Zposition",
        "Xrotation",
        "Yrotation",
        "Zrotation",
    }
)


class BvhValidationError(ValueError):
    """Raised when a BVH or formal-take contract is not safe to publish."""


@dataclass(frozen=True)
class BvhNode:
    """One hierarchy node in source order."""

    name: str
    parent: int
    offset: tuple[float, float, float]
    channels: tuple[str, ...]
    channel_start: int
    is_end_site: bool


@dataclass(frozen=True)
class BvhClip:
    """A validated BVH hierarchy and its channel motion."""

    path: Path
    nodes: tuple[BvhNode, ...]
    motion: np.ndarray
    frame_time_s: float
    source_sha256: str

    @property
    def frame_count(self) -> int:
        return int(self.motion.shape[0])

    @property
    def channel_count(self) -> int:
        return int(self.motion.shape[1])

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time_s


class _TokenStream:
    def __init__(self, text: str, *, source: Path) -> None:
        self.tokens = re.findall(r"[{}]|[^\s{}]+", text)
        self.index = 0
        self.source = source

    def pop(self) -> str:
        if self.index >= len(self.tokens):
            raise BvhValidationError(
                f"Unexpected end of BVH hierarchy in {self.source}"
            )
        token = self.tokens[self.index]
        self.index += 1
        return token

    def expect(self, expected: str) -> None:
        actual = self.pop()
        if actual != expected:
            raise BvhValidationError(
                f"Expected {expected!r}, got {actual!r} in {self.source}"
            )

    def finished(self) -> bool:
        return self.index == len(self.tokens)


def _finite_float(token: str, *, context: str) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise BvhValidationError(f"Invalid float {token!r} in {context}") from exc
    if not math.isfinite(value):
        raise BvhValidationError(f"Non-finite float {token!r} in {context}")
    return value


def _positive_int(token: str, *, context: str, allow_zero: bool = False) -> int:
    try:
        value = int(token)
    except ValueError as exc:
        raise BvhValidationError(f"Invalid integer {token!r} in {context}") from exc
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise BvhValidationError(f"Invalid integer {value} in {context}")
    return value


def _parse_hierarchy(text: str, *, source: Path) -> tuple[BvhNode, ...]:
    stream = _TokenStream(text, source=source)
    stream.expect("HIERARCHY")
    nodes: list[BvhNode] = []
    channel_cursor = 0

    def parse_node(kind: str, parent: int, *, end_ordinal: int = 0) -> int:
        nonlocal channel_cursor
        if kind == "End":
            stream.expect("Site")
            if parent < 0:
                raise BvhValidationError(f"End Site cannot be a root in {source}")
            name = f"{nodes[parent].name}__EndSite{end_ordinal}"
            is_end_site = True
        else:
            if kind not in {"ROOT", "JOINT"}:
                raise BvhValidationError(f"Unsupported node kind {kind!r} in {source}")
            name = stream.pop()
            is_end_site = False

        stream.expect("{")
        offset: tuple[float, float, float] | None = None
        channels: tuple[str, ...] | None = () if is_end_site else None
        node_channel_start: int | None = channel_cursor if is_end_site else None
        node_index = len(nodes)
        # Reserve the node so child parent indices are stable.  It is replaced
        # after OFFSET/CHANNELS have been parsed.
        nodes.append(BvhNode(name, parent, (0.0, 0.0, 0.0), (), channel_cursor, is_end_site))
        child_end_ordinal = 0

        while True:
            token = stream.pop()
            if token == "}":
                break
            if token == "OFFSET":
                if offset is not None:
                    raise BvhValidationError(f"Duplicate OFFSET for {name} in {source}")
                offset = tuple(
                    _finite_float(stream.pop(), context=f"OFFSET for {name} in {source}")
                    for _ in range(3)
                )
            elif token == "CHANNELS":
                if is_end_site:
                    raise BvhValidationError(f"End Site has CHANNELS in {source}")
                if channels is not None:
                    raise BvhValidationError(f"Duplicate CHANNELS for {name} in {source}")
                count = _positive_int(
                    stream.pop(), context=f"CHANNELS for {name} in {source}", allow_zero=True
                )
                parsed = tuple(stream.pop() for _ in range(count))
                unknown = set(parsed) - SUPPORTED_CHANNELS
                if unknown:
                    raise BvhValidationError(
                        f"Unsupported channels {sorted(unknown)} for {name} in {source}"
                    )
                if len(set(parsed)) != len(parsed):
                    raise BvhValidationError(
                        f"Duplicate channel names for {name} in {source}: {parsed}"
                    )
                channels = parsed
                node_channel_start = channel_cursor
                channel_cursor += len(parsed)
            elif token == "JOINT":
                parse_node("JOINT", node_index)
            elif token == "End":
                parse_node("End", node_index, end_ordinal=child_end_ordinal)
                child_end_ordinal += 1
            else:
                raise BvhValidationError(
                    f"Unexpected hierarchy token {token!r} for {name} in {source}"
                )

        if offset is None:
            raise BvhValidationError(f"Missing OFFSET for {name} in {source}")
        if channels is None:
            raise BvhValidationError(f"Missing CHANNELS for {name} in {source}")
        if node_channel_start is None:
            raise AssertionError("Internal channel cursor inconsistency")
        nodes[node_index] = BvhNode(
            name=name,
            parent=parent,
            offset=offset,
            channels=channels,
            channel_start=node_channel_start,
            is_end_site=is_end_site,
        )
        return node_index

    stream.expect("ROOT")
    # parse_node consumes a kind token only for recursive calls, while the
    # initial ROOT keyword was already consumed above.
    root_name = stream.pop()
    stream.index -= 1
    parse_node("ROOT", -1)
    if nodes[0].name != root_name:
        raise AssertionError("Internal ROOT parser inconsistency")
    if not stream.finished():
        raise BvhValidationError(
            f"Unexpected trailing hierarchy tokens in {source}: "
            f"{stream.tokens[stream.index:stream.index + 4]}"
        )
    if len({node.name for node in nodes}) != len(nodes):
        raise BvhValidationError(f"Joint names are not unique in {source}")
    return tuple(nodes)


def parse_bvh(path: Path) -> BvhClip:
    """Parse a standard text BVH and fail closed on malformed motion rows."""

    path = Path(path)
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BvhValidationError(f"BVH is not UTF-8 text: {path}") from exc
    parts = re.split(r"(?m)^\s*MOTION\s*$", text, maxsplit=1)
    if len(parts) != 2:
        raise BvhValidationError(f"BVH must contain exactly one MOTION section: {path}")
    nodes = _parse_hierarchy(parts[0], source=path)

    lines = [line.strip() for line in parts[1].splitlines() if line.strip()]
    if len(lines) < 2:
        raise BvhValidationError(f"Incomplete MOTION header in {path}")
    frames_match = re.fullmatch(r"Frames\s*:\s*(\d+)", lines[0], flags=re.IGNORECASE)
    frame_time_match = re.fullmatch(
        r"Frame\s+Time\s*:\s*([^\s]+)", lines[1], flags=re.IGNORECASE
    )
    if frames_match is None or frame_time_match is None:
        raise BvhValidationError(f"Invalid MOTION header in {path}: {lines[:2]}")
    frame_count = _positive_int(frames_match.group(1), context=f"Frames in {path}")
    frame_time_s = _finite_float(
        frame_time_match.group(1), context=f"Frame Time in {path}"
    )
    if frame_time_s <= 0.0:
        raise BvhValidationError(f"Frame Time must be positive in {path}")

    motion_lines = lines[2:]
    if len(motion_lines) != frame_count:
        raise BvhValidationError(
            f"Declared {frame_count} frames but found {len(motion_lines)} rows in {path}"
        )
    channel_count = sum(len(node.channels) for node in nodes)
    if channel_count <= 0:
        raise BvhValidationError(f"BVH has no motion channels: {path}")
    motion = np.empty((frame_count, channel_count), dtype=np.float64)
    for frame_index, line in enumerate(motion_lines):
        tokens = line.split()
        if len(tokens) != channel_count:
            raise BvhValidationError(
                f"Frame {frame_index} has {len(tokens)} values; expected "
                f"{channel_count} in {path}"
            )
        motion[frame_index] = [
            _finite_float(token, context=f"frame {frame_index} in {path}")
            for token in tokens
        ]
    if not np.all(np.isfinite(motion)):
        raise BvhValidationError(f"Motion contains non-finite values in {path}")

    return BvhClip(
        path=path,
        nodes=nodes,
        motion=motion,
        frame_time_s=frame_time_s,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _axis_rotation(axis: int, degrees: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(degrees)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    matrices = np.zeros((len(degrees), 3, 3), dtype=np.float64)
    matrices[:, axis, axis] = 1.0
    first = (axis + 1) % 3
    second = (axis + 2) % 3
    matrices[:, first, first] = cosine
    matrices[:, first, second] = -sine
    matrices[:, second, first] = sine
    matrices[:, second, second] = cosine
    return matrices


def evaluate_world_positions(
    clip: BvhClip,
    frame_indices: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate BVH channel transforms in declared order using column vectors."""

    if frame_indices is None:
        indices = np.arange(clip.frame_count, dtype=np.int64)
    else:
        indices = np.asarray(frame_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise BvhValidationError("frame_indices must be a non-empty 1-D sequence")
        if np.any(indices < 0) or np.any(indices >= clip.frame_count):
            raise BvhValidationError("frame_indices contains an out-of-range frame")
        if np.any(np.diff(indices) <= 0):
            raise BvhValidationError("frame_indices must be strictly increasing")

    motion = clip.motion[indices]
    frame_count = len(indices)
    world_rotations: list[np.ndarray] = []
    world_positions = np.empty((frame_count, len(clip.nodes), 3), dtype=np.float64)
    axes = {"X": 0, "Y": 1, "Z": 2}

    for node_index, node in enumerate(clip.nodes):
        local_rotation = np.broadcast_to(np.eye(3), (frame_count, 3, 3)).copy()
        local_translation = np.broadcast_to(
            np.asarray(node.offset, dtype=np.float64), (frame_count, 3)
        ).copy()
        for channel_offset, channel in enumerate(node.channels):
            values = motion[:, node.channel_start + channel_offset]
            axis = axes[channel[0]]
            if channel.endswith("position"):
                # Compose a translation on the right.  This also handles the
                # uncommon but valid case where a translation follows a
                # rotation channel.
                local_translation += local_rotation[:, :, axis] * values[:, None]
            else:
                rotation = _axis_rotation(axis, values)
                local_rotation = np.einsum(
                    "nij,njk->nik", local_rotation, rotation, optimize=True
                )

        if node.parent < 0:
            world_rotation = local_rotation
            world_translation = local_translation
        else:
            parent_rotation = world_rotations[node.parent]
            world_rotation = np.einsum(
                "nij,njk->nik", parent_rotation, local_rotation, optimize=True
            )
            world_translation = world_positions[:, node.parent] + np.einsum(
                "nij,nj->ni", parent_rotation, local_translation, optimize=True
            )
        world_rotations.append(world_rotation)
        world_positions[:, node_index] = world_translation

    if not np.all(np.isfinite(world_positions)):
        raise BvhValidationError(f"Forward kinematics produced non-finite values: {clip.path}")
    return world_positions


def leading_zero_motion_frames(clip: BvhClip, *, tolerance: float = 1e-12) -> int:
    zero_rows = np.all(np.abs(clip.motion) <= tolerance, axis=1)
    nonzero = np.flatnonzero(~zero_rows)
    return clip.frame_count if len(nonzero) == 0 else int(nonzero[0])


def sample_source_indices(
    frame_count: int,
    frame_time_s: float,
    *,
    leading_frames_to_omit: int,
    target_fps: float,
) -> tuple[np.ndarray, int]:
    if frame_count <= 0 or frame_time_s <= 0.0 or target_fps <= 0.0:
        raise BvhValidationError("Invalid sampling parameters")
    if leading_frames_to_omit < 0 or leading_frames_to_omit >= frame_count:
        raise BvhValidationError("No usable source frames remain after leading omission")
    source_fps = 1.0 / frame_time_s
    stride = max(1, int(round(source_fps / target_fps)))
    indices = np.arange(leading_frames_to_omit, frame_count, stride, dtype=np.int64)
    # Preserve the final pose and its exact source time.  The browser uses the
    # source_frame_indices array, so the possibly shorter final interval is not
    # mistaken for a constant-rate sample.
    if indices[-1] != frame_count - 1:
        indices = np.append(indices, frame_count - 1)
    return indices, stride


def _non_end_nodes(clip: BvhClip) -> tuple[np.ndarray, list[dict[str, object]]]:
    kept_source_indices = [i for i, node in enumerate(clip.nodes) if not node.is_end_site]
    source_to_export = {source: exported for exported, source in enumerate(kept_source_indices)}
    nodes: list[dict[str, object]] = []
    for source_index in kept_source_indices:
        node = clip.nodes[source_index]
        parent = -1 if node.parent < 0 else source_to_export[node.parent]
        nodes.append(
            {
                "name": node.name,
                "parent": parent,
                "offset_bvh_units": [float(value) for value in node.offset],
                "channels": list(node.channels),
            }
        )
    return np.asarray(kept_source_indices, dtype=np.int64), nodes


def _dataset_relative(path: Path, dataset_root: Path) -> str:
    try:
        return path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except ValueError as exc:
        raise BvhValidationError(f"BVH path is outside dataset root: {path}") from exc


def _project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise BvhValidationError(f"Artifact path is outside project root: {path}") from exc


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_human_frame_counters(path: Path) -> tuple[np.ndarray, str]:
    """Load and validate the Human.cma FrameCounter axis only."""

    path = Path(path)
    digest = hashlib.sha256()
    counters: list[int] = []
    with path.open("rb") as raw:
        header_bytes = raw.readline()
        digest.update(header_bytes)
        try:
            header = header_bytes.decode("utf-8-sig").rstrip("\r\n").split("\t")
        except UnicodeDecodeError as exc:
            raise BvhValidationError(f"Human.cma header is not UTF-8: {path}") from exc
        if "FrameCounter" not in header:
            raise BvhValidationError(f"Human.cma lacks FrameCounter: {path}")
        counter_column = header.index("FrameCounter")
        for row_index, line in enumerate(raw):
            digest.update(line)
            if not line.strip():
                continue
            try:
                columns = line.decode("utf-8").rstrip("\r\n").split("\t")
            except UnicodeDecodeError as exc:
                raise BvhValidationError(
                    f"Human.cma row {row_index} is not UTF-8: {path}"
                ) from exc
            if counter_column >= len(columns):
                raise BvhValidationError(
                    f"Human.cma row {row_index} lacks FrameCounter: {path}"
                )
            try:
                counter = int(columns[counter_column].strip())
            except ValueError as exc:
                raise BvhValidationError(
                    f"Invalid Human.cma FrameCounter at row {row_index}: {path}"
                ) from exc
            counters.append(counter)
    if not counters:
        raise BvhValidationError(f"Human.cma has no data rows: {path}")
    values = np.asarray(counters, dtype=np.int64)
    if np.any(np.diff(values) != 1):
        raise BvhValidationError(f"Human.cma FrameCounter is not contiguous: {path}")
    return values, digest.hexdigest()


def validate_bvh_cma_root_ordinal_contract(
    left: BvhClip,
    right: BvhClip,
    human_cma: Path,
    *,
    tolerance_bvh_units: float = 1e-4,
) -> dict[str, object]:
    """Prove BVH frame 1 maps to CMA row 1 and BVH N-2 to CMA last.

    MovementCap's formal exports map CMA millimetres to unnamed BVH units as
    ``[-X, Z, Y] / 10``.  Checking both roots at the first and last strict
    source frames prevents a one-frame shift from passing on counts alone.
    """

    human_cma = Path(human_cma)
    with human_cma.open("r", encoding="utf-8-sig", newline="") as stream:
        fields = stream.readline().rstrip("\r\n").split("\t")
        side_columns: dict[str, list[int]] = {}
        for skeleton, hand in (("Skeleton_0", "LeftHand"), ("Skeleton_1", "RightHand")):
            columns = [f"{skeleton}_{hand}_Pos_{axis}(mm)" for axis in "XYZ"]
            missing = set(columns) - set(fields)
            if missing:
                raise BvhValidationError(
                    f"Human.cma lacks root position fields {sorted(missing)}: {human_cma}"
                )
            side_columns[skeleton] = [fields.index(column) for column in columns]
        second_values: dict[str, np.ndarray] | None = None
        last_values: dict[str, np.ndarray] | None = None
        row_count = 0
        for line in stream:
            if not line.strip():
                continue
            columns = line.rstrip("\r\n").split("\t")
            try:
                selected = {
                    skeleton: np.asarray(
                        [float(columns[index]) for index in indices], dtype=np.float64
                    )
                    for skeleton, indices in side_columns.items()
                }
            except (IndexError, ValueError) as exc:
                raise BvhValidationError(
                    f"Invalid root position at Human.cma row {row_count}: {human_cma}"
                ) from exc
            if row_count == 1:
                second_values = selected
            last_values = selected
            row_count += 1
    if row_count != left.frame_count - 1 or left.frame_count != right.frame_count:
        raise BvhValidationError(f"Cannot validate BVH/CMA root ordinals: {human_cma}")
    if second_values is None or last_values is None:
        raise BvhValidationError(f"Human.cma has too few rows: {human_cma}")

    checks: dict[str, object] = {}
    for skeleton, clip in (("Skeleton_0", left), ("Skeleton_1", right)):
        def converted(xyz: np.ndarray) -> np.ndarray:
            if not np.all(np.isfinite(xyz)):
                raise BvhValidationError(
                    f"Non-finite root position in Human.cma: {human_cma}"
                )
            return np.asarray([-xyz[0], xyz[2], xyz[1]], dtype=np.float64) / 10.0

        first_error = float(
            np.max(np.abs(clip.motion[1, :3] - converted(second_values[skeleton])))
        )
        last_error = float(
            np.max(
                np.abs(
                    clip.motion[clip.frame_count - 2, :3]
                    - converted(last_values[skeleton])
                )
            )
        )
        if max(first_error, last_error) > tolerance_bvh_units:
            raise BvhValidationError(
                f"{skeleton} BVH/CMA ordinal proof exceeds {tolerance_bvh_units} "
                f"BVH units: first={first_error}, last={last_error}"
            )
        checks[skeleton] = {
            "bvh_frame_1_matches_human_cma_row_1_max_abs_error_bvh_units": first_error,
            "bvh_frame_n_minus_2_matches_last_human_cma_row_max_abs_error_bvh_units": (
                last_error
            ),
        }
    return {
        "status": "pass",
        "cma_mm_to_bvh_units_for_contract_check": "[-X, Z, Y] / 10",
        "tolerance_bvh_units": tolerance_bvh_units,
        "checks": checks,
    }


def load_overlay_video_sync(
    alignment_csv: Path,
    human_cma: Path,
    *,
    bvh_frame_count: int,
    expected_output_frame_count: int | None = None,
) -> dict[str, object]:
    """Build a fail-closed overlay-video-frame to BVH-frame mapping.

    The vendor BVH index 0 is an all-zero seed and occupies the same ordinal as
    Human.cma's first data row.  BVH index 1 exactly matches the second CMA row,
    making ``BVH index = FrameCounter - first``.  The final BVH row has no CMA
    counterpart.  Alignment CSV rows already contain the two bracketing CMA
    counters and interpolation alpha used to render each H.264 output frame.
    """

    alignment_csv = Path(alignment_csv)
    human_cma = Path(human_cma)
    counters, human_sha256 = load_human_frame_counters(human_cma)
    if len(counters) != bvh_frame_count - 1:
        raise BvhValidationError(
            f"Human.cma has {len(counters)} rows but BVH has {bvh_frame_count} "
            f"frames (including one seed): {human_cma}"
        )
    first_counter = int(counters[0])
    if int(counters[-1]) - first_counter != bvh_frame_count - 2:
        raise BvhValidationError(f"Human.cma/BVH endpoint mismatch: {human_cma}")

    required = {
        "output_frame",
        "source_video_frame",
        "source_video_nominal_pts_s",
        "mocap_low_frame_counter",
        "mocap_high_frame_counter",
        "interpolation_alpha",
    }
    output_frames: list[int] = []
    source_video_frames: list[int] = []
    nominal_pts: list[float] = []
    fractional_bvh_frames: list[float] = []
    low_bvh_frames: list[int] = []
    high_bvh_frames: list[int] = []
    with alignment_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise BvhValidationError(
                f"Alignment CSV lacks fields {sorted(missing)}: {alignment_csv}"
            )
        for row_index, row in enumerate(reader):
            try:
                output_frame = int(row["output_frame"])
                source_video_frame = int(row["source_video_frame"])
                pts = float(row["source_video_nominal_pts_s"])
                low_counter = int(row["mocap_low_frame_counter"])
                high_counter = int(row["mocap_high_frame_counter"])
                alpha = float(row["interpolation_alpha"])
            except (TypeError, ValueError) as exc:
                raise BvhValidationError(
                    f"Invalid alignment values at row {row_index}: {alignment_csv}"
                ) from exc
            if output_frame != row_index:
                raise BvhValidationError(
                    f"Alignment output_frame must be contiguous from zero at row "
                    f"{row_index}: {alignment_csv}"
                )
            if not all(math.isfinite(value) for value in (pts, alpha)):
                raise BvhValidationError(
                    f"Non-finite alignment value at row {row_index}: {alignment_csv}"
                )
            if not 0.0 <= alpha <= 1.0:
                raise BvhValidationError(
                    f"Interpolation alpha outside [0,1] at row {row_index}: {alignment_csv}"
                )
            if high_counter - low_counter != 1:
                raise BvhValidationError(
                    f"MOCAP bracket is not consecutive at row {row_index}: {alignment_csv}"
                )
            low_index = low_counter - first_counter
            high_index = high_counter - first_counter
            if low_index < 0 or high_index > bvh_frame_count - 2:
                raise BvhValidationError(
                    f"BVH mapping out of bounds [{low_index}, {high_index}] at row "
                    f"{row_index}: {alignment_csv}"
                )
            fractional = low_index + alpha * (high_index - low_index)
            output_frames.append(output_frame)
            source_video_frames.append(source_video_frame)
            nominal_pts.append(pts)
            low_bvh_frames.append(low_index)
            high_bvh_frames.append(high_index)
            fractional_bvh_frames.append(fractional)

    row_count = len(output_frames)
    if row_count < 2:
        raise BvhValidationError(f"Alignment CSV needs at least two rows: {alignment_csv}")
    if expected_output_frame_count is not None and row_count != expected_output_frame_count:
        raise BvhValidationError(
            f"Alignment CSV has {row_count} rows; expected {expected_output_frame_count}: "
            f"{alignment_csv}"
        )
    if np.any(np.diff(np.asarray(source_video_frames, dtype=np.int64)) != 1):
        raise BvhValidationError(f"source_video_frame is not contiguous: {alignment_csv}")
    if np.any(np.diff(np.asarray(nominal_pts, dtype=np.float64)) <= 0.0):
        raise BvhValidationError(f"Video nominal PTS is not strictly increasing: {alignment_csv}")
    if np.any(np.diff(np.asarray(fractional_bvh_frames, dtype=np.float64)) <= 0.0):
        raise BvhValidationError(
            f"Fractional BVH source frames are not strictly increasing: {alignment_csv}"
        )

    pts_steps = np.diff(np.asarray(nominal_pts, dtype=np.float64))
    median_step = float(np.median(pts_steps))
    if median_step <= 0.0 or not np.allclose(
        pts_steps, median_step, rtol=0.0, atol=1e-9
    ):
        raise BvhValidationError(f"Overlay video nominal PTS cadence is not fixed: {alignment_csv}")
    nominal_fps = 1.0 / median_step
    if not math.isclose(nominal_fps, 30.0, rel_tol=0.0, abs_tol=1e-6):
        raise BvhValidationError(
            f"Unexpected overlay nominal FPS {nominal_fps}: {alignment_csv}"
        )

    unavailable_output_frames = [
        output_frame
        for output_frame, low_index in zip(output_frames, low_bvh_frames)
        if low_index == 0
    ]
    rounded_fractional: list[float | None] = [
        None if low_index == 0 else round(value, 6)
        for low_index, value in zip(low_bvh_frames, fractional_bvh_frames)
    ]
    rounding_errors = [
        abs(original - rounded)
        for original, rounded in zip(fractional_bvh_frames, rounded_fractional)
        if rounded is not None
    ]
    rounding_error = max(rounding_errors, default=0.0)
    first_valid_output_frame = next(
        (index for index, value in enumerate(rounded_fractional) if value is not None),
        None,
    )
    if first_valid_output_frame is None:
        raise BvhValidationError(f"No overlay frame has a usable BVH pose: {alignment_csv}")
    return {
        "master": "mocap_h264_overlay_video",
        "alignment_csv": {
            "path": _project_relative(alignment_csv),
            "sha256": _sha256_path(alignment_csv),
        },
        "human_cma": {
            "path": _project_relative(human_cma),
            "sha256": human_sha256,
            "row_count": len(counters),
            "first_frame_counter": first_counter,
            "last_frame_counter": int(counters[-1]),
        },
        "bvh_seed_contract": {
            "bvh_source_frame_zero_is_all_zero_vendor_seed": True,
            "first_human_cma_frame_counter_maps_to_dummy_bvh_source_frame": 0,
            "second_human_cma_row_maps_to_bvh_source_frame": 1,
            "last_bvh_source_frame_has_no_human_cma_counterpart": True,
            "formula": (
                "bvh_source_frame = Human.cma FrameCounter - human_cma.first_frame_counter"
            ),
        },
        "overlay_output_frame_count": row_count,
        "overlay_nominal_fps": nominal_fps,
        "overlay_source_video_frame_first": source_video_frames[0],
        "overlay_source_video_frame_last": source_video_frames[-1],
        "mapping_encoding": (
            "array index is H.264 overlay output frame; value is fractional raw BVH "
            "source frame; null means the bracket touches dummy BVH frame 0"
        ),
        "bvh_fractional_source_frame_by_output_frame": rounded_fractional,
        "first_valid_output_frame": first_valid_output_frame,
        "unavailable_output_frames": unavailable_output_frames,
        "fractional_frame_rounding_decimals": 6,
        "fractional_frame_max_rounding_error": rounding_error,
        "bvh_bracket_bounds": {
            "minimum_low_source_frame": min(low_bvh_frames),
            "maximum_high_source_frame": max(high_bvh_frames),
            "bvh_source_frame_count": bvh_frame_count,
            "strictly_usable_source_frame_first": 1,
            "strictly_usable_source_frame_last": bvh_frame_count - 2,
        },
        "validation": {
            "status": "pass",
            "alignment_rows_match_overlay_output_frames": True,
            "human_cma_rows_match_non_seed_bvh_frames": True,
            "all_mappings_inside_bvh_bounds": True,
            "dummy_seed_mappings_explicitly_null": True,
            "all_published_video_frames_have_strict_bvh_pose": not unavailable_output_frames,
            "output_frames_contiguous_from_zero": True,
            "fractional_bvh_frames_strictly_increasing": True,
            "overlay_nominal_pts_fixed_30_fps": True,
        },
    }


def _validate_hand_clip(
    clip: BvhClip,
    *,
    side: str,
    segment_key: str,
) -> None:
    if side not in {"left", "right"}:
        raise BvhValidationError(f"Unknown hand side: {side}")
    non_end = [node for node in clip.nodes if not node.is_end_site]
    end_sites = [node for node in clip.nodes if node.is_end_site]
    if len(non_end) != EXPECTED_JOINTS_PER_HAND:
        raise BvhValidationError(
            f"{clip.path} has {len(non_end)} joints; expected {EXPECTED_JOINTS_PER_HAND}"
        )
    if len(end_sites) != EXPECTED_END_SITES_PER_HAND:
        raise BvhValidationError(
            f"{clip.path} has {len(end_sites)} End Sites; expected {EXPECTED_END_SITES_PER_HAND}"
        )
    expected_prefix = "LeftHand" if side == "left" else "RightHand"
    if non_end[0].name != "Hips":
        raise BvhValidationError(f"Unexpected root {non_end[0].name!r} in {clip.path}")
    if any(not node.name.startswith(expected_prefix) for node in non_end[1:]):
        raise BvhValidationError(
            f"{clip.path} does not match the declared {side}-hand Skeleton mapping"
        )
    expected_channels = (
        "Xposition",
        "Yposition",
        "Zposition",
        "Xrotation",
        "Yrotation",
        "Zrotation",
    )
    if non_end[0].channels != expected_channels:
        raise BvhValidationError(f"Unexpected root channel contract in {clip.path}")
    if any(node.channels != ("Xrotation", "Yrotation", "Zrotation") for node in non_end[1:]):
        raise BvhValidationError(f"Unexpected finger channel contract in {clip.path}")
    if clip.channel_count != 66:
        raise BvhValidationError(f"Unexpected channel count {clip.channel_count} in {clip.path}")
    expected_frames = EXPECTED_SOURCE_FRAME_COUNTS[segment_key]
    if clip.frame_count != expected_frames:
        raise BvhValidationError(
            f"{clip.path} has {clip.frame_count} frames; expected {expected_frames}"
        )
    if not math.isclose(
        clip.frame_time_s, EXPECTED_FRAME_TIME_S, rel_tol=0.0, abs_tol=1e-10
    ):
        raise BvhValidationError(
            f"Unexpected frame time {clip.frame_time_s:.12g}s in {clip.path}"
        )
    leading = leading_zero_motion_frames(clip)
    if leading != 1:
        raise BvhValidationError(
            f"{clip.path} has {leading} leading all-zero rows; expected exactly 1"
        )
    if np.any(np.all(np.abs(clip.motion[1:]) <= 1e-12, axis=1)):
        raise BvhValidationError(f"Unexpected all-zero motion row after frame 0 in {clip.path}")


def discover_formal_take(
    dataset_root: Path, segment_key: str
) -> tuple[Path, Path, Path, str]:
    if segment_key not in FORMAL_TAKES:
        raise BvhValidationError(f"Unsupported formal segment {segment_key!r}")
    expected_segment, take_name = FORMAL_TAKES[segment_key]
    segment = Path(dataset_root) / expected_segment
    take_dir = segment / "动捕" / take_name
    left = take_dir / f"{take_name}_Skeleton_0.bvh"
    right = take_dir / f"{take_name}_Skeleton_1.bvh"
    for path in (segment, take_dir, left, right):
        if not path.exists():
            raise BvhValidationError(f"Required formal dataset path is missing: {path}")
    return segment, left, right, take_name


def prepare_segment_payload(
    dataset_root: Path,
    segment_key: str,
    *,
    alignment_dir: Path = DEFAULT_ALIGNMENT_DIR,
    target_fps: float = 60.0,
    quantum_bvh_units: float = 0.001,
) -> dict[str, object]:
    """Validate one formal take and create its browser JSON payload."""

    if not math.isfinite(quantum_bvh_units) or quantum_bvh_units <= 0.0:
        raise BvhValidationError("quantum_bvh_units must be finite and positive")
    segment, left_path, right_path, take_name = discover_formal_take(
        dataset_root, segment_key
    )
    left = parse_bvh(left_path)
    right = parse_bvh(right_path)
    _validate_hand_clip(left, side="left", segment_key=segment_key)
    _validate_hand_clip(right, side="right", segment_key=segment_key)
    if left.frame_count != right.frame_count or not math.isclose(
        left.frame_time_s, right.frame_time_s, rel_tol=0.0, abs_tol=1e-12
    ):
        raise BvhValidationError(
            f"Left/right BVH timing mismatch for {segment.name}: "
            f"{left.frame_count}@{left.frame_time_s} vs "
            f"{right.frame_count}@{right.frame_time_s}"
        )

    human_cma = left_path.parent / f"{take_name}_Human.cma"
    alignment_csv = (
        Path(alignment_dir)
        / f"{segment.name}_mocap_video_aligned_strict25.alignment.csv"
    )
    h264_video = alignment_csv.with_name(
        alignment_csv.name.removesuffix(".alignment.csv") + "_h264.mp4"
    )
    for required_path in (human_cma, alignment_csv, h264_video):
        if not required_path.is_file():
            raise BvhValidationError(f"Required video-sync artifact is missing: {required_path}")
    root_ordinal_validation = validate_bvh_cma_root_ordinal_contract(
        left, right, human_cma
    )
    video_sync = load_overlay_video_sync(
        alignment_csv,
        human_cma,
        bvh_frame_count=left.frame_count,
        expected_output_frame_count=EXPECTED_OVERLAY_FRAME_COUNTS[segment_key],
    )
    if (
        video_sync["unavailable_output_frames"]
        != EXPECTED_UNAVAILABLE_OVERLAY_FRAMES[segment_key]
    ):
        raise BvhValidationError(
            f"Unexpected dummy-seed video mappings for {segment.name}: "
            f"{video_sync['unavailable_output_frames']}"
        )
    video_sync["h264_overlay_video"] = {
        "path": _project_relative(h264_video),
        "sha256": _sha256_path(h264_video),
        "byte_size": h264_video.stat().st_size,
    }

    source_indices, stride = sample_source_indices(
        # BVH frame 0 is a dummy and frame N-1 has no Human.cma counterpart.
        # Passing N-1 as the exclusive source bound exports strict frames
        # 1..N-2 while preserving their final endpoint.
        left.frame_count - 1,
        left.frame_time_s,
        leading_frames_to_omit=1,
        target_fps=target_fps,
    )
    left_source_nodes, left_nodes = _non_end_nodes(left)
    right_source_nodes, right_nodes = _non_end_nodes(right)
    left_positions = evaluate_world_positions(left, source_indices)[:, left_source_nodes]
    right_positions = evaluate_world_positions(right, source_indices)[:, right_source_nodes]
    positions = np.concatenate((left_positions, right_positions), axis=1)

    bounds_min = positions.min(axis=(0, 1))
    bounds_max = positions.max(axis=(0, 1))
    origin = (bounds_min + bounds_max) * 0.5
    quantized = np.rint((positions - origin[None, None, :]) / quantum_bvh_units)
    int32 = np.iinfo(np.int32)
    if quantized.min() < int32.min or quantized.max() > int32.max:
        raise BvhValidationError(
            f"Quantized coordinates exceed int32 for {segment.name}; increase quantum"
        )
    quantized = quantized.astype(np.int32)
    reconstructed = origin[None, None, :] + quantized.astype(np.float64) * quantum_bvh_units
    max_error = float(np.max(np.abs(reconstructed - positions)))
    if max_error > quantum_bvh_units * 0.500001:
        raise BvhValidationError(
            f"Quantization error {max_error} exceeds half quantum for {segment.name}"
        )

    flat_frames = quantized.reshape(len(source_indices), -1).tolist()
    frame_times_s = (source_indices - source_indices[0]) * left.frame_time_s
    edges_per_hand = [[node["parent"], index] for index, node in enumerate(left_nodes) if node["parent"] >= 0]
    right_edges = [[node["parent"], index] for index, node in enumerate(right_nodes) if node["parent"] >= 0]
    if len(edges_per_hand) != EXPECTED_JOINTS_PER_HAND - 1 or len(right_edges) != EXPECTED_JOINTS_PER_HAND - 1:
        raise BvhValidationError(f"Unexpected edge count for {segment.name}")

    duration_s = float(frame_times_s[-1])
    effective_fps = (len(source_indices) - 1) / duration_s if duration_s > 0 else 0.0
    payload: dict[str, object] = {
        "schema": EXPORT_SCHEMA,
        "segment": segment.name,
        "segment_key": segment_key,
        "take": take_name,
        "coordinate_contract": {
            "unit": "bvh_unit",
            "physical_unit_declared_by_bvh": False,
            "axis_order": "XYZ",
            "transform_convention": (
                "column vectors; OFFSET first; CHANNEL transforms composed in declared order"
            ),
            "scope": "standalone_mocap_animation_not_camera_calibrated",
        },
        "source": {
            "left": {
                "path": _dataset_relative(left.path, dataset_root),
                "sha256": left.source_sha256,
                "skeleton": "Skeleton_0",
            },
            "right": {
                "path": _dataset_relative(right.path, dataset_root),
                "sha256": right.source_sha256,
                "skeleton": "Skeleton_1",
            },
        },
        "video_sync": video_sync,
        "motion": {
            "source_frame_count": left.frame_count,
            "source_frame_time_s": left.frame_time_s,
            "source_fps": left.fps,
            "source_duration_s_including_placeholder": (left.frame_count - 1) * left.frame_time_s,
            "leading_all_zero_placeholder_frames_omitted": 1,
            "trailing_bvh_frames_without_human_cma_counterpart_omitted": 1,
            "strict_source_frame_first": 1,
            "strict_source_frame_last": left.frame_count - 2,
            "usable_source_frame_count": left.frame_count - 2,
            "sampling_stride": stride,
            "requested_target_fps": target_fps,
            "nominal_export_fps": left.fps / stride,
            "effective_export_fps_including_final_endpoint": effective_fps,
            "exported_frame_count": len(source_indices),
            "exported_duration_s": duration_s,
            "source_frame_indices": source_indices.tolist(),
        },
        "skeletons": [
            {
                "side": "left",
                "source_skeleton": "Skeleton_0",
                "position_joint_offset": 0,
                "joint_count": len(left_nodes),
                "source_end_sites_omitted": sum(node.is_end_site for node in left.nodes),
                "joints": left_nodes,
                "edges": edges_per_hand,
            },
            {
                "side": "right",
                "source_skeleton": "Skeleton_1",
                "position_joint_offset": len(left_nodes),
                "joint_count": len(right_nodes),
                "source_end_sites_omitted": sum(node.is_end_site for node in right.nodes),
                "joints": right_nodes,
                "edges": right_edges,
            },
        ],
        "positions": {
            "encoding": "frame-major-flat-int32-json",
            "components": "XYZ",
            "joint_order": "left_skeleton_then_right_skeleton",
            "total_joint_count": positions.shape[1],
            "values_per_frame": positions.shape[1] * 3,
            "quantum_bvh_units": quantum_bvh_units,
            "origin_bvh_units": origin.tolist(),
            "max_abs_quantization_error_bvh_units": max_error,
            "bounds_bvh_units": {
                "min": bounds_min.tolist(),
                "max": bounds_max.tolist(),
            },
            "frames": flat_frames,
        },
        "validation": {
            "status": "pass",
            "finite_motion": True,
            "left_right_frame_contract": True,
            "formal_dataset_contract": True,
            "bvh_human_cma_root_ordinal_contract": root_ordinal_validation,
            "strict_cma_matched_source_endpoint_preserved": (
                int(source_indices[-1]) == left.frame_count - 2
            ),
        },
    }
    return payload


def _compact_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_segments(
    dataset_root: Path,
    output_dir: Path,
    segment_keys: Sequence[str],
    *,
    alignment_dir: Path = DEFAULT_ALIGNMENT_DIR,
    target_fps: float = 60.0,
    quantum_bvh_units: float = 0.001,
) -> dict[str, object]:
    """Export takes atomically and publish the manifest only after all pass."""

    normalized = tuple(segment_keys)
    if not normalized or len(set(normalized)) != len(normalized):
        raise BvhValidationError("Segments must be a non-empty unique sequence")
    unknown = set(normalized) - set(FORMAL_SEGMENTS)
    if unknown:
        raise BvhValidationError(f"Unsupported segments: {sorted(unknown)}")

    output_dir = Path(output_dir)
    entries: list[dict[str, object]] = []
    for key in normalized:
        payload = prepare_segment_payload(
            Path(dataset_root),
            key,
            alignment_dir=alignment_dir,
            target_fps=target_fps,
            quantum_bvh_units=quantum_bvh_units,
        )
        filename = f"{payload['segment']}_bvh_motion.json"
        content = _compact_json_bytes(payload)
        output_path = output_dir / filename
        _atomic_write(output_path, content)
        entries.append(
            {
                "segment_key": key,
                "segment": payload["segment"],
                "file": filename,
                "sha256": hashlib.sha256(content).hexdigest(),
                "byte_size": len(content),
                "exported_frame_count": payload["motion"]["exported_frame_count"],
                "exported_duration_s": payload["motion"]["exported_duration_s"],
                "nominal_export_fps": payload["motion"]["nominal_export_fps"],
                "overlay_output_frame_count": payload["video_sync"][
                    "overlay_output_frame_count"
                ],
                "h264_overlay_video": payload["video_sync"]["h264_overlay_video"],
            }
        )

    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "export_schema": EXPORT_SCHEMA,
        "validation": {"status": "pass", "all_requested_segments_exported": True},
        "takes": entries,
    }
    _atomic_write(output_dir / "manifest.json", _compact_json_bytes(manifest))
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_ALIGNMENT_DIR)
    parser.add_argument(
        "--segment",
        dest="segments",
        action="append",
        choices=FORMAL_SEGMENTS,
        help="Formal segment to export; repeat as needed (default: 01, 02, 03)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=60.0,
        help="Browser sampling target; source endpoint is always preserved",
    )
    parser.add_argument(
        "--quantum",
        type=float,
        default=0.001,
        help="Position quantization step in unspecified BVH source units",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    segments = tuple(args.segments or FORMAL_SEGMENTS)
    manifest = export_segments(
        args.dataset_root,
        args.output_dir,
        segments,
        alignment_dir=args.alignment_dir,
        target_fps=args.target_fps,
        quantum_bvh_units=args.quantum,
    )
    for entry in manifest["takes"]:
        print(
            f"{entry['segment_key']}: {entry['file']} "
            f"{entry['exported_frame_count']} frames, "
            f"{entry['byte_size']} bytes, sha256={entry['sha256']}"
        )
    print(f"manifest: {Path(args.output_dir) / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
