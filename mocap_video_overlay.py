#!/usr/bin/env python3
"""Render CMAvatar motion capture directly on its synchronized RGB frames.

This is the video/MOCAP alignment product.  It deliberately does not load or
draw glove data:

* A fixed, full-span set of decoded ROS1 color messages is compared with the
  corresponding decoded MP4 frames before an index-to-index pairing is
  accepted.  Equal frame counts alone are not treated as evidence.
* The delivered clock map converts that device timestamp to CMAvatar time.
* The two bracketing 120 Hz CMAvatar rows are linearly interpolated at the
  camera timestamp rather than rounded to a nearest MOCAP frame.
* The delivered CS-400 transform and RGB lens model project all 21 joints of
  both hands into the original image.
* An optional rear wrist/mount proxy can be extrapolated for visualization.  It
  is disabled by default, is not an additional measured MOCAP joint, and does
  not alter the delivered 21-joint hand poses.
* Output is trimmed to the single contiguous interval in which every output
  frame has a valid MOCAP interpolation.  A CSV records the exact provenance
  of every rendered frame.

"Fully synchronized" below means the selected output interval has a valid
timestamp mapping and interpolation for every frame.  It does not mean that an
independent 2D hand-label set has measured zero reprojection error.
"""

from __future__ import annotations

import argparse
import bz2
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

try:
    import lz4.frame as lz4_frame
except ImportError:  # Reproducible dependency is declared; CLI remains a fallback.
    lz4_frame = None

import gt_calib_viz as viz
import bvh_web_export as bvh_export


ALIGNMENT_SCHEMA = "gt_calib.mocap_video_alignment.v2"
LEFT_COLOR_BGR = (225, 65, 245)
RIGHT_COLOR_BGR = (245, 220, 45)
OUTLINE_BGR = (12, 12, 12)
ROS_OP_CHUNK = 5
ROS_OP_MESSAGE_DATA = 2
FRAME_CONTENT_SAMPLE_COUNT = 31
FRAME_CONTENT_ANALYSIS_SIZE = (240, 135)
FRAME_CONTENT_MAX_LUMA_MSE = 6.0
FRAME_CONTENT_MIN_PSNR_DB = 39.0
FRAME_CONTENT_MAX_DHASH_HAMMING = 4
FRAME_CONTENT_MIN_DISTINGUISHABLE = 24
FRAME_CONTENT_OFFSETS = (-2, -1, 0, 1, 2)
MOCAP_WRIST_INDEX = 0
MOCAP_MIDDLE_MCP_INDEX = 9
# Visualization-only: extend behind the wrist by 80% of wrist->middle-MCP.
VISUALIZATION_REAR_WRIST_MOUNT_SCALE = 0.8
MOCAP_WORLD_TRANSLATION_CANDIDATE_PROFILE = (
    "candidate_origin_translation_not_ground_truth"
)
MOCAP_POSITION_SOURCE_HUMAN_CMA = "human-cma"
MOCAP_POSITION_SOURCE_SKELETON_BVH = "skeleton-bvh"
MOCAP_POSITION_SOURCES = (
    MOCAP_POSITION_SOURCE_HUMAN_CMA,
    MOCAP_POSITION_SOURCE_SKELETON_BVH,
)
BVH_TO_MOCAP_WORLD_CONTRACT = "[-10*X_bvh, 10*Z_bvh, 10*Y_bvh] mm"
BVH_CMA_ROOT_TOLERANCE_MM = 1e-3


@dataclass(frozen=True)
class InterpolationAudit:
    low_indices: np.ndarray
    high_indices: np.ndarray
    alpha: np.ndarray
    bracket_span_s: np.ndarray
    nearest_delta_s: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class PreparedMocapVideo:
    segment_dir: Path
    video: viz.VideoInfo
    rgb_device_s: np.ndarray
    cmavatar_s: np.ndarray
    clock_inside: np.ndarray
    clock_mapping: viz.ClockMapping
    mocap_path: Path
    mocap: viz.MocapSequence
    mocap_position_source: str
    mocap_position_paths: Mapping[str, Path]
    mocap_position_contract: Mapping[str, Any]
    mocap_mm: np.ndarray
    mocap_world_translation_mm: tuple[float, float, float]
    mocap_valid: np.ndarray
    interpolation: InterpolationAudit
    calibration: viz.PreviewCalibration
    sync_report: Mapping[str, Any]
    max_interpolation_gap_ms: float


@dataclass(frozen=True)
class Ros1MessageRef:
    timestamp_ns: int
    chunk_data_offset: int
    chunk_data_length: int
    compression: str
    uncompressed_size: int
    record_offset: int
    connection_id: int


@dataclass(frozen=True)
class BagColorSample:
    message_index: int
    message_timestamp_ns: int
    device_timestamp_us: int
    frame_number: int
    encoding: str
    width: int
    height: int
    jpeg_sha256: str
    decoded_bgr_sha256: str
    analysis_luma: np.ndarray
    analysis_luma_sha256: str
    dhash64: int


def visualization_rear_wrist_mount_anchor(
    joints_world: np.ndarray,
    *,
    scale: float = VISUALIZATION_REAR_WRIST_MOUNT_SCALE,
) -> np.ndarray:
    """Extrapolate a display-only rear mount point without changing joints.

    The anchor lies behind ``wrist`` on the opposite ray from
    ``wrist -> middle_mcp``.  It is an overlay aid, not an additional MOCAP GT
    joint.  Leading dimensions (for example left/right hands) are preserved.
    """

    joints = np.asarray(joints_world)
    if (
        joints.ndim < 2
        or joints.shape[-2] <= MOCAP_MIDDLE_MCP_INDEX
        or joints.shape[-1] != 3
    ):
        raise ValueError(
            "MOCAP joints must have shape (..., at least 10 joints, 3)"
        )
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError(
            "Visualization rear wrist/mount scale must be finite and non-negative"
        )

    wrist = np.asarray(
        joints[..., MOCAP_WRIST_INDEX, :], dtype=np.float64
    ).copy()
    middle_mcp = np.asarray(
        joints[..., MOCAP_MIDDLE_MCP_INDEX, :], dtype=np.float64
    )
    return wrist - float(scale) * (middle_mcp - wrist)


def _mocap_world_translation_header(
    translation_xyz_mm: Sequence[float],
) -> str | None:
    """Format a visible warning for a non-identity candidate origin shift."""

    translation = np.asarray(translation_xyz_mm, dtype=np.float64)
    if translation.shape != (3,) or np.any(~np.isfinite(translation)):
        raise ValueError("MOCAP world translation must be finite XYZ")
    components: list[str] = []
    for axis, value in zip("XYZ", translation, strict=True):
        if value == 0.0:
            continue
        magnitude = np.format_float_positional(abs(float(value)), trim="-")
        sign = "+" if value > 0.0 else "-"
        components.append(f"{sign}{axis}{magnitude}")
    if not components:
        return None
    return f"CANDIDATE origin {' '.join(components)} mm / not GT"


def _normalize_mocap_position_source(value: str) -> str:
    source = str(value)
    if source not in MOCAP_POSITION_SOURCES:
        raise ValueError(
            f"Unsupported MOCAP position source {source!r}; expected one of "
            f"{MOCAP_POSITION_SOURCES}"
        )
    return source


def bvh_positions_to_mocap_world_mm(points_bvh_units: np.ndarray) -> np.ndarray:
    """Invert MovementCap's CMA-world to BVH position conversion.

    The vendor export writes CMA world millimetres as ``[-X, Z, Y] / 10``.
    Forward-kinematics positions therefore return to the calibrated MOCAP
    world as ``[-10*X_bvh, 10*Z_bvh, 10*Y_bvh]`` millimetres.
    """

    points = np.asarray(points_bvh_units, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError("BVH positions must have shape (..., 3)")
    if np.any(~np.isfinite(points)):
        raise ValueError("BVH positions must be finite before axis/unit conversion")
    return np.stack(
        (-10.0 * points[..., 0], 10.0 * points[..., 2], 10.0 * points[..., 1]),
        axis=-1,
    )


def _expected_bvh_joint_names(side: str) -> tuple[str, ...]:
    if side not in {"left", "right"}:
        raise ValueError(f"Unknown BVH hand side: {side}")
    hand = "LeftHand" if side == "left" else "RightHand"
    return (
        "Hips",
        *(
            f"{hand}{finger}{index}"
            for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky")
            for index in range(1, 5)
        ),
    )


def _load_skeleton_bvh_positions(
    segment: Path,
    human_cma_path: Path,
    human_mocap: viz.MocapSequence,
    take_number: str,
) -> tuple[np.ndarray, dict[str, Path], dict[str, Any]]:
    """Load Skeleton_0/1 FK positions on the untouched Human.cma time axis."""

    segment_key = segment.name.split("_", 1)[0]
    formal = bvh_export.FORMAL_TAKES.get(segment_key)
    take_name = f"Take_{take_number}"
    if formal != (segment.name, take_name):
        raise bvh_export.BvhValidationError(
            "skeleton-bvh is fail-closed to the formal 01/02/03 take layout; "
            f"got segment={segment.name!r}, take={take_name!r}"
        )

    take_dir = segment / "动捕" / take_name
    paths = {
        "left": take_dir / f"{take_name}_Skeleton_0.bvh",
        "right": take_dir / f"{take_name}_Skeleton_1.bvh",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise bvh_export.BvhValidationError(
            "Required Skeleton BVH input is missing: " + ", ".join(missing)
        )

    clips = {
        side: bvh_export.parse_bvh(path) for side, path in paths.items()
    }
    for side, clip in clips.items():
        bvh_export._validate_hand_clip(  # noqa: SLF001 - shared formal contract
            clip,
            side=side,
            segment_key=segment_key,
        )
        names = tuple(node.name for node in clip.nodes if not node.is_end_site)
        expected_names = _expected_bvh_joint_names(side)
        if names != expected_names:
            raise bvh_export.BvhValidationError(
                f"Unexpected {side} Skeleton BVH joint order in {clip.path}: {names}"
            )

    left = clips["left"]
    right = clips["right"]
    if left.frame_count != right.frame_count or not math.isclose(
        left.frame_time_s,
        right.frame_time_s,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise bvh_export.BvhValidationError(
            "Skeleton_0/1 BVH frame count or frame time does not match"
        )
    cma_rows = len(human_mocap.frame_counters)
    if left.frame_count != cma_rows + 1:
        raise bvh_export.BvhValidationError(
            f"Human.cma has {cma_rows} rows but Skeleton BVH has "
            f"{left.frame_count} frames; expected exactly one trailing BVH-only frame"
        )

    root_ordinal_contract = bvh_export.validate_bvh_cma_root_ordinal_contract(
        left,
        right,
        human_cma_path,
    )
    # CMA row i maps to raw BVH frame i.  BVH frame 0 is the vendor's all-zero
    # seed and is explicitly unavailable; BVH frame N-1 has no CMA timestamp.
    ordinal_indices = np.arange(cma_rows, dtype=np.int64)
    side_positions: list[np.ndarray] = []
    for side in ("left", "right"):
        clip = clips[side]
        joint_indices = np.asarray(
            [index for index, node in enumerate(clip.nodes) if not node.is_end_site],
            dtype=np.int64,
        )
        positions_bvh = bvh_export.evaluate_world_positions(
            clip,
            ordinal_indices,
        )[:, joint_indices]
        side_positions.append(bvh_positions_to_mocap_world_mm(positions_bvh))
    points_mm = np.stack(side_positions, axis=1)
    if points_mm.shape != human_mocap.points_mm.shape:
        raise bvh_export.BvhValidationError(
            f"Skeleton BVH FK shape {points_mm.shape} does not match "
            f"Human.cma position shape {human_mocap.points_mm.shape}"
        )

    root_error_mm = float(
        np.max(
            np.abs(
                points_mm[1:, :, MOCAP_WRIST_INDEX]
                - human_mocap.points_mm[1:, :, MOCAP_WRIST_INDEX]
            )
        )
    )
    if root_error_mm > BVH_CMA_ROOT_TOLERANCE_MM:
        raise bvh_export.BvhValidationError(
            "Skeleton BVH/CMA all-frame root ordinal check exceeds "
            f"{BVH_CMA_ROOT_TOLERANCE_MM} mm: {root_error_mm} mm"
        )
    points_mm[0] = np.nan

    contract: dict[str, Any] = {
        "status": "pass",
        "position_source": MOCAP_POSITION_SOURCE_SKELETON_BVH,
        "timestamp_and_frame_counter_source": "Human.cma",
        "human_cma_timestamps_preserved_without_resampling": True,
        "human_cma_joint_positions_used_as_render_source": False,
        "human_cma_root_positions_used_for_ordinal_validation_only": True,
        "bvh_forward_kinematics": "bvh_web_export.evaluate_world_positions",
        "bvh_source_sha256": {
            side: clips[side].source_sha256 for side in ("left", "right")
        },
        "cma_world_mm_to_bvh_units": "[-X, Z, Y] / 10",
        "bvh_units_to_mocap_world_mm": BVH_TO_MOCAP_WORLD_CONTRACT,
        "ordinal_mapping": "raw BVH frame i = Human.cma data-row ordinal i",
        "cma_row_count": cma_rows,
        "bvh_source_frame_count": left.frame_count,
        "bvh_frame_zero_is_all_zero_vendor_seed_and_unavailable": True,
        "strictly_usable_cma_ordinal_first": 1,
        "strictly_usable_cma_ordinal_last": cma_rows - 1,
        "final_bvh_frame_without_cma_timestamp_omitted": True,
        "joint_order": list(viz.MOCAP_NAMES),
        "all_usable_root_ordinals_max_abs_error_mm": root_error_mm,
        "all_usable_root_ordinals_tolerance_mm": BVH_CMA_ROOT_TOLERANCE_MM,
        "endpoint_root_ordinal_validation": root_ordinal_contract,
    }
    return points_mm, paths, contract


@dataclass(frozen=True)
class Mp4FrameSample:
    frame_index: int
    width: int
    height: int
    decoded_bgr_sha256: str
    analysis_luma: np.ndarray
    analysis_luma_sha256: str
    dhash64: int


def interpolation_audit(
    source_times_s: np.ndarray,
    target_times_s: np.ndarray,
    *,
    max_gap_s: float,
) -> InterpolationAudit:
    """Return the exact source bracket used for every target timestamp."""

    source = np.asarray(source_times_s, dtype=np.float64)
    target = np.asarray(target_times_s, dtype=np.float64)
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("Interpolation timestamps must be one-dimensional")
    if len(source) < 2 or np.any(~np.isfinite(source)) or np.any(np.diff(source) <= 0.0):
        raise ValueError("Source timestamps must be finite and strictly increasing")
    if max_gap_s <= 0.0:
        raise ValueError("max_gap_s must be positive")

    low, high, alpha, span, valid = viz._interpolation_brackets(
        source,
        target,
        max_gap_s=max_gap_s,
    )
    nearest = np.minimum(
        np.abs(target - source[low]),
        np.abs(source[high] - target),
    )
    alpha = alpha.astype(np.float64, copy=True)
    alpha[~valid] = np.nan
    span = span.astype(np.float64, copy=True)
    nearest = nearest.astype(np.float64, copy=True)
    span[~valid] = np.nan
    nearest[~valid] = np.nan
    return InterpolationAudit(low, high, alpha, span, nearest, valid)


def synchronized_interval(valid: np.ndarray) -> tuple[int, int]:
    """Return a half-open, gap-free synchronized interval or fail closed."""

    mask = np.asarray(valid, dtype=bool)
    if mask.ndim != 1 or not np.any(mask):
        raise ValueError("No synchronized MOCAP/video frames")
    indices = np.flatnonzero(mask)
    first, stop = int(indices[0]), int(indices[-1] + 1)
    interior_missing = np.flatnonzero(~mask[first:stop])
    if len(interior_missing):
        example = (interior_missing[:8] + first).tolist()
        raise ValueError(
            "Synchronized MOCAP interval contains interior gaps; refusing to "
            f"call it frame-complete (examples: {example})"
        )
    return first, stop


def prepare_mocap_video(
    segment_dir: Path,
    calibration_path: Path,
    *,
    max_interpolation_gap_ms: float = 25.0,
    mocap_position_source: str = MOCAP_POSITION_SOURCE_HUMAN_CMA,
    mocap_world_translation_mm: Sequence[float] = (0.0, 0.0, 0.0),
) -> PreparedMocapVideo:
    segment = Path(segment_dir)
    if max_interpolation_gap_ms <= 0.0:
        raise ValueError("max_interpolation_gap_ms must be positive")
    position_source = _normalize_mocap_position_source(mocap_position_source)
    translation = np.asarray(mocap_world_translation_mm, dtype=np.float64)
    if translation.shape != (3,) or np.any(~np.isfinite(translation)):
        raise ValueError(
            "mocap_world_translation_mm must contain exactly three finite XYZ values"
        )
    translation_tuple = tuple(float(value) for value in translation)

    video = viz.video_info(segment / "视频" / "RGB.mp4")
    rgb_device_s = viz.ros1_topic_index_timestamps(
        segment / "原始BAG与内参" / "camera_1_rgb_depth.bag"
    )
    if len(rgb_device_s) != video.frame_count:
        raise ValueError(
            f"RGB bag messages ({len(rgb_device_s)}) do not match MP4 frames "
            f"({video.frame_count})"
        )

    clocks = viz.load_clock_mapping(
        segment / "同步校验" / "camera_cmavatar_alignment.csv"
    )
    _, cmavatar_s, clock_inside = clocks.map(rgb_device_s)
    take_number = viz._take_number(segment)
    mocap_path = (
        segment / "动捕" / f"Take_{take_number}" / f"Take_{take_number}_Human.cma"
    )
    human_mocap = viz.load_mocap_human(mocap_path)
    if position_source == MOCAP_POSITION_SOURCE_HUMAN_CMA:
        mocap = human_mocap
        position_paths: dict[str, Path] = {"human_cma": mocap_path}
        position_contract: dict[str, Any] = {
            "status": "pass",
            "position_source": MOCAP_POSITION_SOURCE_HUMAN_CMA,
            "timestamp_and_frame_counter_source": "Human.cma",
            "human_cma_timestamps_preserved_without_resampling": True,
            "human_cma_joint_positions_used_as_render_source": True,
            "position_unit": "millimetres in MOCAP world",
            "joint_order": list(viz.MOCAP_NAMES),
        }
    else:
        bvh_points_mm, position_paths, position_contract = (
            _load_skeleton_bvh_positions(
                segment,
                mocap_path,
                human_mocap,
                take_number,
            )
        )
        mocap = replace(human_mocap, points_mm=bvh_points_mm)
    max_gap_s = max_interpolation_gap_ms * 1e-3
    interpolation = interpolation_audit(
        mocap.times_s,
        cmavatar_s,
        max_gap_s=max_gap_s,
    )
    mocap_mm, series_valid = viz.interpolate_series(
        mocap.times_s,
        mocap.points_mm,
        cmavatar_s,
        max_gap_s=max_gap_s,
    )
    if not np.array_equal(series_valid, interpolation.valid):
        raise AssertionError("Interpolation audit does not match position interpolation")
    if np.any(translation != 0.0):
        # Candidate rigid origin correction: apply the same vector to all
        # 2 hands x 21 joints only after timestamp interpolation.
        mocap_mm = mocap_mm + translation.reshape(1, 1, 1, 3)

    sync_report = json.loads(
        (segment / "同步校验" / "common_interval_sync_report.json").read_text(
            encoding="utf-8"
        )
    )
    if not bool(sync_report.get("pass_within_common_interval")):
        raise ValueError(f"Delivered common-interval synchronization did not pass: {segment}")
    common_start_s = int(sync_report["cmavatar_start_ns"]) * 1e-9
    common_end_s = int(sync_report["cmavatar_end_ns"]) * 1e-9
    mocap_valid = (
        series_valid
        & interpolation.valid
        & clock_inside
        & (cmavatar_s >= common_start_s)
        & (cmavatar_s <= common_end_s)
        & np.all(np.isfinite(mocap_mm), axis=(1, 2, 3))
    )
    synchronized_interval(mocap_valid)

    calibration = viz.load_preview_calibration(calibration_path)
    applies_to = set(calibration.payload.get("applies_to", []))
    if segment.name not in applies_to:
        raise ValueError(
            f"Calibration {calibration.path} does not declare applicability to {segment.name}"
        )
    if not bool(calibration.payload.get("camera_consistency", {}).get("passed")):
        raise ValueError("Delivered camera-consistency check did not pass")

    return PreparedMocapVideo(
        segment_dir=segment,
        video=video,
        rgb_device_s=rgb_device_s,
        cmavatar_s=cmavatar_s,
        clock_inside=clock_inside,
        clock_mapping=clocks,
        mocap_path=mocap_path,
        mocap=mocap,
        mocap_position_source=position_source,
        mocap_position_paths=position_paths,
        mocap_position_contract=position_contract,
        mocap_mm=mocap_mm,
        mocap_world_translation_mm=translation_tuple,
        mocap_valid=mocap_valid,
        interpolation=interpolation,
        calibration=calibration,
        sync_report=sync_report,
        max_interpolation_gap_ms=max_interpolation_gap_ms,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"|")
    digest.update(",".join(str(value) for value in array.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _fixed_span_sample_indices(
    frame_count: int,
    sample_count: int = FRAME_CONTENT_SAMPLE_COUNT,
) -> list[int]:
    """Return deterministic, uniform samples including both endpoints."""

    if frame_count <= 0 or sample_count < 0:
        raise ValueError("frame_count must be positive and sample_count non-negative")
    if sample_count == 0:
        return list(range(frame_count))
    if frame_count <= sample_count:
        return list(range(frame_count))
    indices = np.unique(
        np.rint(np.linspace(0, frame_count - 1, sample_count)).astype(np.int64)
    )
    if len(indices) != sample_count or indices[0] != 0 or indices[-1] != frame_count - 1:
        raise AssertionError("Full-span sample construction did not preserve its contract")
    return [int(value) for value in indices]


def _ros_time_ns(raw: bytes) -> int:
    if len(raw) != 8:
        raise ValueError("ROS time field must contain sec/nsec uint32 values")
    seconds, nanoseconds = struct.unpack("<II", raw)
    if nanoseconds >= 1_000_000_000:
        raise ValueError("ROS time nanoseconds field is outside [0, 1e9)")
    return int(seconds) * 1_000_000_000 + int(nanoseconds)


def _ros1_topic_message_refs(
    bag_path: Path,
    topic: str = viz.COLOR_TOPIC,
    *,
    expected_message_type: str | None = "sensor_msgs/Image",
) -> tuple[list[Ros1MessageRef], str]:
    """Resolve indexed ROS1 messages to their compressed chunk locations."""

    source = Path(bag_path)
    size = source.stat().st_size
    with source.open("rb") as handle:
        if handle.read(len(viz.ROS_BAG_MAGIC)) != viz.ROS_BAG_MAGIC:
            raise ValueError(f"{source} is not a ROSBAG V2.0 file")
        bag_header, _ = viz._read_ros_record(handle, read_data=False)
        index_position = viz._field_int(bag_header, "index_pos")
        records_start = handle.tell()
        if not (records_start < index_position < size):
            raise ValueError(f"Invalid ROS bag index position in {source}")

        handle.seek(index_position)
        candidates: list[tuple[int, str]] = []
        while handle.tell() < size:
            try:
                header, data = viz._read_ros_record(handle, read_data=True)
            except EOFError:
                break
            if viz._field_int(header, "op") != viz.ROS_OP_CONNECTION:
                continue
            raw_topic = header.get("topic", b"").decode("utf-8", errors="replace")
            if raw_topic != topic:
                continue
            connection_fields = viz._ros_header_fields(data)
            candidates.append(
                (
                    viz._field_int(header, "conn"),
                    connection_fields.get("type", b"").decode(
                        "utf-8", errors="replace"
                    ),
                )
            )
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one {topic!r} connection in {source}, found {len(candidates)}"
            )
        connection_id, message_type = candidates[0]
        if expected_message_type is not None and message_type != expected_message_type:
            raise ValueError(
                f"Expected {expected_message_type} for {topic!r}, found {message_type!r}"
            )

        refs: list[Ros1MessageRef] = []
        current_chunk: tuple[int, int, str, int] | None = None
        handle.seek(records_start)
        while handle.tell() < index_position:
            header_length_raw = handle.read(4)
            if not header_length_raw:
                break
            if len(header_length_raw) != 4:
                raise ValueError("Truncated ROS bag record header length")
            header_length = struct.unpack("<I", header_length_raw)[0]
            header_raw = handle.read(header_length)
            if len(header_raw) != header_length:
                raise ValueError("Truncated ROS bag record header")
            header = viz._ros_header_fields(header_raw)
            data_length = viz._read_u32(handle)
            data_offset = handle.tell()
            op = viz._field_int(header, "op")
            if op == ROS_OP_CHUNK:
                current_chunk = (
                    data_offset,
                    data_length,
                    header.get("compression", b"").decode(
                        "ascii", errors="replace"
                    ),
                    viz._field_int(header, "size"),
                )
                handle.seek(data_length, 1)
                continue
            if op == viz.ROS_OP_INDEX_DATA and viz._field_int(
                header, "conn"
            ) == connection_id:
                if current_chunk is None:
                    raise ValueError("ROS bag index data appeared before a chunk")
                data = handle.read(data_length)
                count = viz._field_int(header, "count")
                if len(data) != count * 12:
                    raise ValueError("Unexpected ROS1 index-data entry size")
                for offset in range(0, len(data), 12):
                    refs.append(
                        Ros1MessageRef(
                            timestamp_ns=_ros_time_ns(data[offset : offset + 8]),
                            chunk_data_offset=current_chunk[0],
                            chunk_data_length=current_chunk[1],
                            compression=current_chunk[2],
                            uncompressed_size=current_chunk[3],
                            record_offset=struct.unpack_from(
                                "<I", data, offset + 8
                            )[0],
                            connection_id=connection_id,
                        )
                    )
                continue
            handle.seek(data_length, 1)

    if not refs:
        raise ValueError(f"No indexed {topic!r} messages in {source}")
    return refs, message_type


def _decompress_ros1_chunk(
    compressed: bytes,
    compression: str,
    expected_size: int,
) -> bytes:
    if compression == "none":
        output = compressed
    elif compression == "bz2":
        output = bz2.decompress(compressed)
    elif compression == "lz4":
        if lz4_frame is not None:
            output = lz4_frame.decompress(compressed)
        else:
            executable = shutil.which("lz4")
            if executable is None:
                raise RuntimeError(
                    "Python lz4 (declared in requirements.txt) or the lz4 CLI "
                    "is required to verify ROS bag image content"
                )
            completed = subprocess.run(
                [executable, "-q", "-d", "-c"],
                input=compressed,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode != 0:
                message = completed.stderr.decode(
                    "utf-8", errors="replace"
                ).strip()
                raise RuntimeError(
                    f"Could not decompress ROS1 LZ4 chunk: {message}"
                )
            output = completed.stdout
    else:
        raise ValueError(f"Unsupported ROS1 chunk compression {compression!r}")
    if len(output) != expected_size:
        raise ValueError(
            f"ROS1 chunk decoded to {len(output)} bytes, expected {expected_size}"
        )
    return output


def _read_ros_string(payload: bytes, offset: int) -> tuple[str, int]:
    if offset + 4 > len(payload):
        raise ValueError("Truncated ROS string length")
    length = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    stop = offset + length
    if stop > len(payload):
        raise ValueError("Truncated ROS string")
    return payload[offset:stop].decode("utf-8", errors="replace"), stop


def _decode_orbbec_mjpg_image(
    serialized: bytes,
    *,
    message_index: int,
    message_timestamp_ns: int,
) -> BagColorSample:
    """Decode Orbbec's recorded Image wire order, including frame metadata."""

    if len(serialized) < 73:
        raise ValueError("Truncated sensor_msgs/Image payload")
    _, stamp_seconds, stamp_nanoseconds = struct.unpack_from("<III", serialized, 0)
    header_timestamp_ns = (
        int(stamp_seconds) * 1_000_000_000 + int(stamp_nanoseconds)
    )
    if header_timestamp_ns != message_timestamp_ns:
        raise ValueError("Orbbec Image header time disagrees with BAG message time")
    frame_id, offset = _read_ros_string(serialized, 12)
    del frame_id
    if offset + 8 > len(serialized):
        raise ValueError("Truncated sensor_msgs/Image dimensions")
    height, width = struct.unpack_from("<II", serialized, offset)
    encoding, offset = _read_ros_string(serialized, offset + 8)
    if offset + 13 > len(serialized):
        raise ValueError("Truncated sensor_msgs/Image payload header")
    offset += 1  # is_bigendian
    _ = struct.unpack_from("<I", serialized, offset)[0]  # step
    offset += 4
    metadata_size, packed_data_size = struct.unpack_from("<II", serialized, offset)
    offset += 8
    packed_stop = offset + packed_data_size
    if packed_stop + 32 != len(serialized):
        raise ValueError(
            "Orbbec Image packed-data length does not leave four uint64 metadata fields"
        )
    if metadata_size > packed_data_size:
        raise ValueError("Orbbec Image metadata exceeds packed-data size")
    packed_data = serialized[offset:packed_stop]
    jpeg = packed_data[metadata_size:]
    frame_number, device_timestamp_us, _, _ = struct.unpack_from(
        "<QQQQ", serialized, packed_stop
    )
    if not jpeg.startswith(b"\xff\xd8\xff") or not jpeg.endswith(b"\xff\xd9"):
        raise ValueError("Orbbec MJPG payload has no complete JPEG SOI/EOI markers")
    if (message_timestamp_ns + 500) // 1000 != int(device_timestamp_us):
        raise ValueError(
            "Orbbec device timestamp does not match the BAG message time at "
            "microsecond precision"
        )
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("OpenCV could not decode the Orbbec MJPG payload")
    if decoded.shape[:2] != (height, width):
        raise ValueError(
            f"Decoded ROS image shape {decoded.shape[:2]} does not match "
            f"header {(height, width)}"
        )
    if encoding.lower() not in {"mjpg", "jpeg", "jpg"}:
        raise ValueError(f"Expected an MJPG image payload, found {encoding!r}")
    analysis_luma = _analysis_luma(decoded)
    return BagColorSample(
        message_index=message_index,
        message_timestamp_ns=message_timestamp_ns,
        device_timestamp_us=int(device_timestamp_us),
        frame_number=int(frame_number),
        encoding=encoding,
        width=int(width),
        height=int(height),
        jpeg_sha256=_bytes_sha256(jpeg),
        decoded_bgr_sha256=_array_sha256(decoded),
        analysis_luma=analysis_luma,
        analysis_luma_sha256=_array_sha256(analysis_luma),
        dhash64=_dhash64(decoded),
    )


def _read_bag_color_samples(
    bag_path: Path,
    refs: Sequence[Ros1MessageRef],
    sample_indices: Sequence[int],
) -> dict[int, BagColorSample]:
    requested = [int(value) for value in sample_indices]
    if len(set(requested)) != len(requested) or requested != sorted(requested):
        raise ValueError("BAG content sample indices must be unique and ordered")
    if not requested or min(requested) < 0 or max(requested) >= len(refs):
        raise ValueError("BAG content sample index is outside the message range")

    result: dict[int, BagColorSample] = {}
    cached_chunk_key: tuple[int, int] | None = None
    cached_chunk: bytes | None = None
    with Path(bag_path).open("rb") as handle:
        for message_index in requested:
            ref = refs[message_index]
            chunk_key = (ref.chunk_data_offset, ref.chunk_data_length)
            if cached_chunk_key != chunk_key or cached_chunk is None:
                handle.seek(ref.chunk_data_offset)
                compressed = handle.read(ref.chunk_data_length)
                if len(compressed) != ref.chunk_data_length:
                    raise ValueError("Truncated ROS1 compressed chunk")
                cached_chunk = _decompress_ros1_chunk(
                    compressed,
                    ref.compression,
                    ref.uncompressed_size,
                )
                cached_chunk_key = chunk_key
            nested = BytesIO(cached_chunk)
            nested.seek(ref.record_offset)
            header, serialized = viz._read_ros_record(nested, read_data=True)
            if viz._field_int(header, "op") != ROS_OP_MESSAGE_DATA:
                raise ValueError("ROS1 index offset does not point to message data")
            if viz._field_int(header, "conn") != ref.connection_id:
                raise ValueError("ROS1 index offset points to a different connection")
            indexed_timestamp_ns = _ros_time_ns(header["time"])
            if indexed_timestamp_ns != ref.timestamp_ns:
                raise ValueError("ROS1 message timestamp disagrees with INDEX_DATA")
            result[message_index] = _decode_orbbec_mjpg_image(
                serialized,
                message_index=message_index,
                message_timestamp_ns=ref.timestamp_ns,
            )
    return result


def _decode_mp4_frames(
    video_path: Path,
    frame_indices: Sequence[int],
) -> dict[int, Mp4FrameSample]:
    requested = sorted(set(int(value) for value in frame_indices))
    if not requested or requested[0] < 0:
        raise ValueError("MP4 frame indices must be non-empty and non-negative")
    targets = set(requested)
    result: dict[int, Mp4FrameSample] = {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_index = 0
    try:
        while targets and frame_index <= requested[-1]:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"MP4 decoding stopped at frame {frame_index} during content QA"
                )
            if frame_index in targets:
                analysis_luma = _analysis_luma(frame)
                result[frame_index] = Mp4FrameSample(
                    frame_index=frame_index,
                    width=int(frame.shape[1]),
                    height=int(frame.shape[0]),
                    decoded_bgr_sha256=_array_sha256(frame),
                    analysis_luma=analysis_luma,
                    analysis_luma_sha256=_array_sha256(analysis_luma),
                    dhash64=_dhash64(frame),
                )
                targets.remove(frame_index)
            frame_index += 1
    finally:
        capture.release()
    if targets:
        raise RuntimeError(f"MP4 content QA did not decode frames {sorted(targets)}")
    return result


def _analysis_luma(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Content QA expects a BGR frame")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(
        gray,
        FRAME_CONTENT_ANALYSIS_SIZE,
        interpolation=cv2.INTER_AREA,
    )


def _dhash64(frame: np.ndarray) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (resized[:, 1:] > resized[:, :-1]).reshape(-1)
    value = 0
    for bit_index, bit in enumerate(bits):
        value |= int(bool(bit)) << bit_index
    return value


def _numeric_distribution(values: Sequence[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data):
        raise ValueError("Cannot summarize an empty numeric distribution")
    return {
        "count": int(len(data)),
        "min": float(np.min(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "max": float(np.max(data)),
    }


def _compare_sampled_frame_content(
    bag_samples: Mapping[int, BagColorSample],
    mp4_frames: Mapping[int, Mp4FrameSample],
    sample_indices: Sequence[int],
    *,
    total_frame_count: int,
    requested_sample_count: int = FRAME_CONTENT_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Evaluate a candidate BAG-message-index -> MP4-frame-index mapping."""

    indices = [int(value) for value in sample_indices]
    if not indices or len(indices) != len(set(indices)) or indices != sorted(indices):
        raise ValueError("Content QA indices must be non-empty, unique, and ordered")
    if indices[0] < 0 or indices[-1] >= total_frame_count:
        raise ValueError("Content QA sample lies outside total_frame_count")
    missing_bag = [value for value in indices if value not in bag_samples]
    missing_mp4 = [value for value in indices if value not in mp4_frames]
    if missing_bag or missing_mp4:
        raise ValueError(
            f"Missing sampled frames: bag={missing_bag}, mp4={missing_mp4}"
        )

    rows: list[dict[str, Any]] = []
    same_mse_values: list[float] = []
    same_mae_values: list[float] = []
    same_psnr_values: list[float] = []
    same_dhash_hamming_values: list[int] = []
    offset_values: dict[int, list[float]] = {
        offset: [] for offset in FRAME_CONTENT_OFFSETS
    }
    same_index_best_count = 0
    distinguishable_count = 0
    distinguishable_same_index_best_count = 0
    dimensions_match = True
    for index in indices:
        bag_sample = bag_samples[index]
        same_mp4 = mp4_frames[index]
        dimensions_match &= (
            bag_sample.width == same_mp4.width
            and bag_sample.height == same_mp4.height
        )
        bag_luma_u8 = bag_sample.analysis_luma
        bag_luma = bag_luma_u8.astype(np.float32)
        candidate_mse: dict[int, float] = {}
        candidate_mae: dict[int, float] = {}
        candidate_hashes: dict[int, str] = {}
        for offset in FRAME_CONTENT_OFFSETS:
            candidate_index = index + offset
            frame = mp4_frames.get(candidate_index)
            if frame is None:
                continue
            mp4_luma_u8 = frame.analysis_luma
            difference = bag_luma - mp4_luma_u8.astype(np.float32)
            candidate_mae[offset] = float(np.mean(np.abs(difference)))
            candidate_mse[offset] = float(np.mean(difference * difference))
            candidate_hashes[offset] = frame.analysis_luma_sha256
        if 0 not in candidate_mse:
            raise ValueError(f"No same-index MP4 comparison for sample {index}")

        ranking = sorted(
            candidate_mse,
            key=lambda offset: (candidate_mse[offset], abs(offset), offset),
        )
        best_offset = ranking[0]
        best_mse = candidate_mse[best_offset]
        if len(ranking) > 1:
            runner_up_mse = candidate_mse[ranking[1]]
            separation_ratio = runner_up_mse / max(best_mse, 1e-12)
        else:
            separation_ratio = 1.0
        distinguishable = separation_ratio >= 1.05
        same_index_best = best_offset == 0
        same_index_best_count += int(same_index_best)
        distinguishable_count += int(distinguishable)
        distinguishable_same_index_best_count += int(
            distinguishable and same_index_best
        )
        same_mse_values.append(candidate_mse[0])
        same_mae_values.append(candidate_mae[0])
        same_psnr = (
            10.0 * math.log10(255.0**2 / candidate_mse[0])
            if candidate_mse[0] > 0.0
            else 120.0
        )
        same_psnr_values.append(same_psnr)
        dhash_hamming = (
            bag_sample.dhash64 ^ mp4_frames[index].dhash64
        ).bit_count()
        same_dhash_hamming_values.append(dhash_hamming)
        if all(offset in candidate_mse for offset in FRAME_CONTENT_OFFSETS):
            for offset in FRAME_CONTENT_OFFSETS:
                offset_values[offset].append(candidate_mse[offset])
        rows.append(
            {
                "bag_message_index": index,
                "candidate_mp4_frame_index": index,
                "bag_message_timestamp_ns": bag_sample.message_timestamp_ns,
                "bag_device_timestamp_us": bag_sample.device_timestamp_us,
                "bag_frame_number": bag_sample.frame_number,
                "bag_encoding": bag_sample.encoding,
                "bag_jpeg_sha256": bag_sample.jpeg_sha256,
                "bag_decoded_bgr_sha256": bag_sample.decoded_bgr_sha256,
                "bag_analysis_luma_sha256": (
                    bag_sample.analysis_luma_sha256
                ),
                "mp4_decoded_bgr_sha256": (
                    mp4_frames[index].decoded_bgr_sha256
                ),
                "mp4_analysis_luma_sha256": candidate_hashes[0],
                "same_index_luma_mae": candidate_mae[0],
                "same_index_luma_mse": candidate_mse[0],
                "same_index_luma_psnr_db": same_psnr,
                "same_index_dhash64_hamming": dhash_hamming,
                "neighbor_candidate_luma_mse": {
                    str(offset): candidate_mse[offset]
                    for offset in sorted(candidate_mse)
                },
                "best_candidate_offset_frames": best_offset,
                "best_vs_runner_up_mse_ratio": separation_ratio,
                "distinguishable_at_ratio_1_05": distinguishable,
                "same_index_is_best_neighbor_candidate": same_index_best,
            }
        )

    offset_summary = {
        str(offset): {
            "compared_samples": len(offset_values[offset]),
            "mean_luma_mse": float(np.mean(offset_values[offset])),
        }
        for offset in FRAME_CONTENT_OFFSETS
        if offset_values[offset]
    }
    if set(offset_summary) != {str(value) for value in FRAME_CONTENT_OFFSETS}:
        raise ValueError("Not enough interior samples for off-by-one content QA")
    best_global_offset = min(
        FRAME_CONTENT_OFFSETS,
        key=lambda offset: (
            offset_summary[str(offset)]["mean_luma_mse"],
            abs(offset),
            offset,
        ),
    )
    if requested_sample_count < 0:
        raise ValueError("requested_sample_count must be non-negative")
    full_frame_evidence = requested_sample_count == 0
    expected_count = (
        total_frame_count
        if full_frame_evidence
        else min(
            max(requested_sample_count, FRAME_CONTENT_SAMPLE_COUNT),
            total_frame_count,
        )
    )
    sample_count_ok = len(indices) >= expected_count
    covers_span = indices[0] == 0 and indices[-1] == total_frame_count - 1
    absolute_error_ok = max(same_mse_values) <= FRAME_CONTENT_MAX_LUMA_MSE
    psnr_ok = min(same_psnr_values) >= FRAME_CONTENT_MIN_PSNR_DB
    dhash_ok = max(same_dhash_hamming_values) <= FRAME_CONTENT_MAX_DHASH_HAMMING
    same_best_fraction = same_index_best_count / len(indices)
    required_distinguishable = min(
        FRAME_CONTENT_MIN_DISTINGUISHABLE, len(indices)
    )
    enough_distinguishable = distinguishable_count >= required_distinguishable
    distinguishable_all_same = (
        distinguishable_same_index_best_count == distinguishable_count
    )
    acceptance = {
        "requested_count_and_at_least_31_samples_or_all_frames_when_shorter": (
            sample_count_ok
        ),
        "samples_include_first_and_last_frame": covers_span,
        "same_index_dimensions_match": dimensions_match,
        "same_index_max_luma_mse_at_most_6": absolute_error_ok,
        "same_index_min_luma_psnr_db_at_least_39": psnr_ok,
        "same_index_max_dhash64_hamming_at_most_4": dhash_ok,
        "same_index_is_best_global_offset_within_plus_minus_2": (
            best_global_offset == 0
        ),
        "same_index_best_neighbor_fraction_at_least_0_90": (
            same_best_fraction >= 0.90
        ),
        "at_least_24_samples_are_neighbor_distinguishable": (
            enough_distinguishable
        ),
        "all_distinguishable_samples_prefer_same_index": (
            distinguishable_all_same
        ),
    }
    acceptance["pass"] = bool(all(acceptance.values()))
    return {
        "evidence_scope": (
            "full-frame content evidence"
            if full_frame_evidence
            else "fixed sampled content evidence; not an all-frame pixel comparison"
        ),
        "full_frame_content_comparison": full_frame_evidence,
        "candidate_mapping": "ROS BAG color message index i -> MP4 decoded frame index i",
        "sampling_method": (
            "every source frame in index order"
            if full_frame_evidence
            else (
                "unique rint(linspace(0, frame_count - 1, requested_sample_count)); "
                "includes first, last, and uniform full-duration coverage"
            )
        ),
        "requested_sample_count": requested_sample_count,
        "sample_count": len(indices),
        "sample_indices": indices,
        "analysis_method": (
            "decode Orbbec MJPG from indexed ROS1 chunks and MP4 sequentially; "
            "compare 240x135 grayscale with MAE/MSE, PSNR, dHash64, and test "
            "offsets -2/-1/0/+1/+2"
        ),
        "analysis_width": FRAME_CONTENT_ANALYSIS_SIZE[0],
        "analysis_height": FRAME_CONTENT_ANALYSIS_SIZE[1],
        "same_index_luma_mae": _numeric_distribution(same_mae_values),
        "same_index_luma_mse": _numeric_distribution(same_mse_values),
        "same_index_luma_psnr_db": _numeric_distribution(same_psnr_values),
        "same_index_dhash64_hamming": _numeric_distribution(
            same_dhash_hamming_values
        ),
        "neighbor_offset_aggregate": offset_summary,
        "best_global_offset_frames": best_global_offset,
        "same_index_best_neighbor_count": same_index_best_count,
        "same_index_best_neighbor_fraction": same_best_fraction,
        "distinguishable_sample_count": distinguishable_count,
        "required_distinguishable_sample_count": required_distinguishable,
        "distinguishable_same_index_best_count": (
            distinguishable_same_index_best_count
        ),
        "samples": rows,
        "acceptance": acceptance,
    }


def sampled_bag_mp4_content_validation(
    prepared: PreparedMocapVideo,
    *,
    sample_count: int = FRAME_CONTENT_SAMPLE_COUNT,
) -> dict[str, Any]:
    bag_path = (
        prepared.segment_dir
        / "原始BAG与内参"
        / "camera_1_rgb_depth.bag"
    )
    refs, message_type = _ros1_topic_message_refs(bag_path)
    if len(refs) != prepared.video.frame_count:
        raise ValueError(
            f"Indexed BAG color messages ({len(refs)}) do not match MP4 frames "
            f"({prepared.video.frame_count}) during content QA"
        )
    sample_indices = _fixed_span_sample_indices(
        prepared.video.frame_count,
        sample_count,
    )
    bag_samples = _read_bag_color_samples(bag_path, refs, sample_indices)
    mp4_indices = sorted(
        {
            candidate
            for index in sample_indices
            for candidate in (
                index - 2,
                index - 1,
                index,
                index + 1,
                index + 2,
            )
            if 0 <= candidate < prepared.video.frame_count
        }
    )
    mp4_frames = _decode_mp4_frames(prepared.video.path, mp4_indices)
    result = _compare_sampled_frame_content(
        bag_samples,
        mp4_frames,
        sample_indices,
        total_frame_count=prepared.video.frame_count,
        requested_sample_count=sample_count,
    )
    reference_us = np.asarray(
        [(ref.timestamp_ns + 500) // 1000 for ref in refs], dtype=np.int64
    )
    prepared_us = np.rint(prepared.rgb_device_s * 1e6).astype(np.int64)
    timestamps_match = bool(np.array_equal(reference_us, prepared_us))
    reference_ns = np.asarray([ref.timestamp_ns for ref in refs], dtype=np.int64)
    refs_unique = len(np.unique(reference_ns)) == len(reference_ns)
    refs_ordered = bool(np.all(np.diff(reference_ns) > 0))
    sampled_frame_numbers = np.asarray(
        [bag_samples[index].frame_number for index in sample_indices],
        dtype=np.int64,
    )
    sampled_device_us = np.asarray(
        [bag_samples[index].device_timestamp_us for index in sample_indices],
        dtype=np.int64,
    )
    sampled_metadata_ordered = bool(
        np.all(np.diff(sampled_frame_numbers) > 0)
        and np.all(np.diff(sampled_device_us) > 0)
    )
    sampled_message_indices_exact = all(
        bag_samples[index].message_index == index for index in sample_indices
    )
    result["ros_topic"] = viz.COLOR_TOPIC
    result["ros_connection_id"] = refs[0].connection_id
    result["ros_message_type"] = message_type
    result["ros_chunk_compressions_observed"] = sorted(
        {refs[index].compression for index in sample_indices}
    )
    result["bag_index_refs_match_prepared_timestamp_sequence"] = timestamps_match
    result["bag_index_message_timestamps_unique"] = refs_unique
    result["bag_index_message_timestamps_strictly_increasing"] = refs_ordered
    result["sampled_frame_number_and_device_timestamp_strictly_increasing"] = (
        sampled_metadata_ordered
    )
    result["sampled_message_indices_equal_requested_indices"] = (
        sampled_message_indices_exact
    )
    result["acceptance"][
        "bag_index_refs_match_prepared_timestamp_sequence"
    ] = timestamps_match
    result["acceptance"]["bag_index_message_timestamps_unique"] = refs_unique
    result["acceptance"][
        "bag_index_message_timestamps_strictly_increasing"
    ] = refs_ordered
    result["acceptance"][
        "sampled_frame_number_and_device_timestamp_strictly_increasing"
    ] = sampled_metadata_ordered
    result["acceptance"][
        "sampled_message_indices_equal_requested_indices"
    ] = sampled_message_indices_exact
    result["acceptance"]["pass"] = bool(
        all(
            value
            for key, value in result["acceptance"].items()
            if key != "pass"
        )
    )
    return result


def _distribution_ms(values_s: np.ndarray) -> dict[str, float]:
    values = np.asarray(values_s, dtype=np.float64) * 1000.0
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("Cannot summarize an empty timestamp distribution")
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _timecode_bag_index_match(
    timecode_timestamps_us: Sequence[int],
    bag_timestamps_us: Sequence[int],
) -> dict[str, Any]:
    """Check timestamp identity without hiding duplicates or row reordering."""

    timecode = [int(value) for value in timecode_timestamps_us]
    bag = [int(value) for value in bag_timestamps_us]
    if not timecode or not bag:
        raise ValueError("Timecode and BAG timestamp sequences must be non-empty")
    timecode_duplicate_count = len(timecode) - len(set(timecode))
    bag_duplicate_count = len(bag) - len(set(bag))
    timecode_ordered = all(
        current > previous for previous, current in zip(timecode, timecode[1:])
    )
    bag_ordered = all(
        current > previous for previous, current in zip(bag, bag[1:])
    )

    bag_indices_by_timestamp: dict[int, list[int]] = {}
    for index, timestamp in enumerate(bag):
        bag_indices_by_timestamp.setdefault(timestamp, []).append(index)
    first_us, last_us = min(bag), max(bag)
    overlap = [
        timestamp
        for timestamp in timecode
        if first_us <= timestamp <= last_us
    ]
    exact_indices: list[int] = []
    exact = 0
    each_matches_once = True
    for timestamp in overlap:
        candidates = bag_indices_by_timestamp.get(timestamp, [])
        exact += int(bool(candidates))
        each_matches_once &= len(candidates) == 1
        if len(candidates) == 1:
            exact_indices.append(candidates[0])
    indices_unique = len(exact_indices) == len(set(exact_indices))
    indices_ordered = all(
        current > previous
        for previous, current in zip(exact_indices, exact_indices[1:])
    )
    one_to_one = bool(
        overlap
        and exact == len(overlap)
        and len(exact_indices) == len(overlap)
        and each_matches_once
        and indices_unique
        and indices_ordered
    )
    passed = bool(
        timecode_duplicate_count == 0
        and bag_duplicate_count == 0
        and timecode_ordered
        and bag_ordered
        and one_to_one
    )
    return {
        "timecode_total_rows": len(timecode),
        "timecode_duplicate_timestamp_count": timecode_duplicate_count,
        "timecode_timestamps_strictly_increasing": timecode_ordered,
        "bag_message_count": len(bag),
        "bag_duplicate_timestamp_count": bag_duplicate_count,
        "bag_timestamps_strictly_increasing": bag_ordered,
        "timecode_rows_inside_bag_interval": len(overlap),
        "exact_bag_index_timestamp_matches": exact,
        "exact_match_fraction": exact / len(overlap) if overlap else 0.0,
        "matched_bag_message_indices": exact_indices,
        "matched_bag_message_indices_unique": indices_unique,
        "matched_bag_message_indices_strictly_increasing": indices_ordered,
        "each_timecode_row_matches_exactly_one_unique_bag_index": one_to_one,
        "pass": passed,
    }


def _camera_timecode_bag_match(prepared: PreparedMocapVideo) -> dict[str, Any]:
    path = prepared.segment_dir / "同步校验" / "camera_cmavatar_alignment.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    timecode_us = [int(row["color_device_timestamp_us"]) for row in rows]
    bag_us = np.rint(prepared.rgb_device_s * 1e6).astype(np.int64)
    return _timecode_bag_index_match(timecode_us, bag_us.tolist())


def _centered_smooth(values: np.ndarray, width: int) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if width <= 1:
        return data.copy()
    if width % 2 == 0:
        raise ValueError("Centered smoothing width must be odd")
    pad = width // 2
    padded = np.pad(data, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def _video_motion_energy(
    video_path: Path,
    first: int,
    stop: int,
    *,
    analysis_size: tuple[int, int] = (240, 135),
    roi_xyxy: tuple[int, int, int, int] = (40, 35, 120, 115),
) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    previous: np.ndarray | None = None
    energy: list[float] = []
    x0, y0, x1, y1 = roi_xyxy
    try:
        for source_frame in range(first, stop):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"RGB decoding stopped during motion QA at frame {source_frame}"
                )
            small = cv2.resize(frame, analysis_size, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            crop = gray[y0:y1, x0:x1]
            if crop.size == 0:
                raise ValueError("Motion-validation ROI is empty")
            if previous is not None:
                energy.append(float(np.mean(np.abs(crop - previous))))
            previous = crop
    finally:
        capture.release()
    return np.asarray(energy, dtype=np.float64)


def _mocap_projected_speed(
    prepared: PreparedMocapVideo,
    target_times_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points, valid = viz.interpolate_series(
        prepared.mocap.times_s,
        prepared.mocap.points_mm,
        target_times_s,
        max_gap_s=prepared.max_interpolation_gap_ms * 1e-3,
    )
    translation = np.asarray(
        prepared.mocap_world_translation_mm,
        dtype=np.float64,
    )
    if np.any(translation != 0.0):
        points = points + translation.reshape(1, 1, 1, 3)
    pixels, positive = viz.project_world_to_rgb(points, prepared.calibration)
    displacement = np.linalg.norm(np.diff(pixels, axis=0), axis=-1)
    speed = np.median(displacement, axis=(1, 2))
    pair_valid = (
        valid[:-1]
        & valid[1:]
        & np.all(positive[:-1], axis=(1, 2))
        & np.all(positive[1:], axis=(1, 2))
        & np.isfinite(speed)
    )
    return speed, pair_valid


def _motion_correlation(
    image_energy: np.ndarray,
    mocap_speed: np.ndarray,
    valid: np.ndarray,
    *,
    smoothing_frames: int,
) -> tuple[float, int]:
    image = _centered_smooth(image_energy, smoothing_frames)
    motion = _centered_smooth(mocap_speed, smoothing_frames)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(image) & np.isfinite(motion)
    edge = smoothing_frames // 2
    if edge:
        mask[:edge] = False
        mask[-edge:] = False
    if np.count_nonzero(mask) < 30:
        return float("nan"), int(np.count_nonzero(mask))
    selected_image = image[mask]
    selected_motion = motion[mask]
    if np.std(selected_image) <= 1e-12 or np.std(selected_motion) <= 1e-12:
        return float("nan"), int(np.count_nonzero(mask))
    return (
        float(np.corrcoef(selected_image, selected_motion)[0, 1]),
        int(np.count_nonzero(mask)),
    )


def _search_motion_adjustment(
    prepared: PreparedMocapVideo,
    image_energy: np.ndarray,
    base_times_s: np.ndarray,
    adjustments_ms: Sequence[int],
    *,
    smoothing_frames: int,
) -> dict[str, float | int]:
    best: tuple[float, int, int] | None = None
    for adjustment_ms in adjustments_ms:
        speed, valid = _mocap_projected_speed(
            prepared,
            base_times_s + float(adjustment_ms) * 1e-3,
        )
        correlation, count = _motion_correlation(
            image_energy,
            speed,
            valid,
            smoothing_frames=smoothing_frames,
        )
        if np.isfinite(correlation) and (best is None or correlation > best[0]):
            best = (correlation, int(adjustment_ms), count)
    if best is None:
        raise ValueError("No finite visual/MOCAP motion correlation")
    return {
        "best_timestamp_adjustment_ms": best[1],
        "best_correlation": best[0],
        "compared_motion_steps": best[2],
    }


def visual_motion_validation(
    prepared: PreparedMocapVideo,
    first: int,
    stop: int,
) -> dict[str, Any]:
    """Validate exposure correction without feeding visual lag back into the fit."""

    image_energy = _video_motion_energy(prepared.video.path, first, stop)
    corrected_times = prepared.cmavatar_s[first:stop]
    _, poll_times, poll_inside = prepared.clock_mapping.map_poll_time(
        prepared.rgb_device_s[first:stop]
    )
    if not np.all(poll_inside):
        raise ValueError("Published clip falls outside callback-time clock mapping")
    smoothing_frames = 7

    corrected_speed, corrected_valid = _mocap_projected_speed(
        prepared, corrected_times
    )
    corrected_zero, corrected_count = _motion_correlation(
        image_energy,
        corrected_speed,
        corrected_valid,
        smoothing_frames=smoothing_frames,
    )
    poll_speed, poll_valid = _mocap_projected_speed(prepared, poll_times)
    poll_zero, poll_count = _motion_correlation(
        image_energy,
        poll_speed,
        poll_valid,
        smoothing_frames=smoothing_frames,
    )
    corrected_search = _search_motion_adjustment(
        prepared,
        image_energy,
        corrected_times,
        range(-100, 101),
        smoothing_frames=smoothing_frames,
    )
    poll_search = _search_motion_adjustment(
        prepared,
        image_energy,
        poll_times,
        range(-350, -99),
        smoothing_frames=smoothing_frames,
    )
    one_frame_ms = 1000.0 / prepared.video.fps
    gain = corrected_zero - poll_zero
    residual = int(corrected_search["best_timestamp_adjustment_ms"])
    return {
        "purpose": (
            "QA-only cross-correlation; the visual residual is not written back "
            "as a fitted per-take time offset"
        ),
        "image_signal": (
            "mean absolute grayscale frame difference in fixed 240x135 ROI "
            "x=40:120,y=35:115"
        ),
        "mocap_signal": "median projected-pixel speed over both hands and all 42 joints",
        "centered_smoothing_frames": smoothing_frames,
        "correlation_definition": (
            "corr(video_energy[t], mocap_speed_at(camera_time[t] + adjustment))"
        ),
        "legacy_callback_time": {
            "zero_adjustment_correlation": poll_zero,
            "zero_adjustment_compared_motion_steps": poll_count,
            **poll_search,
        },
        "acquisition_time_corrected": {
            "zero_adjustment_correlation": corrected_zero,
            "zero_adjustment_compared_motion_steps": corrected_count,
            **corrected_search,
        },
        "zero_adjustment_correlation_gain": gain,
        "one_video_frame_ms": one_frame_ms,
        "acceptance": {
            "corrected_zero_correlation_at_least_0_80": corrected_zero >= 0.80,
            "correlation_gain_at_least_0_20": gain >= 0.20,
            "residual_best_adjustment_within_one_video_frame": (
                abs(residual) <= one_frame_ms
            ),
            "pass": bool(
                corrected_zero >= 0.80
                and gain >= 0.20
                and abs(residual) <= one_frame_ms
            ),
        },
    }


def frame_mapping_rows(
    prepared: PreparedMocapVideo,
    first: int,
    stop: int,
) -> list[dict[str, int | float]]:
    if first < 0 or stop > prepared.video.frame_count or first >= stop:
        raise ValueError(f"Invalid frame interval [{first}, {stop})")
    if not np.all(prepared.mocap_valid[first:stop]):
        raise ValueError("Frame-map interval contains an invalid MOCAP frame")

    rows: list[dict[str, int | float]] = []
    for output_frame, source_frame in enumerate(range(first, stop)):
        low = int(prepared.interpolation.low_indices[source_frame])
        high = int(prepared.interpolation.high_indices[source_frame])
        projected, positive = viz.project_world_to_rgb(
            prepared.mocap_mm[source_frame, :, 0], prepared.calibration
        )
        in_frame = (
            positive
            & np.all(np.isfinite(projected), axis=1)
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] < prepared.video.width)
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] < prepared.video.height)
        )
        rows.append(
            {
                "output_frame": output_frame,
                "source_video_frame": source_frame,
                "source_video_nominal_pts_s": source_frame / prepared.video.fps,
                "source_bag_elapsed_s": float(
                    prepared.rgb_device_s[source_frame] - prepared.rgb_device_s[0]
                ),
                "matched_clip_bag_elapsed_s": float(
                    prepared.rgb_device_s[source_frame] - prepared.rgb_device_s[first]
                ),
                "rgb_device_timestamp_us": int(
                    round(prepared.rgb_device_s[source_frame] * 1e6)
                ),
                "cmavatar_target_timestamp_ns": int(
                    round(prepared.cmavatar_s[source_frame] * 1e9)
                ),
                "mocap_low_frame_counter": int(prepared.mocap.frame_counters[low]),
                "mocap_high_frame_counter": int(prepared.mocap.frame_counters[high]),
                "mocap_low_timestamp_ns": int(round(prepared.mocap.times_s[low] * 1e9)),
                "mocap_high_timestamp_ns": int(round(prepared.mocap.times_s[high] * 1e9)),
                "interpolation_alpha": float(prepared.interpolation.alpha[source_frame]),
                "bracket_span_ms": float(
                    prepared.interpolation.bracket_span_s[source_frame] * 1000.0
                ),
                "nearest_mocap_sample_delta_ms": float(
                    prepared.interpolation.nearest_delta_s[source_frame] * 1000.0
                ),
                "left_wrist_rgb_x": float(projected[0, 0]),
                "left_wrist_rgb_y": float(projected[0, 1]),
                "left_wrist_inside_frame": int(in_frame[0]),
                "right_wrist_rgb_x": float(projected[1, 0]),
                "right_wrist_rgb_y": float(projected[1, 1]),
                "right_wrist_inside_frame": int(in_frame[1]),
            }
        )
    return rows


def write_frame_mapping(
    path: Path,
    prepared: PreparedMocapVideo,
    first: int,
    stop: int,
) -> Path:
    rows = frame_mapping_rows(prepared, first, stop)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def alignment_metrics(
    prepared: PreparedMocapVideo,
    first: int,
    stop: int,
    *,
    artifacts: Mapping[str, Path] | None = None,
    motion_validation: Mapping[str, Any] | None = None,
    frame_content_validation: Mapping[str, Any] | None = None,
    frame_content_sample_count: int = FRAME_CONTENT_SAMPLE_COUNT,
    draw_rear_mount_proxy: bool = False,
) -> dict[str, Any]:
    if not np.all(prepared.mocap_valid[first:stop]):
        raise ValueError("Metrics interval contains invalid MOCAP frames")
    selected = slice(first, stop)
    pixels, positive = viz.project_world_to_rgb(
        prepared.mocap_mm[selected], prepared.calibration
    )
    finite = np.all(np.isfinite(pixels), axis=-1)
    in_frame = (
        positive
        & finite
        & (pixels[..., 0] >= 0.0)
        & (pixels[..., 0] < prepared.video.width)
        & (pixels[..., 1] >= 0.0)
        & (pixels[..., 1] < prepared.video.height)
    )
    wrist_in_frame = in_frame[:, :, 0]
    clock_model = prepared.sync_report["clock_model"]
    calibration_payload = prepared.calibration.payload
    input_paths = {
        "rgb_video": prepared.video.path,
        "rgb_timestamp_bag": prepared.segment_dir
        / "原始BAG与内参"
        / "camera_1_rgb_depth.bag",
        "clock_mapping": prepared.segment_dir
        / "同步校验"
        / "camera_cmavatar_alignment.csv",
        "common_interval_sync_report": prepared.segment_dir
        / "同步校验"
        / "common_interval_sync_report.json",
        "mocap_human": prepared.mocap_path,
        "camera_to_world": prepared.calibration.path,
    }
    if prepared.mocap_position_source == MOCAP_POSITION_SOURCE_SKELETON_BVH:
        if set(prepared.mocap_position_paths) != {"left", "right"}:
            raise ValueError(
                "skeleton-bvh metrics require exactly left/right source paths"
            )
        input_paths.update(
            {
                "mocap_skeleton_0_bvh": prepared.mocap_position_paths["left"],
                "mocap_skeleton_1_bvh": prepared.mocap_position_paths["right"],
            }
        )
    inputs = {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in input_paths.items()
    }
    if prepared.mocap_position_source == MOCAP_POSITION_SOURCE_SKELETON_BVH:
        expected_hashes = prepared.mocap_position_contract.get(
            "bvh_source_sha256"
        )
        if not isinstance(expected_hashes, Mapping) or set(expected_hashes) != {
            "left",
            "right",
        }:
            raise ValueError(
                "skeleton-bvh position contract lacks exact left/right source hashes"
            )
        for side, input_name in (
            ("left", "mocap_skeleton_0_bvh"),
            ("right", "mocap_skeleton_1_bvh"),
        ):
            if inputs[input_name]["sha256"] != expected_hashes[side]:
                raise ValueError(
                    f"{side} Skeleton BVH changed after FK preparation; refusing "
                    "to write misleading provenance"
                )
    artifact_payload: dict[str, Any] = {}
    for name, path in (artifacts or {}).items():
        if Path(path).is_file():
            artifact_payload[name] = {
                "path": str(Path(path).resolve()),
                "bytes": Path(path).stat().st_size,
                "sha256": _sha256(path),
            }

    count = stop - first
    bracket = _distribution_ms(prepared.interpolation.bracket_span_s[selected])
    nearest = _distribution_ms(prepared.interpolation.nearest_delta_s[selected])
    callback_latency = _distribution_ms(prepared.clock_mapping.capture_to_poll_s)
    device_match = _camera_timecode_bag_match(prepared)
    content_qa = dict(
        frame_content_validation
        if frame_content_validation is not None
        else sampled_bag_mp4_content_validation(
            prepared,
            sample_count=frame_content_sample_count,
        )
    )
    motion_qa = dict(
        motion_validation
        if motion_validation is not None
        else visual_motion_validation(prepared, first, stop)
    )
    fully_mapped = bool(np.all(prepared.mocap_valid[selected]))
    temporal_pass = bool(
        fully_mapped
        and bracket["max"] <= prepared.max_interpolation_gap_ms
        and float(clock_model["estimated_clock_uncertainty_ms"]) <= 1.0
        and device_match["pass"]
        and content_qa["acceptance"]["pass"]
        and motion_qa["acceptance"]["pass"]
    )
    world_translation = [
        float(value) for value in prepared.mocap_world_translation_mm
    ]
    world_translation_enabled = any(value != 0.0 for value in world_translation)
    return {
        "schema": ALIGNMENT_SCHEMA,
        "artifact_type": "evaluation_metrics",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "segment": prepared.segment_dir.name,
        "status": (
            "frame_complete_timestamp_mapping_with_content_validated_rgb_index_"
            "correspondence_and_metric_reprojection"
        ),
        "claim": (
            "Every published clip frame has a valid camera-timestamp -> CMAvatar "
            "interpolation bracket. The BAG-message -> MP4-frame index mapping is "
            "accepted only after decoded image-content validation, and joints are "
            "projected with the delivered camera/world calibration. Frame-complete "
            "does not mean independently measured <=1 ms physical synchronization."
            + (
                " A candidate MOCAP-world origin translation is active; it is "
                "explicitly not a measured ground-truth calibration."
                if world_translation_enabled
                else ""
            )
        ),
        "inputs": inputs,
        "mocap_position_source": {
            "selected": prepared.mocap_position_source,
            "human_cma_supplies_timestamps_and_frame_counters": True,
            "contract": dict(prepared.mocap_position_contract),
        },
        "video_frame_contract": {
            "source_mp4_frames": prepared.video.frame_count,
            "source_rgb_bag_messages": len(prepared.rgb_device_s),
            "bag_message_count_equals_mp4_frame_count": (
                len(prepared.rgb_device_s) == prepared.video.frame_count
            ),
            "bag_message_i_equals_mp4_frame_i_content_validated": bool(
                content_qa["acceptance"]["pass"]
            ),
            "bag_mp4_frame_content_validation": content_qa,
            "timecode_to_bag_timestamp_check": device_match,
            "source_frame_first": first,
            "source_frame_last_inclusive": stop - 1,
            "published_clip_frames": count,
            "published_clip_all_frames_mocap_valid": fully_mapped,
            "published_clip_coverage_percent": 100.0 if fully_mapped else 0.0,
            "trimmed_prefix_frames": first,
            "trimmed_suffix_frames": prepared.video.frame_count - stop,
            "nominal_fps": prepared.video.fps,
            "width": prepared.video.width,
            "height": prepared.video.height,
        },
        "temporal_alignment": {
            "absolute_time_key": "cmavatar_windows_estimated_ns",
            "mocap_index_key": "FrameCounter",
            "mapping": (
                "RGB device acquisition timestamp -> remove measured callback latency "
                "row by row -> CMAvatar acquisition time -> linear 120 Hz "
                f"{prepared.mocap_position_source} position interpolation"
            ),
            "camera_capture_to_thor_poll_latency_ms": callback_latency,
            "clock_model_estimated_uncertainty_ms": float(
                clock_model["estimated_clock_uncertainty_ms"]
            ),
            "clock_uncertainty_interpretation": (
                "Delivered software clock-model estimate; it is not an independent "
                "per-frame exposure-to-MOCAP timing-error measurement and does not "
                "make the frame-complete mapping a <=1 ms ground-truth claim."
            ),
            "mocap_bracket_span_ms": bracket,
            "nearest_mocap_sample_delta_ms": nearest,
            "mocap_sample_timebase": prepared.mocap.timestamp_fit,
            "max_allowed_bracket_span_ms": prepared.max_interpolation_gap_ms,
            "interpolated_at_camera_timestamp_not_nearest_frame_rounded": True,
            "visual_motion_validation": motion_qa,
            "acceptance": {
                "all_published_frames_valid": fully_mapped,
                "delivered_clock_model_estimate_at_most_1ms": (
                    float(clock_model["estimated_clock_uncertainty_ms"]) <= 1.0
                ),
                "frame_complete_is_not_claimed_as_measured_sub_millisecond_gt": True,
                "bracket_max_at_most_configured_gate": (
                    bracket["max"] <= prepared.max_interpolation_gap_ms
                ),
                "timecode_rows_match_bag_index_timestamps": device_match["pass"],
                "bag_mp4_decoded_content_correspondence_pass": content_qa[
                    "acceptance"
                ]["pass"],
                "visual_motion_validation_pass": motion_qa["acceptance"]["pass"],
                "pass": temporal_pass,
            },
        },
        "spatial_alignment": {
            "calibration_schema": calibration_payload["schema"],
            "calibration_status": calibration_payload["status"],
            "method": calibration_payload["method"],
            "fixed_camera_assumption": calibration_payload["fixed_camera_assumption"],
            "camera_consistency_passed": calibration_payload["camera_consistency"]["passed"],
            "world_to_color_camera": prepared.calibration.world_to_color.tolist(),
            "rgb_intrinsics_fx_fy_cx_cy": prepared.calibration.intrinsics.tolist(),
            "rgb_distortion_orbbec_k1_k2_k3_k4_k5_k6_p1_p2": (
                prepared.calibration.distortion.tolist()
            ),
            "table_plane_residual_mm": {
                "p50": float(
                    calibration_payload["table_plane_in_depth_camera"]["residual_mm_p50"]
                ),
                "p95": float(
                    calibration_payload["table_plane_in_depth_camera"]["residual_mm_p95"]
                ),
            },
            "projected_joint_samples": int(in_frame.size),
            "positive_depth_fraction": float(np.mean(positive)),
            "inside_rgb_frame_fraction": float(np.mean(in_frame)),
            "wrist_inside_rgb_frame_fraction": float(np.mean(wrist_in_frame)),
            "mocap_world_translation": {
                "enabled": world_translation_enabled,
                "profile": (
                    MOCAP_WORLD_TRANSLATION_CANDIDATE_PROFILE
                    if world_translation_enabled
                    else "identity_default"
                ),
                "translation_xyz_mm": world_translation,
                "applied_after_timestamp_interpolation_to_all_42_joints": True,
                "changes_joint_relative_geometry": False,
                "ground_truth_claimed": False,
                "interpretation": (
                    "Optional candidate MOCAP-world origin correction for visual "
                    "alignment; a non-zero value is not a measured ground-truth "
                    "calibration."
                ),
            },
            "rear_wrist_mount_visualization": {
                "enabled": bool(draw_rear_mount_proxy),
                "method": (
                    "anchor = wrist - scale * (middle_mcp - wrist), evaluated "
                    "in MOCAP world coordinates before RGB projection"
                ),
                "wrist_joint_index": MOCAP_WRIST_INDEX,
                "middle_mcp_joint_index": MOCAP_MIDDLE_MCP_INDEX,
                "scale": VISUALIZATION_REAR_WRIST_MOUNT_SCALE,
                "original_21_joint_positions_modified": False,
                "additional_mocap_gt_joint": False,
                "interpretation": (
                    "Display-only extrapolation toward the visible rear wrist "
                    "module; the delivered Forearm rigid-body tracks are zero "
                    "placeholders and do not measure this anchor."
                ),
            },
            "independent_2d_joint_labels_available": False,
            "independent_2d_reprojection_error_measured": False,
            "world_to_color_linear_determinant": float(
                np.linalg.det(prepared.calibration.world_to_color[:3, :3])
            ),
            "world_to_color_linear_orthogonality_max_abs_error": float(
                np.max(
                    np.abs(
                        prepared.calibration.world_to_color[:3, :3].T
                        @ prepared.calibration.world_to_color[:3, :3]
                        - np.eye(3)
                    )
                )
            ),
        },
        "artifacts": artifact_payload,
        "interpretation_limits": [
            "Frame-complete means 100% of the trimmed clip has an auditable timestamp bracket; it is not a zero-error physical claim.",
            "Frame-complete timestamp mapping does not imply <=1 ms measured camera-to-MOCAP synchronization; the <=1 ms field is only the delivered software clock-model uncertainty estimate.",
            (
                "BAG/MP4 frame identity is full-frame decoded content evidence."
                if content_qa.get("full_frame_content_comparison")
                else "BAG/MP4 frame identity uses fixed full-span sampled decoded content evidence, explicitly not an all-frame pixel comparison."
            ),
            "The camera/MOCAP clock relationship is estimated in software; no shared exposure/OptiTrack TTL pulse was delivered.",
            "The CS-400 spatial calibration uses manual marker identity selection plus multi-frame metric depth.",
            "The delivered depth-to-color 3x3 block is slightly non-orthogonal, so world-to-color is treated as the recorded forward calibration rather than claimed as a pure SE(3).",
            "No independent per-frame 2D hand landmarks are available to measure pixel reprojection error over the full motion.",
            (
                "Human.cma does not include per-joint visibility, residual, "
                "occlusion, or gap-fill quality fields."
                if prepared.mocap_position_source
                == MOCAP_POSITION_SOURCE_HUMAN_CMA
                else (
                    "Skeleton BVH supplies FK joint positions but no per-joint "
                    "visibility, residual, occlusion, or gap-fill quality fields; "
                    "Human.cma remains the timestamp/frame-counter authority."
                )
            ),
        ],
    }


def _scaled_frame(frame: np.ndarray, output_width: int) -> tuple[np.ndarray, float, float]:
    source_height, source_width = frame.shape[:2]
    if output_width <= 0 or output_width == source_width:
        return frame.copy(), 1.0, 1.0
    target_height = int(round(source_height * output_width / source_width))
    if target_height % 2:
        target_height += 1
    resized = cv2.resize(frame, (output_width, target_height), interpolation=cv2.INTER_AREA)
    return resized, output_width / source_width, target_height / source_height


def render_mocap_frame(
    prepared: PreparedMocapVideo,
    source_frame: int,
    output_frame: int,
    frame: np.ndarray,
    *,
    output_width: int,
    first_source_frame: int,
    draw_rear_mount_proxy: bool = False,
) -> np.ndarray:
    if not prepared.mocap_valid[source_frame]:
        raise ValueError(f"Source frame {source_frame} is not synchronized")
    output, scale_x, scale_y = _scaled_frame(frame, output_width)
    projected, positive = viz.project_world_to_rgb(
        prepared.mocap_mm[source_frame], prepared.calibration
    )
    projected[..., 0] *= scale_x
    projected[..., 1] *= scale_y
    projected[~positive] = np.nan

    mount_projected: np.ndarray | None = None
    if draw_rear_mount_proxy:
        # This extra point is a visualization-only extrapolation toward the
        # rear wrist-mounted black module.  The 21 delivered joints remain intact.
        mount_world = visualization_rear_wrist_mount_anchor(
            prepared.mocap_mm[source_frame]
        )
        mount_projected, mount_positive = viz.project_world_to_rgb(
            mount_world, prepared.calibration
        )
        mount_projected[..., 0] *= scale_x
        mount_projected[..., 1] *= scale_y
        mount_projected[~mount_positive] = np.nan

    for side_index, (label, color) in enumerate(
        (("L", LEFT_COLOR_BGR), ("R", RIGHT_COLOR_BGR))
    ):
        pixels = projected[side_index]
        if mount_projected is not None:
            mount_pixels = np.stack((mount_projected[side_index], pixels[0]))
            viz._draw_chain(output, mount_pixels, (0, 1), OUTLINE_BGR, 9)
            viz._draw_chain(output, mount_pixels, (0, 1), color, 4)
            mount = mount_pixels[0]
            if viz._finite_pixel(mount):
                x, y = np.rint(mount).astype(int)
                height, width = output.shape[:2]
                if -10 <= x < width + 10 and -10 <= y < height + 10:
                    cv2.circle(
                        output,
                        (int(x), int(y)),
                        10,
                        OUTLINE_BGR,
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.circle(
                        output,
                        (int(x), int(y)),
                        6,
                        color,
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.circle(
                        output,
                        (int(x), int(y)),
                        2,
                        (245, 245, 245),
                        -1,
                        cv2.LINE_AA,
                    )
        viz.draw_hand_skeleton(
            output,
            pixels,
            viz.MOCAP_CHAINS,
            OUTLINE_BGR,
            thickness=7,
            radius=7,
            hollow=False,
        )
        viz.draw_hand_skeleton(
            output,
            pixels,
            viz.MOCAP_CHAINS,
            color,
            thickness=3,
            radius=4,
            hollow=False,
        )
        wrist = pixels[0]
        if viz._finite_pixel(wrist):
            x, y = np.rint(wrist).astype(int)
            viz._draw_text(
                output,
                f"{label} MOCAP",
                (int(x + 8), int(y - 9)),
                scale=0.46,
                color=color,
            )

    panel_height = min(91, output.shape[0])
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (output.shape[1], panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.66, output, 0.34, 0.0, output)
    audit = prepared.interpolation
    low = int(audit.low_indices[source_frame])
    high = int(audit.high_indices[source_frame])
    elapsed_s = prepared.rgb_device_s[source_frame] - prepared.rgb_device_s[first_source_frame]
    translation_header = _mocap_world_translation_header(
        prepared.mocap_world_translation_mm
    )
    position_label = (
        "Human.cma"
        if prepared.mocap_position_source == MOCAP_POSITION_SOURCE_HUMAN_CMA
        else "Skeleton_0/1 BVH FK"
    )
    header = f"VIDEO <-> MOCAP | position={position_label}"
    if translation_header is not None:
        header += f" | {translation_header}"
    elif not draw_rear_mount_proxy:
        header += " | metric reprojection"
    if draw_rear_mount_proxy:
        header += " | rear proxy VIZ-only"
    viz._draw_text(output, header, (14, 23), scale=0.52)
    viz._draw_text(
        output,
        (
            f"{prepared.segment_dir.name} | output {output_frame} | source RGB "
            f"{source_frame} | matched t={elapsed_s:7.3f}s"
        ),
        (14, 49),
        scale=0.48,
    )
    viz._draw_text(
        output,
        (
            f"CMA {prepared.mocap.frame_counters[low]} -> "
            f"{prepared.mocap.frame_counters[high]} | {prepared.mocap_position_source} "
            f"ordinal {low}->{high} | alpha="
            f"{audit.alpha[source_frame]:.3f} | nearest dt="
            f"{audit.nearest_delta_s[source_frame] * 1000.0:.2f}ms"
        ),
        (14, 75),
        scale=0.48,
        color=(175, 245, 175),
    )
    return output


def _write_contact_sheet(path: Path, snapshots: Sequence[np.ndarray]) -> Path | None:
    if not snapshots:
        return None
    columns = min(3, len(snapshots))
    rows = int(math.ceil(len(snapshots) / columns))
    height, width = snapshots[0].shape[:2]
    sheet = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, snapshot in enumerate(snapshots):
        row, column = divmod(index, columns)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = snapshot
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), sheet):
        raise RuntimeError(f"Could not write contact sheet: {destination}")
    return destination


def transcode_h264(source: Path, destination: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for H.264 delivery output")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
    )
    return destination


def render_mocap_video(
    prepared: PreparedMocapVideo,
    output_path: Path,
    *,
    output_width: int = 960,
    snapshot_count: int = 6,
    h264: bool = True,
    frame_content_sample_count: int = FRAME_CONTENT_SAMPLE_COUNT,
    draw_rear_mount_proxy: bool = False,
) -> dict[str, Path]:
    first, stop = synchronized_interval(prepared.mocap_valid)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = viz._scaled_size(prepared.video, output_width)
    capture = cv2.VideoCapture(str(prepared.video.path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        prepared.video.fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output}")

    snapshot_indices: set[int] = set()
    if snapshot_count > 0:
        snapshot_indices = set(
            np.rint(np.linspace(first, stop - 1, min(snapshot_count, stop - first))).astype(int)
        )
    snapshots: list[np.ndarray] = []
    source_frame = first
    try:
        while source_frame < stop:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"RGB decoding stopped at source frame {source_frame}")
            rendered = render_mocap_frame(
                prepared,
                source_frame,
                source_frame - first,
                frame,
                output_width=output_width,
                first_source_frame=first,
                draw_rear_mount_proxy=draw_rear_mount_proxy,
            )
            writer.write(rendered)
            if source_frame in snapshot_indices:
                snapshots.append(rendered.copy())
            source_frame += 1
            if source_frame == first + 1 or (source_frame - first) % 300 == 0:
                print(
                    f"{prepared.segment_dir.name}: rendered {source_frame - first}/{stop - first}",
                    flush=True,
                )
    finally:
        capture.release()
        writer.release()

    mapping_path = output.with_suffix(".alignment.csv")
    write_frame_mapping(mapping_path, prepared, first, stop)
    contact_path = output.with_suffix(".contact.jpg")
    _write_contact_sheet(contact_path, snapshots)
    artifacts: dict[str, Path] = {
        "mp4v_preview": output,
        "frame_mapping_csv": mapping_path,
        "contact_sheet": contact_path,
    }
    if h264:
        h264_path = output.with_name(output.stem + "_h264.mp4")
        transcode_h264(output, h264_path)
        artifacts["h264_delivery_video"] = h264_path
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            alignment_metrics(
                prepared,
                first,
                stop,
                artifacts=artifacts,
                frame_content_sample_count=frame_content_sample_count,
                draw_rear_mount_proxy=draw_rear_mount_proxy,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts["metrics_json"] = metrics_path
    return artifacts


def _gap_stem_token(max_interpolation_gap_ms: float) -> str:
    value = float(max_interpolation_gap_ms)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("max_interpolation_gap_ms must be finite and positive")
    if abs(value - 25.0) <= 1e-9:
        return "strict25"
    encoded = np.format_float_positional(value, trim="-").replace(".", "p")
    return f"gap{encoded}"


def alignment_output_stem(
    segment_name: str,
    max_interpolation_gap_ms: float,
    mocap_position_source: str = MOCAP_POSITION_SOURCE_HUMAN_CMA,
) -> str:
    source = _normalize_mocap_position_source(mocap_position_source)
    stem = (
        f"{segment_name}_mocap_video_aligned_"
        f"{_gap_stem_token(max_interpolation_gap_ms)}"
    )
    if source == MOCAP_POSITION_SOURCE_SKELETON_BVH:
        stem += "_skeleton_bvh"
    return stem


def _existing_alignment_artifacts(
    output_dir: Path,
    stem: str,
    frame_mapping_csv: Path,
) -> dict[str, Path]:
    """Keep existing rendered-media hashes when refreshing analyze-only metrics."""

    directory = Path(output_dir)
    candidates = {
        "frame_mapping_csv": Path(frame_mapping_csv),
        "mp4v_preview": directory / f"{stem}.mp4",
        "contact_sheet": directory / f"{stem}.contact.jpg",
        "h264_delivery_video": directory / f"{stem}_h264.mp4",
    }
    return {name: path for name, path in candidates.items() if path.is_file()}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Render a frame-complete MOCAP-only overlay on synchronized RGB video."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "同步整理_20260829_三段",
    )
    parser.add_argument(
        "--segment",
        action="append",
        default=[],
        help="01, 02, 03, Take_000, or exact segment name; repeatable.",
    )
    parser.add_argument("--calibration", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "mocap_video_alignment_review",
    )
    parser.add_argument("--output-width", type=int, default=960)
    parser.add_argument("--snapshot-count", type=int, default=6)
    parser.add_argument("--max-interpolation-gap-ms", type=float, default=25.0)
    parser.add_argument(
        "--mocap-position-source",
        choices=MOCAP_POSITION_SOURCES,
        default=MOCAP_POSITION_SOURCE_HUMAN_CMA,
        help=(
            "3D position source: Human.cma joint positions, or Skeleton_0/1 BVH "
            "forward kinematics on the preserved Human.cma timestamp axis."
        ),
    )
    parser.add_argument(
        "--mocap-world-x-offset-mm",
        type=float,
        default=0.0,
        help=(
            "Candidate translation of all interpolated MOCAP joints along world X; "
            "default 0. Non-zero values are visualization candidates, not GT."
        ),
    )
    parser.add_argument(
        "--mocap-world-y-offset-mm",
        type=float,
        default=0.0,
        help=(
            "Candidate translation of all interpolated MOCAP joints along world Y; "
            "default 0. Non-zero values are visualization candidates, not GT."
        ),
    )
    parser.add_argument(
        "--mocap-world-z-offset-mm",
        type=float,
        default=0.0,
        help=(
            "Candidate translation of all interpolated MOCAP joints along world Z; "
            "default 0. Non-zero values are visualization candidates, not GT."
        ),
    )
    parser.add_argument(
        "--draw-rear-mount-proxy",
        action="store_true",
        help=(
            "Draw a visualization-only point extrapolated behind each wrist. "
            "Disabled by default; this point is not MOCAP GT."
        ),
    )
    parser.add_argument(
        "--frame-content-samples",
        type=int,
        default=FRAME_CONTENT_SAMPLE_COUNT,
        help=(
            "Decoded BAG/MP4 content samples spanning the full recording. "
            "Default: 31. Use 0 to validate every frame."
        ),
    )
    parser.add_argument("--no-h264", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    calibration = (
        args.calibration.expanduser().resolve()
        if args.calibration is not None
        else viz._default_calibration(dataset_root)
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for key in args.segment or ("01", "02", "03"):
        segment = viz._discover_segment(dataset_root, key)
        print(f"Preparing {segment.name}", flush=True)
        prepared = prepare_mocap_video(
            segment,
            calibration,
            max_interpolation_gap_ms=args.max_interpolation_gap_ms,
            mocap_position_source=args.mocap_position_source,
            mocap_world_translation_mm=(
                args.mocap_world_x_offset_mm,
                args.mocap_world_y_offset_mm,
                args.mocap_world_z_offset_mm,
            ),
        )
        first, stop = synchronized_interval(prepared.mocap_valid)
        stem = alignment_output_stem(
            segment.name,
            args.max_interpolation_gap_ms,
            args.mocap_position_source,
        )
        if args.analyze_only:
            mapping_path = write_frame_mapping(
                output_dir / f"{stem}.alignment.csv", prepared, first, stop
            )
            artifacts = _existing_alignment_artifacts(
                output_dir,
                stem,
                mapping_path,
            )
            metrics_path = output_dir / f"{stem}.metrics.json"
            metrics_path.write_text(
                json.dumps(
                    alignment_metrics(
                        prepared,
                        first,
                        stop,
                        artifacts=artifacts,
                        frame_content_sample_count=args.frame_content_samples,
                        draw_rear_mount_proxy=args.draw_rear_mount_proxy,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"Wrote {mapping_path}, {metrics_path}", flush=True)
            continue
        artifacts = render_mocap_video(
            prepared,
            output_dir / f"{stem}.mp4",
            output_width=args.output_width,
            snapshot_count=args.snapshot_count,
            h264=not args.no_h264,
            frame_content_sample_count=args.frame_content_samples,
            draw_rear_mount_proxy=args.draw_rear_mount_proxy,
        )
        print("Wrote " + ", ".join(str(path) for path in artifacts.values()), flush=True)


if __name__ == "__main__":
    main()
