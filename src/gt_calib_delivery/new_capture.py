from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np

from .manual_profiles import (
    FINAL_NINE_PROFILE_SCHEMA,
    SIDE_RESIDUAL_VIDEO_IDS,
    load_final_nine_profile,
)


CALIBRATION_MEMBER = (
    "movementcap_20260831_worldcalib/results/"
    "20260831_processed_tabletop_origin/camera_to_world.json"
)
CALIBRATION_REFERENCE_RGB_MEMBER = (
    "movementcap_20260831_worldcalib/results/"
    "20260831_processed_tabletop_origin/reference_rgb.png"
)

ACTION_RECORDING = "camera_glove_recording_20260831_161912"
ACTION_TAKE = "Take_007"
NO_GLOVE_RECORDING = "camera_glove_recording_20260831_155410"
ANATOMICAL_SIDES = ("left", "right")

# The user-supplied placement photograph numbers eleven reflectors per hand:
# 1/3/5/7/9 are fingertips, 2/4/6/8/10 are proximal/base reflectors, and
# 11 is the black dorsum-module reflector.  Number 11 conditions the virtual
# wrist but is deliberately not drawn as a hand joint.  Left thumb-tip track
# 11781 is re-identified as 12503 after a two-frame gap in Take_007.
MARKER_TRACKS = {
    "left": (
        ("11781", "12503"),
        ("11978",),
        ("12003",),
        ("12001",),
        ("12006",),
        ("12013",),
        ("12045",),
        ("12015",),
        ("12076",),
        ("12051",),
        ("12064",),
    ),
    "right": (
        ("12436",),
        ("12435",),
        ("11619",),
        ("11614",),
        ("11616",),
        ("12434",),
        ("11625",),
        ("11685",),
        ("11627",),
        ("11630",),
        ("12433",),
    ),
}
MARKER_NAMES = (
    "thumb_tip",
    "thumb_base",
    "index_tip",
    "index_base",
    "middle_tip",
    "middle_base",
    "ring_tip",
    "ring_base",
    "pinky_tip",
    "pinky_base",
    "dorsum_module",
)
DISPLAY_MARKER_NUMBERS = tuple(range(1, 11))
TIP_MARKER_INDICES = np.asarray((0, 2, 4, 6, 8), dtype=np.int32)
BASE_MARKER_INDICES = np.asarray((1, 3, 5, 7, 9), dtype=np.int32)
MCP_MARKER_INDICES = np.asarray((3, 5, 7, 9), dtype=np.int32)
DORSUM_MARKER_INDEX = 10
DEFAULT_REAR_OFFSET_MM = 20.0

# Workbench/raw-skeleton node zero is the synthetic wrist; nodes one through
# ten retain the photograph's marker numbering.
RAW_CHAINS = (
    (0, 2, 1),
    (0, 4, 3),
    (0, 6, 5),
    (0, 8, 7),
    (0, 10, 9),
)

# CMM marker number 1..10 -> glove solved-keypoint index.
MARKER_TO_GLOVE_INDICES = np.asarray((3, 1, 7, 4, 11, 8, 15, 12, 19, 16), dtype=np.int32)

GLOVE_NAMES = (
    "wrist",
    "thumb_mcp", "thumb_pip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
GLOVE_CHAINS = (
    (0, 1, 2, 3),
    (0, 4, 5, 6, 7),
    (0, 8, 9, 10, 11),
    (0, 12, 13, 14, 15),
    (0, 16, 17, 18, 19),
)
GLOVE_PALM_INDICES = np.asarray((4, 8, 12, 16), dtype=np.int32)

RAW_LEFT_BGR = (255, 220, 45)
RAW_RIGHT_BGR = (245, 55, 230)
SOLVED_LEFT_BGR = (80, 245, 95)
SOLVED_RIGHT_BGR = (55, 225, 255)


@dataclass(frozen=True)
class CameraModel:
    matrix: np.ndarray
    distortion: np.ndarray
    world_to_color: np.ndarray
    calibration_payload: dict
    intrinsics_payload: dict
    recording: str
    intrinsics_path: Path


@dataclass(frozen=True)
class FrameClock:
    output_indices: np.ndarray
    anchor_indices: np.ndarray
    anchor_device_us: np.ndarray
    target_device_us: np.ndarray
    target_monotonic_s: np.ndarray
    target_cmm_counter: np.ndarray
    nearest_anchor_row: np.ndarray
    lower_anchor_row: np.ndarray
    upper_anchor_row: np.ndarray
    anchor_alignment_error_ms: np.ndarray
    camera_valid: np.ndarray
    cadence_us: float
    step_counts: dict[int, int]


@dataclass(frozen=True)
class NewCapture:
    root: Path
    recording: str
    take: str
    rgb_path: Path
    cmm_path: Path
    alignment_path: Path
    frame_summary_path: Path
    camera: CameraModel
    clock: FrameClock
    # (frame, anatomical side [left, right], photograph marker #1..#11, xyz)
    marker_world_mm: np.ndarray
    marker_tracks: dict[str, tuple[tuple[str, ...], ...]]
    source_rgb_frame_count: int
    source_depth_frame_count: int


@dataclass(frozen=True)
class ManualCalibration:
    global_world_xyz_mm: np.ndarray
    side_world_xyz_mm: dict[str, np.ndarray]
    rear_offset_mm: float = DEFAULT_REAR_OFFSET_MM


def default_manual_calibration() -> ManualCalibration:
    return ManualCalibration(
        global_world_xyz_mm=np.zeros(3, dtype=np.float64),
        side_world_xyz_mm={side: np.zeros(3, dtype=np.float64) for side in ANATOMICAL_SIDES},
        rear_offset_mm=DEFAULT_REAR_OFFSET_MM,
    )


def load_manual_calibration(
    path: Path | None,
    *,
    video_id: str | None = None,
) -> ManualCalibration:
    if path is None:
        return default_manual_calibration()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") == FINAL_NINE_PROFILE_SCHEMA:
        if video_id not in SIDE_RESIDUAL_VIDEO_IDS:
            raise ValueError(
                "Final-nine manual profile requires a Take_007 video_id for this renderer"
            )
        profile = load_final_nine_profile(path)
        annotation = profile.annotations[video_id]
        if not annotation.apply_translation:
            raise ValueError(f"Final-nine profile excludes {video_id}")
        return ManualCalibration(
            global_world_xyz_mm=(
                profile.global_world_xyz_mm
                + annotation.per_video_world_xyz_mm
            ),
            side_world_xyz_mm={
                side: np.asarray(
                    annotation.side_residual_world_xyz_mm[side], dtype=np.float64
                )
                for side in ANATOMICAL_SIDES
            },
            rear_offset_mm=DEFAULT_REAR_OFFSET_MM,
        )
    required = {
        "schema",
        "source_recording",
        "source_take",
        "coordinate_system",
        "units",
        "global_world_xyz_mm",
        "left_world_xyz_mm",
        "right_world_xyz_mm",
        "rear_offset_mm",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Manual XYZ calibration is missing required fields: {missing}")
    if payload["schema"] != "gt_calib.manual_xyz_profile.v1":
        raise ValueError("Unsupported manual XYZ calibration schema")
    if payload["source_recording"] != ACTION_RECORDING:
        raise ValueError("Manual profile belongs to a different recording")
    if payload["source_take"] != ACTION_TAKE:
        raise ValueError("Manual profile belongs to a different take")
    if payload["coordinate_system"] != "mocap_world_mm" or payload["units"] != "mm":
        raise ValueError("Manual profile must use mocap_world_mm coordinates in mm")

    def vector(key: str) -> np.ndarray:
        value = np.asarray(payload[key], dtype=np.float64)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"Manual calibration {key} must contain three finite millimetres")
        return value

    rear = float(payload["rear_offset_mm"])
    if not np.isfinite(rear) or not np.isclose(rear, DEFAULT_REAR_OFFSET_MM):
        raise ValueError(
            f"rear_offset_mm must remain the fixed {DEFAULT_REAR_OFFSET_MM:g} mm"
        )
    return ManualCalibration(
        global_world_xyz_mm=vector("global_world_xyz_mm"),
        side_world_xyz_mm={
            "left": vector("left_world_xyz_mm"),
            "right": vector("right_world_xyz_mm"),
        },
        rear_offset_mm=rear,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def capture_source_assets(capture: NewCapture) -> dict[str, dict[str, str]]:
    """Return the immutable inputs that identify one Take_007 render.

    These hashes make the phrase "latest Take_007 CMM" auditable from an
    extracted delivery without relying only on a recording/take label.
    Project-local paths are converted to ``project://`` by the delivery
    provenance normalizer after rendering.
    """

    paths = {
        "rgb_video": capture.rgb_path,
        "cmm_markers": capture.cmm_path,
        "camera_cmavatar_alignment": capture.alignment_path,
        "glove_aligned_frame_summary": capture.frame_summary_path,
        "camera_intrinsics": capture.camera.intrinsics_path,
    }
    return {
        name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def verify_calibration_reference_rgb(
    calibration_archive: Path,
    no_glove_rgb: Path,
    camera: CameraModel,
) -> dict[str, object]:
    """Prove that the calibration reference image comes from the no-glove clip."""

    source = camera.calibration_payload.get("input", {})
    frame_index = source.get("reference_rgb_frame_index")
    if source.get("reference_recording") != camera.recording:
        raise ValueError("Calibration reference recording does not match the RGB clip")
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError("Calibration reference RGB frame index is missing or invalid")
    frame_count, _, _, _ = _video_properties(no_glove_rgb)
    if source.get("rgb_frame_count") != frame_count or frame_index >= frame_count:
        raise ValueError("Calibration reference RGB frame contract does not match the clip")

    capture = cv2.VideoCapture(str(no_glove_rgb))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open no-glove reference video: {no_glove_rgb}")
    try:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            raise RuntimeError(f"Could not seek no-glove RGB frame {frame_index}")
        ok, decoded_frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"Could not decode no-glove RGB frame {frame_index}")

    with tarfile.open(calibration_archive, "r:gz") as archive:
        member = archive.getmember(CALIBRATION_REFERENCE_RGB_MEMBER)
        handle = archive.extractfile(member)
        if handle is None:
            raise FileNotFoundError(CALIBRATION_REFERENCE_RGB_MEMBER)
        reference_bytes = handle.read()
    reference = cv2.imdecode(
        np.frombuffer(reference_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if reference is None or reference.shape != decoded_frame.shape:
        raise ValueError("Calibration reference PNG shape does not match no-glove RGB")
    if not np.array_equal(reference, decoded_frame):
        delta = np.abs(reference.astype(np.int16) - decoded_frame.astype(np.int16))
        raise ValueError(
            "Calibration reference PNG is not pixel-identical to no-glove RGB "
            f"frame {frame_index}; max channel delta={int(np.max(delta))}"
        )
    decoded_sha256 = hashlib.sha256(decoded_frame.tobytes()).hexdigest()
    return {
        "status": "pixel_identical",
        "reference_recording": camera.recording,
        "reference_rgb_frame_index": frame_index,
        "archive_member": CALIBRATION_REFERENCE_RGB_MEMBER,
        "archive_png_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "decoded_bgr_sha256": decoded_sha256,
    }


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _video_properties(path: Path) -> tuple[int, int, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if min(count, width, height) <= 0 or not np.isclose(fps, 30.0, atol=1e-3):
        raise ValueError(f"Unexpected video properties for {path}: {count}, {width}x{height}, {fps}")
    return count, width, height, fps


def load_camera_model(
    calibration_archive: Path,
    intrinsics_path: Path,
    *,
    expected_recording: str,
) -> CameraModel:
    with tarfile.open(calibration_archive, "r:gz") as archive:
        member = archive.getmember(CALIBRATION_MEMBER)
        if not member.isfile():
            raise ValueError(f"Calibration member is not a regular file: {member.name}")
        handle = archive.extractfile(member)
        if handle is None:
            raise FileNotFoundError(CALIBRATION_MEMBER)
        calibration = json.load(handle)
    if calibration.get("schema") != "movementcap.camera_to_mocap_world.v1":
        raise ValueError("Unsupported camera/world calibration schema")
    intrinsics = json.loads(Path(intrinsics_path).read_text(encoding="utf-8"))
    camera_param = intrinsics["camera_param"]
    matrix = np.asarray(camera_param["rgb_intrinsic"]["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(
        camera_param["rgb_distortion"]["opencv_coefficients"], dtype=np.float64
    )
    world_to_color = np.asarray(
        calibration["transforms"]["world_to_color_camera"], dtype=np.float64
    )
    if matrix.shape != (3, 3) or distortion.shape != (5,) or world_to_color.shape != (4, 4):
        raise ValueError("Malformed camera calibration arrays")
    applies_to = set(calibration.get("applies_to", []))
    if expected_recording not in applies_to:
        raise ValueError(
            f"Calibration does not declare applicability to {expected_recording}"
        )
    return CameraModel(
        matrix,
        distortion,
        world_to_color,
        calibration,
        intrinsics,
        expected_recording,
        Path(intrinsics_path),
    )


def derive_frame_clock(
    alignment_rows: Sequence[dict[str, str]],
    summary_rows: Sequence[dict[str, str]],
    *,
    rgb_frame_count: int,
    depth_frame_count: int,
) -> FrameClock:
    if len(alignment_rows) != len(summary_rows) or len(alignment_rows) < 2:
        raise ValueError("Camera alignment and validity rows must have the same non-trivial length")
    device_us = np.asarray(
        [int(row["color_device_timestamp_us"]) for row in alignment_rows],
        dtype=np.int64,
    )
    delta_us = np.diff(device_us).astype(np.float64)
    steps = np.maximum(1, np.rint(delta_us / 33333.3333333333).astype(np.int64))
    if not np.all(np.isin(steps, (1, 2, 3, 4))):
        raise ValueError(f"Unexpected camera frame steps: {np.unique(steps)}")
    anchor_indices = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(steps)))
    # Take_006 closes on its final depth frame; Take_007 leaves one unanchored
    # RGB/depth tail frame.  Publish only through the last reconstructed anchor.
    last_anchor = int(anchor_indices[-1])
    if last_anchor >= min(rgb_frame_count, depth_frame_count):
        raise ValueError(
            f"Derived last anchor {last_anchor} exceeds available RGB/depth frames"
        )
    rgb_tail = rgb_frame_count - last_anchor - 1
    depth_tail = depth_frame_count - last_anchor - 1
    if rgb_tail not in (0, 1) or depth_tail not in (0, 1):
        raise ValueError(
            "Expected at most one unanchored RGB/depth tail frame, got "
            f"RGB={rgb_tail}, depth={depth_tail}"
        )
    output_indices = np.arange(last_anchor + 1, dtype=np.int64)
    cadence_us = float(np.median(delta_us / steps))

    realtime_ns = np.asarray([int(row["thor_realtime_ns"]) for row in alignment_rows])
    monotonic_ns = np.asarray([int(row["thor_monotonic_ns"]) for row in alignment_rows])
    system_ns = np.asarray(
        [1000 * int(row["color_system_timestamp_us"]) for row in alignment_rows]
    )
    capture_to_poll_ns = realtime_ns - system_ns
    exposure_monotonic_s = (monotonic_ns - capture_to_poll_ns) * 1e-9
    if np.any(np.diff(exposure_monotonic_s) <= 0.0):
        raise ValueError("Exposure-corrected monotonic clock is not increasing")

    cmm_counter = np.asarray(
        [
            int(row["cmavatar_frame_counter"])
            + float(row["alignment_error_ms"]) * 0.12
            for row in alignment_rows
        ],
        dtype=np.float64,
    )
    target_device_us = np.interp(output_indices, anchor_indices, device_us)
    target_monotonic_s = np.interp(
        output_indices, anchor_indices, exposure_monotonic_s
    )
    target_cmm_counter = np.interp(output_indices, anchor_indices, cmm_counter)

    insertion = np.searchsorted(anchor_indices, output_indices, side="left")
    upper = np.clip(insertion, 0, len(anchor_indices) - 1)
    lower = np.clip(insertion - 1, 0, len(anchor_indices) - 1)
    exact = anchor_indices[upper] == output_indices
    lower[exact] = upper[exact]
    choose_lower = (
        np.abs(output_indices - anchor_indices[lower])
        <= np.abs(anchor_indices[upper] - output_indices)
    )
    nearest = np.where(choose_lower, lower, upper)
    camera_valid_anchor = np.asarray(
        [row["valid"] == "1" for row in summary_rows], dtype=bool
    )
    camera_valid = camera_valid_anchor[nearest]
    alignment_error_ms = np.asarray(
        [float(row["alignment_error_ms"]) for row in alignment_rows], dtype=np.float64
    )
    values, counts = np.unique(steps, return_counts=True)
    return FrameClock(
        output_indices=output_indices,
        anchor_indices=anchor_indices,
        anchor_device_us=device_us,
        target_device_us=target_device_us,
        target_monotonic_s=target_monotonic_s,
        target_cmm_counter=target_cmm_counter,
        nearest_anchor_row=nearest,
        lower_anchor_row=lower,
        upper_anchor_row=upper,
        anchor_alignment_error_ms=alignment_error_ms,
        camera_valid=camera_valid,
        cadence_us=cadence_us,
        step_counts={int(value): int(count) for value, count in zip(values, counts, strict=True)},
    )


def _fill_short_gaps(values: np.ndarray, *, max_gap: int = 4) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    valid = np.all(np.isfinite(output), axis=1)
    cursor = 0
    while cursor < len(output):
        if valid[cursor]:
            cursor += 1
            continue
        start = cursor
        while cursor < len(output) and not valid[cursor]:
            cursor += 1
        stop = cursor
        gap = stop - start
        if start == 0 or stop == len(output) or gap > max_gap:
            continue
        for index in range(start, stop):
            alpha = (index - start + 1) / (gap + 1)
            output[index] = output[start - 1] * (1.0 - alpha) + output[stop] * alpha
        valid[start:stop] = True
    return output


def load_marker_tracks(
    cmm_path: Path,
    target_counters: np.ndarray,
    marker_tracks: dict[str, tuple[tuple[str, ...], ...]] = MARKER_TRACKS,
) -> np.ndarray:
    counters: list[int] = []
    samples: list[np.ndarray] = []
    with Path(cmm_path).open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
        metadata = {}
        for _ in range(4):
            key, value = next(handle).rstrip("\r\n").split("\t", 1)
            metadata[key] = value
        if int(metadata["FREQUENCY"]) != 120:
            raise ValueError("CMM frequency is not 120 Hz")
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or [])
        source_ids = {
            marker
            for side in ANATOMICAL_SIDES
            for logical_track in marker_tracks[side]
            for marker in logical_track
        }
        required = {
            f"Marker_{marker}_Pos_{axis}(mm)"
            for marker in source_ids
            for axis in "XYZ"
        }
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"CMM is missing persistent marker fields: {missing[:3]}")
        first = math.floor(float(np.min(target_counters))) - 1
        last = math.ceil(float(np.max(target_counters))) + 1
        for row in reader:
            counter = int(row["FrameCounter"])
            if counter < first:
                continue
            if counter > last:
                break
            sides = []
            for side in ANATOMICAL_SIDES:
                points = []
                for logical_track in marker_tracks[side]:
                    candidates = []
                    for marker in logical_track:
                        values = [
                            row[f"Marker_{marker}_Pos_{axis}(mm)"].strip()
                            for axis in "XYZ"
                        ]
                        if all(values):
                            candidates.append(np.asarray(values, dtype=np.float64))
                    if len(candidates) > 1:
                        separation = float(np.linalg.norm(candidates[0] - candidates[1]))
                        if separation > 2.0:
                            raise ValueError(
                                f"Logical marker aliases diverge by {separation:.3f} mm "
                                f"at CMM frame {counter}"
                            )
                    points.append(
                        candidates[0]
                        if candidates
                        else np.full(3, np.nan, dtype=np.float64)
                    )
                sides.append(np.asarray(points, dtype=np.float64))
            counters.append(counter)
            samples.append(np.asarray(sides, dtype=np.float64))
    source_counters = np.asarray(counters, dtype=np.float64)
    source = np.asarray(samples, dtype=np.float64)
    if len(source_counters) < 2 or np.any(np.diff(source_counters) != 1):
        raise ValueError("Selected CMM counters are not contiguous")
    for side_index in range(len(ANATOMICAL_SIDES)):
        for marker_index in range(11):
            source[:, side_index, marker_index] = _fill_short_gaps(
                source[:, side_index, marker_index]
            )

    low_counter = np.floor(target_counters).astype(np.int64)
    high_counter = np.ceil(target_counters).astype(np.int64)
    low = low_counter - int(source_counters[0])
    high = high_counter - int(source_counters[0])
    if np.any(low < 0) or np.any(high >= len(source)):
        raise ValueError("Target CMM counters exceed the selected contiguous source interval")
    alpha = target_counters - low_counter
    low_values = source[low]
    high_values = source[high]
    output = low_values * (1.0 - alpha[:, None, None, None]) + high_values * alpha[
        :, None, None, None
    ]
    return output


def load_new_capture(
    dataset_root: Path,
    calibration_archive: Path,
    *,
    recording: str = ACTION_RECORDING,
    take: str = ACTION_TAKE,
    marker_tracks: Mapping[str, Sequence[Sequence[str]]] = MARKER_TRACKS,
) -> NewCapture:
    root = Path(dataset_root) / recording
    rgb_path = root / "rgbd_unpack" / "RGB.mp4"
    depth_path = root / "rgbd_unpack" / "Depth.mp4"
    intrinsics_path = root / "rgbd_unpack" / "camera_1_intrinsics.json"
    cmm_path = root / "mocap" / take / f"{take}.cmm"
    alignment_path = (
        root / "mocap" / take / "alignment" / "camera_cmavatar_alignment.csv"
    )
    frame_summary_path = (
        root / "glove_processing" / "aligned" / "primary" / "aligned_frame_summary.csv"
    )
    required = (
        rgb_path, depth_path, intrinsics_path, cmm_path, alignment_path,
        frame_summary_path, calibration_archive,
    )
    absent = [str(path) for path in required if not Path(path).is_file()]
    if absent:
        raise FileNotFoundError("Missing new-capture inputs: " + ", ".join(absent))
    rgb_count, _, _, _ = _video_properties(rgb_path)
    depth_count, _, _, _ = _video_properties(depth_path)
    alignment_rows = _csv_rows(alignment_path)
    summary_rows = _csv_rows(frame_summary_path)
    clock = derive_frame_clock(
        alignment_rows,
        summary_rows,
        rgb_frame_count=rgb_count,
        depth_frame_count=depth_count,
    )
    camera = load_camera_model(
        calibration_archive,
        intrinsics_path,
        expected_recording=recording,
    )
    normalized_marker_tracks = {
        side: tuple(tuple(track) for track in marker_tracks[side])
        for side in ANATOMICAL_SIDES
    }
    markers = load_marker_tracks(
        cmm_path,
        clock.target_cmm_counter,
        marker_tracks=normalized_marker_tracks,
    )
    return NewCapture(
        root=root,
        recording=recording,
        take=take,
        rgb_path=rgb_path,
        cmm_path=cmm_path,
        alignment_path=alignment_path,
        frame_summary_path=frame_summary_path,
        camera=camera,
        clock=clock,
        marker_world_mm=markers,
        marker_tracks=normalized_marker_tracks,
        source_rgb_frame_count=rgb_count,
        source_depth_frame_count=depth_count,
    )


def project_world(points_world_mm: np.ndarray, camera: CameraModel) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world_mm, dtype=np.float64)
    rotation = camera.world_to_color[:3, :3]
    translation = camera.world_to_color[:3, 3]
    camera_points = points @ rotation.T + translation
    pixels, _ = cv2.projectPoints(
        points.reshape(-1, 3),
        rotation,
        translation,
        camera.matrix,
        camera.distortion,
    )
    return pixels.reshape(points.shape[:-1] + (2,)), camera_points[..., 2] > 0.0


def _load_glove_local(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        fields = next(reader)
        time_index = fields.index("host_monotonic_ns")
        indices = [
            [fields.index(f"local_{name}_{axis}_m") for axis in "xyz"]
            for name in GLOVE_NAMES
        ]
        times: list[float] = []
        points: list[np.ndarray] = []
        for row in reader:
            times.append(int(row[time_index]) * 1e-9)
            points.append(
                np.asarray(
                    [[float(row[index]) * 1000.0 for index in xyz] for xyz in indices],
                    dtype=np.float64,
                )
            )
    time_array = np.asarray(times, dtype=np.float64)
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.shape[1:] != (20, 3) or np.any(np.diff(time_array) <= 0.0):
        raise ValueError(f"Malformed glove keypoints: {path}")
    return time_array, point_array


def _interpolate_glove(
    times: np.ndarray,
    points: np.ndarray,
    target_times: np.ndarray,
    camera_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    insertion = np.searchsorted(times, target_times, side="left")
    high = np.clip(insertion, 0, len(times) - 1)
    low = np.clip(insertion - 1, 0, len(times) - 1)
    exact = times[high] == target_times
    low[exact] = high[exact]
    span = times[high] - times[low]
    alpha = np.divide(
        target_times - times[low],
        span,
        out=np.zeros_like(target_times),
        where=span > 0.0,
    )
    interpolated = points[low] * (1.0 - alpha[:, None, None]) + points[high] * alpha[:, None, None]
    nearest_error_s = np.minimum(
        np.abs(target_times - times[low]),
        np.abs(times[high] - target_times),
    )
    valid = (
        (target_times >= times[0])
        & (target_times <= times[-1])
        & (nearest_error_s <= 0.020)
        & camera_valid
        & np.all(np.isfinite(interpolated), axis=(1, 2))
    )
    interpolated[~valid] = np.nan
    return interpolated, valid, nearest_error_s * 1000.0


def load_solved_pose(capture: NewCapture) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    poses: dict[str, np.ndarray] = {}
    valid_sides: list[np.ndarray] = []
    nearest_errors: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        path = (
            capture.root / "glove_processing" / "solved" / "primary"
            / f"{side}_hand_keypoints.csv"
        )
        times, local = _load_glove_local(path)
        pose, valid, nearest_ms = _interpolate_glove(
            times,
            local,
            capture.clock.target_monotonic_s,
            capture.clock.camera_valid,
        )
        poses[side] = pose
        valid_sides.append(valid)
        nearest_errors[side] = nearest_ms
    valid_both = valid_sides[0] & valid_sides[1]
    diagnostics = {
        "valid_frames": int(np.count_nonzero(valid_both)),
        "total_frames": int(len(valid_both)),
        "coverage_percent": float(np.mean(valid_both) * 100.0),
        "nearest_solver_sample_error_ms": {
            side: {
                "median": float(np.median(values[valid_both])),
                "p95": float(np.percentile(values[valid_both], 95)),
                "max": float(np.max(values[valid_both])),
            }
            for side, values in nearest_errors.items()
        },
    }
    return poses, valid_both, diagnostics


def _proper_kabsch_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    covariance = np.asarray(source).T @ np.asarray(target)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    return rotation


def virtual_wrist(
    markers_world_mm: np.ndarray,
    *,
    rear_offset_mm: float = DEFAULT_REAR_OFFSET_MM,
) -> np.ndarray:
    markers = np.asarray(markers_world_mm, dtype=np.float64)
    if markers.shape[-2:] != (11, 3):
        raise ValueError("Expected photograph marker #1..#11 with xyz coordinates")
    mcp_center = np.mean(markers[..., MCP_MARKER_INDICES, :], axis=-2)
    dorsum = markers[..., DORSUM_MARKER_INDEX, :]
    backward = dorsum - mcp_center
    norm = np.linalg.norm(backward, axis=-1, keepdims=True)
    direction = np.divide(
        backward,
        norm,
        out=np.full_like(backward, np.nan),
        where=norm > 1e-9,
    )
    return dorsum + float(rear_offset_mm) * direction


def raw_mocap_nodes(
    markers_world_mm: np.ndarray,
    *,
    rear_offset_mm: float = DEFAULT_REAR_OFFSET_MM,
) -> np.ndarray:
    markers = np.asarray(markers_world_mm, dtype=np.float64)
    root = virtual_wrist(markers, rear_offset_mm=rear_offset_mm)
    return np.concatenate((root[..., None, :], markers[..., :10, :]), axis=-2)


def _alignment_reference_vectors(
    local_points_mm: np.ndarray,
    markers_world_mm: np.ndarray,
    *,
    rear_offset_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    local = np.asarray(local_points_mm, dtype=np.float64)
    markers = np.asarray(markers_world_mm, dtype=np.float64)
    root = virtual_wrist(markers, rear_offset_mm=rear_offset_mm)
    source = local[MARKER_TO_GLOVE_INDICES] - local[0]
    target = markers[:10] - root
    return source, target


def _frame_similarity_scale(source: np.ndarray, target: np.ndarray) -> float:
    rotation = _proper_kabsch_rotation(source, target)
    mapped = source @ rotation.T
    denominator = float(np.sum(mapped * mapped))
    if denominator <= 1e-9:
        return float("nan")
    return float(np.sum(mapped * target) / denominator)


def estimate_marker_scale(
    local_points_mm: np.ndarray,
    markers_world_mm: np.ndarray,
    valid: np.ndarray,
    *,
    rear_offset_mm: float = DEFAULT_REAR_OFFSET_MM,
) -> tuple[float, dict[str, float]]:
    samples: list[float] = []
    for frame_index in np.flatnonzero(valid):
        source, target = _alignment_reference_vectors(
            local_points_mm[frame_index],
            markers_world_mm[frame_index],
            rear_offset_mm=rear_offset_mm,
        )
        if not (np.all(np.isfinite(source)) and np.all(np.isfinite(target))):
            continue
        value = _frame_similarity_scale(source, target)
        if np.isfinite(value) and 0.25 <= value <= 2.5:
            samples.append(value)
    if not samples:
        raise ValueError("No valid marker frames are available for scale calibration")
    values = np.asarray(samples, dtype=np.float64)
    scale = float(np.median(values))
    return scale, {
        "sample_count": int(len(values)),
        "median": scale,
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def condition_solved_pose(
    local_points_mm: np.ndarray,
    markers_world_mm: np.ndarray,
    side: str,
    *,
    scale: float,
    rear_offset_mm: float = DEFAULT_REAR_OFFSET_MM,
    world_offset_mm: np.ndarray | None = None,
) -> np.ndarray:
    if side not in ANATOMICAL_SIDES:
        raise ValueError(f"Unknown anatomical side: {side}")
    source, target = _alignment_reference_vectors(
        local_points_mm,
        markers_world_mm,
        rear_offset_mm=rear_offset_mm,
    )
    rotation = _proper_kabsch_rotation(source, target)
    root = virtual_wrist(markers_world_mm, rear_offset_mm=rear_offset_mm)
    offset = (
        np.zeros(3, dtype=np.float64)
        if world_offset_mm is None
        else np.asarray(world_offset_mm, dtype=np.float64)
    )
    return (
        (np.asarray(local_points_mm, dtype=np.float64) - local_points_mm[0])
        @ rotation.T
        * float(scale)
        + root
        + offset
    )


def align_solved_pose(
    capture: NewCapture,
    poses: dict[str, np.ndarray],
    valid: np.ndarray,
    manual: ManualCalibration,
) -> tuple[dict[str, np.ndarray], dict]:
    world: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict] = {}
    for side_index, side in enumerate(ANATOMICAL_SIDES):
        markers = capture.marker_world_mm[:, side_index]
        scale, scale_stats = estimate_marker_scale(
            poses[side],
            markers,
            valid,
            rear_offset_mm=manual.rear_offset_mm,
        )
        aligned = np.full_like(poses[side], np.nan, dtype=np.float64)
        side_offset = manual.global_world_xyz_mm + manual.side_world_xyz_mm[side]
        for frame_index in np.flatnonzero(valid):
            if not np.all(np.isfinite(markers[frame_index])):
                continue
            aligned[frame_index] = condition_solved_pose(
                poses[side][frame_index],
                markers[frame_index],
                side,
                scale=scale,
                rear_offset_mm=manual.rear_offset_mm,
                world_offset_mm=side_offset,
            )
        correspondence = aligned[:, MARKER_TO_GLOVE_INDICES]
        target = markers[:, :10] + side_offset
        residual = np.linalg.norm(correspondence - target, axis=-1)
        finite = np.isfinite(residual) & valid[:, None]
        tips = finite[:, TIP_MARKER_INDICES]
        bases = finite[:, BASE_MARKER_INDICES]

        def stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
            selected = values[mask]
            return {
                "count": int(len(selected)),
                "median_mm": float(np.median(selected)),
                "p95_mm": float(np.percentile(selected, 95)),
                "max_mm": float(np.max(selected)),
            }

        world[side] = aligned
        diagnostics[side] = {
            "frozen_10_marker_scale": scale,
            "scale_samples": scale_stats,
            "all_10_surface_marker_correspondences": stats(residual, finite),
            "five_fingertip_correspondences_used_in_fit": stats(
                residual[:, TIP_MARKER_INDICES], tips
            ),
            "five_base_surface_markers": stats(
                residual[:, BASE_MARKER_INDICES], bases
            ),
        }
    return world, diagnostics


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.55,
    color: tuple[int, int, int] = (245, 245, 245),
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _draw_header(image: np.ndarray, lines: Sequence[tuple[str, tuple[int, int, int]]]) -> None:
    height = 12 + 25 * len(lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, image, 0.32, 0.0, image)
    for index, (line, color) in enumerate(lines):
        _put_text(image, line, (14, 24 + index * 24), scale=0.50, color=color)


def _draw_hand_skeleton(
    image: np.ndarray,
    pixels: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    rectangle = (0, 0, image.shape[1], image.shape[0])
    for chain in GLOVE_CHAINS:
        for first, second in zip(chain, chain[1:]):
            if not (np.all(np.isfinite(pixels[first])) and np.all(np.isfinite(pixels[second]))):
                continue
            p1 = tuple(np.rint(pixels[first]).astype(int))
            p2 = tuple(np.rint(pixels[second]).astype(int))
            visible, a, b = cv2.clipLine(rectangle, p1, p2)
            if visible:
                cv2.line(image, a, b, color, 4, cv2.LINE_AA)
    for point in pixels:
        if np.all(np.isfinite(point)):
            x, y = np.rint(point).astype(int)
            if -6 <= x < image.shape[1] + 6 and -6 <= y < image.shape[0] + 6:
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.circle(image, (int(x), int(y)), 5, color, 2, cv2.LINE_AA)


def _transcode_h264(intermediate: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(intermediate),
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    subprocess.run(command, check=True)
    intermediate.unlink()


def _contact_sheet(frames: Sequence[np.ndarray], output: Path) -> None:
    if not frames:
        return
    columns = min(3, len(frames))
    rows = math.ceil(len(frames) / columns)
    height, width = frames[0].shape[:2]
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        canvas[row * height : (row + 1) * height, column * width : (column + 1) * width] = frame
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Could not write contact sheet: {output}")


def write_frame_map(capture: NewCapture, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    anchors = {int(value) for value in capture.clock.anchor_indices}
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "output_frame", "source_rgb_frame", "mapping_kind",
                "derived_color_device_timestamp_us", "cmm_fractional_frame_counter",
                "lower_camera_timecode_id", "upper_camera_timecode_id",
                "nearest_anchor_alignment_error_ms", "camera_glove_valid",
            )
        )
        for frame in capture.clock.output_indices:
            low = int(capture.clock.lower_anchor_row[frame])
            high = int(capture.clock.upper_anchor_row[frame])
            nearest = int(capture.clock.nearest_anchor_row[frame])
            writer.writerow(
                (
                    int(frame), int(frame),
                    "camera_timecode_anchor" if int(frame) in anchors else "timestamp_inferred",
                    f"{capture.clock.target_device_us[frame]:.3f}",
                    f"{capture.clock.target_cmm_counter[frame]:.6f}",
                    low + 1, high + 1,
                    f"{capture.clock.anchor_alignment_error_ms[nearest]:.6f}",
                    int(capture.clock.camera_valid[frame]),
                )
            )


def _write_metrics(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_raw_markers(
    capture: NewCapture,
    output: Path,
    poster: Path,
    metrics: Path,
    *,
    output_width: int = 960,
    manual: ManualCalibration | None = None,
) -> None:
    manual = manual or default_manual_calibration()
    source_count, source_width, source_height, fps = _video_properties(capture.rgb_path)
    output_height = int(round(source_height * output_width / source_width))
    output_height += output_height % 2
    intermediate = output.with_name(output.stem + ".mp4v.mp4")
    intermediate.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {intermediate}")
    capture_video = cv2.VideoCapture(str(capture.rgb_path))
    snapshots: list[np.ndarray] = []
    snapshot_indices = set(
        np.rint(np.linspace(0, len(capture.clock.output_indices) - 1, 6)).astype(int)
    )
    nodes = raw_mocap_nodes(
        capture.marker_world_mm,
        rear_offset_mm=manual.rear_offset_mm,
    )
    for side_index, side in enumerate(ANATOMICAL_SIDES):
        nodes[:, side_index] += (
            manual.global_world_xyz_mm + manual.side_world_xyz_mm[side]
        )
    all_pixels, all_positive = project_world(nodes, capture.camera)
    scale = output_width / source_width
    all_pixels *= scale
    try:
        for frame_index in capture.clock.output_indices:
            ok, frame = capture_video.read()
            if not ok:
                raise RuntimeError(f"RGB decode stopped at frame {frame_index}")
            frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
            for side_index, (side, color) in enumerate(
                (("left", RAW_LEFT_BGR), ("right", RAW_RIGHT_BGR))
            ):
                pixels = all_pixels[frame_index, side_index]
                positive = all_positive[frame_index, side_index]
                rectangle = (0, 0, frame.shape[1], frame.shape[0])
                for chain in RAW_CHAINS:
                    for first, second in zip(chain, chain[1:]):
                        if not (positive[first] and positive[second]):
                            continue
                        p1 = tuple(np.rint(pixels[first]).astype(int))
                        p2 = tuple(np.rint(pixels[second]).astype(int))
                        visible, a, b = cv2.clipLine(rectangle, p1, p2)
                        if visible:
                            cv2.line(frame, a, b, color, 3, cv2.LINE_AA)
                for node_index in range(1, 11):
                    if not positive[node_index]:
                        continue
                    for trail_offset, radius in ((8, 2), (4, 3)):
                        prior = max(int(frame_index) - trail_offset, 0)
                        if all_positive[prior, side_index, node_index]:
                            x, y = np.rint(
                                all_pixels[prior, side_index, node_index]
                            ).astype(int)
                            cv2.circle(
                                frame,
                                (int(x), int(y)),
                                radius,
                                color,
                                -1,
                                cv2.LINE_AA,
                            )
                    x, y = np.rint(pixels[node_index]).astype(int)
                    cv2.circle(frame, (int(x), int(y)), 7, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(frame, (int(x), int(y)), 5, color, -1, cv2.LINE_AA)
                    _put_text(
                        frame,
                        str(node_index),
                        (int(x + 7), int(y - 7)),
                        scale=0.34,
                        color=color,
                    )
                if positive[0]:
                    x, y = np.rint(pixels[0]).astype(int)
                    cv2.circle(frame, (int(x), int(y)), 9, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(frame, (int(x), int(y)), 7, (255, 255, 255), 2, cv2.LINE_AA)
                    _put_text(
                        frame,
                        f"{side[0].upper()} root",
                        (int(x + 10), int(y + 15)),
                        scale=0.36,
                        color=color,
                    )
            nearest = int(capture.clock.nearest_anchor_row[frame_index])
            kind = (
                "ANCHOR" if frame_index in set(capture.clock.anchor_indices)
                else "INFERRED"
            )
            _draw_header(
                frame,
                (
                    ("Take_007 | CMM 20 labeled marker points -> RGB", (245, 245, 245)),
                    (
                        f"RGB {frame_index}/{len(capture.clock.output_indices)-1} | "
                        f"CMM {capture.clock.target_cmm_counter[frame_index]:.3f} | {kind} | "
                        f"nearest dt={capture.clock.anchor_alignment_error_ms[nearest]:+.3f} ms",
                        (210, 230, 245),
                    ),
                    (
                        "#11 hidden as a joint | virtual root = dorsum #11 + "
                        f"{manual.rear_offset_mm:g} mm proximal",
                        (90, 220, 255),
                    ),
                ),
            )
            writer.write(frame)
            if int(frame_index) in snapshot_indices:
                snapshots.append(frame.copy())
    finally:
        capture_video.release()
        writer.release()
    _transcode_h264(intermediate, output)
    _contact_sheet(snapshots, poster)
    _write_metrics(
        metrics,
        {
            "schema": "gt_calib.take007_labeled_cmm_markers.v2",
            "status": "visualization_complete",
            "source_recording": capture.recording,
            "source_take": capture.take,
            "source_assets": capture_source_assets(capture),
            "source_rgb_frames": source_count,
            "rendered_frames": int(len(capture.clock.output_indices)),
            "excluded_unsynchronized_rgb_frames": [source_count - 1],
            "source_marker_count": 22,
            "displayed_measured_marker_count": 20,
            "displayed_marker_count_per_hand": 10,
            "hidden_conditioning_marker_number": 11,
            "virtual_root_count": 2,
            "virtual_root_definition": (
                "marker #11 plus rear_offset_mm along normalize(#11 - mean(#4,#6,#8,#10))"
            ),
            "rear_offset_mm": manual.rear_offset_mm,
            "marker_names_1_to_11": list(MARKER_NAMES),
            "marker_tracks": {
                side: [list(track) for track in MARKER_TRACKS[side]]
                for side in ANATOMICAL_SIDES
            },
            "manual_calibration": {
                "global_world_xyz_mm": manual.global_world_xyz_mm.tolist(),
                "left_world_xyz_mm": manual.side_world_xyz_mm["left"].tolist(),
                "right_world_xyz_mm": manual.side_world_xyz_mm["right"].tolist(),
            },
            "mocap_representation": (
                "ten photographed surface-marker correspondences per hand plus a VIZ-only virtual wrist"
            ),
            "frame_mapping": {
                "status": "derived_from_sparse_device_timestamps",
                "anchor_count": int(len(capture.clock.anchor_indices)),
                "cadence_us": capture.clock.cadence_us,
                "step_counts": capture.clock.step_counts,
                "last_anchor_rgb_index": int(capture.clock.anchor_indices[-1]),
            },
            "camera_alignment_abs_error_ms": {
                "median": float(np.median(np.abs(capture.clock.anchor_alignment_error_ms))),
                "p95": float(np.percentile(np.abs(capture.clock.anchor_alignment_error_ms), 95)),
                "max": float(np.max(np.abs(capture.clock.anchor_alignment_error_ms))),
            },
            "limitations": [
                "The MP4 has no embedded per-frame hardware timestamps; the frame map is structurally derived.",
                "Surface reflectors approximate glove bases/tips; they are not anatomical joint centers.",
                "Marker #11 conditions the virtual wrist but is not drawn as a twenty-first joint.",
                "Pixel alignment is calibration consistency, not independent dynamic hand GT accuracy.",
            ],
        },
    )


def render_solved_pose(
    capture: NewCapture,
    output: Path,
    poster: Path,
    metrics: Path,
    *,
    output_width: int = 960,
    manual: ManualCalibration | None = None,
    prepared: tuple[dict[str, np.ndarray], np.ndarray, dict] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    manual = manual or default_manual_calibration()
    poses, valid, diagnostics = load_solved_pose(capture)
    if prepared is None:
        world_poses, alignment_diagnostics = align_solved_pose(
            capture, poses, valid, manual
        )
    else:
        world_poses, prepared_valid, alignment_diagnostics = prepared
        if not np.array_equal(valid, prepared_valid):
            raise ValueError("Prepared solved-pose validity does not match source data")
    _, source_width, source_height, fps = _video_properties(capture.rgb_path)
    output_height = int(round(source_height * output_width / source_width))
    output_height += output_height % 2
    intermediate = output.with_name(output.stem + ".mp4v.mp4")
    writer = cv2.VideoWriter(
        str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {intermediate}")
    capture_video = cv2.VideoCapture(str(capture.rgb_path))
    snapshots: list[np.ndarray] = []
    snapshot_indices = set(
        np.rint(np.linspace(0, len(capture.clock.output_indices) - 1, 6)).astype(int)
    )
    projection_scale = output_width / source_width
    try:
        for frame_index in capture.clock.output_indices:
            ok, frame = capture_video.read()
            if not ok:
                raise RuntimeError(f"RGB decode stopped at frame {frame_index}")
            frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
            if valid[frame_index]:
                for side, color in (("left", SOLVED_LEFT_BGR), ("right", SOLVED_RIGHT_BGR)):
                    world = world_poses[side][frame_index]
                    pixels, positive = project_world(world, capture.camera)
                    pixels[~positive] = np.nan
                    pixels *= projection_scale
                    _draw_hand_skeleton(frame, pixels, color)
                    wrist = pixels[0]
                    if np.all(np.isfinite(wrist)):
                        x, y = np.rint(wrist).astype(int)
                        _put_text(frame, side[0].upper(), (int(x + 8), int(y - 8)), color=color)
            else:
                _put_text(
                    frame,
                    "SOLVED POSE UNAVAILABLE (camera/glove 20 ms validity gate)",
                    (18, output_height - 22),
                    scale=0.48,
                    color=(80, 170, 255),
                )
            _draw_header(
                frame,
                (
                    ("Take_007 | GLOVE SOLVED ARTICULATION 20-joint / hand", (245, 245, 245)),
                    (
                        "root = CMM #11 + "
                        f"{manual.rear_offset_mm:g} mm proximal | frozen scale + "
                        "per-frame rotation from markers #1..#10",
                        (105, 245, 160),
                    ),
                    (
                        "all ten surface markers participate | same-take alignment, not independent accuracy GT",
                        (90, 220, 255),
                    ),
                ),
            )
            writer.write(frame)
            if int(frame_index) in snapshot_indices:
                snapshots.append(frame.copy())
    finally:
        capture_video.release()
        writer.release()
    _transcode_h264(intermediate, output)
    _contact_sheet(snapshots, poster)
    _write_metrics(
        metrics,
        {
            "schema": "gt_calib.take007_cmm_aligned_glove_pose.v2",
            "status": "visualization_complete",
            "source_recording": capture.recording,
            "source_take": capture.take,
            "source_assets": capture_source_assets(capture),
            "rendered_frames": int(len(capture.clock.output_indices)),
            "validity": diagnostics,
            "root_conditioning": {
                "conditional_on_cmm_dorsum_marker_translation": True,
                "conditional_on_mocap_palm_orientation": True,
                "per_frame_fit_uses_virtual_wrist_and_ten_surface_markers": True,
                "finger_or_fingertip_markers_used_in_fit": True,
                "glove_input": "root-local solved 20-joint keypoints",
                "rear_offset_mm": manual.rear_offset_mm,
                "fit_marker_numbers": list(DISPLAY_MARKER_NUMBERS),
                "hidden_dorsum_marker_number": 11,
            },
            "alignment_diagnostics": alignment_diagnostics,
            "manual_calibration": {
                "global_world_xyz_mm": manual.global_world_xyz_mm.tolist(),
                "left_world_xyz_mm": manual.side_world_xyz_mm["left"].tolist(),
                "right_world_xyz_mm": manual.side_world_xyz_mm["right"].tolist(),
            },
            "evaluation_scope": "same-take visualization fit; not an independent accuracy evaluation",
            "limitations": [
                "CMM reflectors lie on glove surfaces and are not anatomical joint centers.",
                "Global wrist/palm pose is conditioned on MOCAP markers and is not glove-only 6DoF.",
                "Invalid glove/camera intervals are shown as unavailable and are not held or extrapolated.",
            ],
        },
    )
    return world_poses, valid, alignment_diagnostics


def _dashed_line(
    image: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
    color: tuple[int, int, int],
    *,
    segments: int = 14,
) -> None:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    for index in range(0, segments, 2):
        start = a + (b - a) * (index / segments)
        stop = a + (b - a) * (min(index + 1, segments) / segments)
        cv2.line(image, tuple(np.rint(start).astype(int)), tuple(np.rint(stop).astype(int)), color, 2, cv2.LINE_AA)


def render_no_glove_calibration(
    no_glove_rgb: Path,
    camera: CameraModel,
    output: Path,
    poster: Path,
    metrics: Path,
    *,
    reference_verification: dict[str, object],
    output_width: int = 960,
) -> None:
    frame_count, source_width, source_height, fps = _video_properties(no_glove_rgb)
    if frame_count != 98:
        raise ValueError(f"Expected 98 no-glove calibration frames, got {frame_count}")
    output_height = int(round(source_height * output_width / source_width))
    output_height += output_height % 2
    intermediate = output.with_name(output.stem + ".mp4v.mp4")
    writer = cv2.VideoWriter(
        str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {intermediate}")
    axis_points = np.asarray(
        ([0, 0, 0], [300, 0, 0], [0, 300, 0], [0, 0, 300]), dtype=np.float64
    )
    marker_points = np.asarray(
        ([370, 0, 45], [110, 0, 45], [0, 65, 45], [0, 210, 45]),
        dtype=np.float64,
    )
    plane_points = np.asarray(
        ([0, 0, 45], [370, 0, 45], [0, 210, 45]), dtype=np.float64
    )
    axis_pixels, _ = project_world(axis_points, camera)
    marker_pixels, _ = project_world(marker_points, camera)
    plane_pixels, _ = project_world(plane_points, camera)
    projection_scale = output_width / source_width
    axis_pixels *= projection_scale
    marker_pixels *= projection_scale
    plane_pixels *= projection_scale
    capture_video = cv2.VideoCapture(str(no_glove_rgb))
    snapshots: list[np.ndarray] = []
    snapshot_indices = {0, 30, 60, 97}
    axis_colors = ((40, 60, 250), (65, 230, 80), (245, 100, 45))
    axis_labels = ("+X 300 mm", "+Y 300 mm", "+Z 300 mm")
    try:
        for frame_index in range(frame_count):
            ok, frame = capture_video.read()
            if not ok:
                raise RuntimeError(f"No-glove RGB decode stopped at frame {frame_index}")
            frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
            origin = tuple(np.rint(axis_pixels[0]).astype(int))
            for endpoint, color, label in zip(axis_pixels[1:], axis_colors, axis_labels, strict=True):
                point = tuple(np.rint(endpoint).astype(int))
                cv2.arrowedLine(frame, origin, point, color, 4, cv2.LINE_AA, tipLength=0.08)
                _put_text(frame, label, (point[0] + 7, point[1] - 5), scale=0.46, color=color)
            marker_labels = ("long_far", "long_near", "short_near", "short_far")
            for point, label in zip(marker_pixels, marker_labels, strict=True):
                x, y = np.rint(point).astype(int)
                cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, (int(x), int(y)), 6, (40, 220, 255), 2, cv2.LINE_AA)
                _put_text(
                    frame,
                    label,
                    (int(x + 9), int(y - 8)),
                    scale=0.35,
                    color=(40, 220, 255),
                )
            _dashed_line(
                frame,
                tuple(np.rint(plane_pixels[0]).astype(int)),
                tuple(np.rint(plane_pixels[1]).astype(int)),
                (40, 220, 255),
            )
            _dashed_line(
                frame,
                tuple(np.rint(plane_pixels[0]).astype(int)),
                tuple(np.rint(plane_pixels[2]).astype(int)),
                (40, 220, 255),
            )
            _draw_header(
                frame,
                (
                    (
                        "NO-GLOVE 155410 CALIBRATION | matching CS-400 world -> RGB",
                        (245, 245, 245),
                    ),
                    ("tabletop origin O | marker plane z=+45 mm | right-handed XYZ in millimetres", (210, 230, 245)),
                    ("RGB PnP RMS 0.294 px / max 0.407 px | fixed-camera probe max 1.986 px", (90, 220, 255)),
                ),
            )
            _put_text(
                frame,
                "Metric pose is RGB+CS400 PnP; lossy Depth.mp4 preview is NOT used as metric depth.",
                (14, output_height - 18),
                scale=0.43,
                color=(230, 230, 230),
            )
            writer.write(frame)
            if frame_index in snapshot_indices:
                snapshots.append(frame.copy())
    finally:
        capture_video.release()
        writer.release()
    _transcode_h264(intermediate, output)
    _contact_sheet(snapshots, poster)
    quality = camera.calibration_payload["pose_quality"]
    validation = camera.calibration_payload["diagnostics"]["fixed_camera_validation"]
    _write_metrics(
        metrics,
        {
            "schema": "gt_calib.no_glove_world_calibration_video.v1",
            "status": "calibration_evidence_complete",
            "source_recording": camera.recording,
            "source_rgb": str(no_glove_rgb),
            "source_rgb_sha256": sha256_file(no_glove_rgb),
            "source_intrinsics": str(camera.intrinsics_path),
            "source_intrinsics_sha256": sha256_file(camera.intrinsics_path),
            "calibration_reference_frame": camera.calibration_payload["input"][
                "reference_rgb_frame_index"
            ],
            "calibration_reference_identity": reference_verification,
            "calibration_method": camera.calibration_payload.get("method"),
            "frames": frame_count,
            "world_coordinate_system": camera.calibration_payload["coordinate_system"],
            "rgb_pnp_reprojection_rms_px": quality["reprojection_rms_px"],
            "rgb_pnp_reprojection_max_px": quality["reprojection_max_px"],
            "fixed_camera_validation": validation,
            "depth_policy": (
                "Depth.mp4 is an 8-bit lossy preview and was not used for the metric RGB PnP pose"
            ),
            "evaluation_scope": (
                "internal calibration reprojection and fixed-camera consistency; "
                "not independent dynamic hand GT accuracy"
            ),
        },
    )


def export_calibration_workbench(
    capture: NewCapture,
    world_poses: dict[str, np.ndarray],
    valid: np.ndarray,
    destination: Path,
    *,
    rear_offset_mm: float = DEFAULT_REAR_OFFSET_MM,
) -> dict[str, Path]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    frame_count = int(len(capture.clock.output_indices))
    _, source_width, source_height, _ = _video_properties(capture.rgb_path)
    video_width = 960
    video_height = int(round(source_height * video_width / source_width))
    video_height += video_height % 2
    clean_video = destination / "take007_clean_rgb.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(capture.rgb_path),
            "-frames:v",
            str(frame_count),
            "-an",
            "-vf",
            f"scale={video_width}:{video_height}",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(clean_video),
        ],
        check=True,
    )

    mocap = raw_mocap_nodes(
        capture.marker_world_mm,
        rear_offset_mm=rear_offset_mm,
    ).astype("<f4")
    solved = np.stack(
        [world_poses[side] for side in ANATOMICAL_SIDES], axis=1
    ).astype("<f4")
    solved[~valid] = np.nan
    mocap_path = destination / "take007_mocap.f32"
    solved_path = destination / "take007_solved.f32"
    mocap.tofile(mocap_path)
    solved.tofile(solved_path)

    metadata = {
        "schema": "gt_calib.manual_xyz_workbench.v1",
        "source_recording": capture.recording,
        "source_take": capture.take,
        "source_assets": capture_source_assets(capture),
        "video": {
            "path": clean_video.name,
            "width": video_width,
            "height": video_height,
            "source_width": source_width,
            "source_height": source_height,
            "fps": 30.0,
            "frame_count": frame_count,
            "sha256": sha256_file(clean_video),
        },
        "camera": {
            "source_width": source_width,
            "source_height": source_height,
            "matrix": capture.camera.matrix.tolist(),
            "distortion": capture.camera.distortion.tolist(),
            "world_to_color": capture.camera.world_to_color.tolist(),
            "units": "millimetres",
        },
        "rear_offset_mm": rear_offset_mm,
        "rear_offset_definition": (
            "marker #11 plus offset along normalize(#11 - mean(#4,#6,#8,#10))"
        ),
        "side_order": list(ANATOMICAL_SIDES),
        "layers": {
            "mocap": {
                "path": mocap_path.name,
                "sha256": sha256_file(mocap_path),
                "dtype": "float32-le",
                "shape": list(mocap.shape),
                "node_semantics": ["virtual_wrist"]
                + [f"marker_{number}" for number in DISPLAY_MARKER_NUMBERS],
                "chains": [list(chain) for chain in RAW_CHAINS],
            },
            "solved": {
                "path": solved_path.name,
                "sha256": sha256_file(solved_path),
                "dtype": "float32-le",
                "shape": list(solved.shape),
                "node_semantics": list(GLOVE_NAMES),
                "chains": [list(chain) for chain in GLOVE_CHAINS],
            },
        },
        "default_manual_profile": {
            "schema": "gt_calib.manual_xyz_profile.v1",
            "source_recording": capture.recording,
            "source_take": capture.take,
            "coordinate_system": "mocap_world_mm",
            "units": "mm",
            "global_world_xyz_mm": [0.0, 0.0, 0.0],
            "left_world_xyz_mm": [0.0, 0.0, 0.0],
            "right_world_xyz_mm": [0.0, 0.0, 0.0],
            "rear_offset_mm": rear_offset_mm,
        },
        "claim_boundary": (
            "Manual XYZ is an operator-selected display calibration, not independent accuracy evidence."
        ),
    }
    metadata_path = destination / "take007_alignment.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    outputs = {
        "metadata": metadata_path,
        "clean_video": clean_video,
        "mocap": mocap_path,
        "solved": solved_path,
    }
    return outputs


def render_new_three(
    project_root: Path,
    destination: Path,
    *,
    manual_profile: Path | None = None,
) -> dict[str, Path]:
    project_root = Path(project_root)
    destination = Path(destination)
    videos = destination / "videos"
    posters = destination / "posters"
    metrics = destination / "metrics"
    frame_maps = destination / "frame_maps"
    workbench = destination / "calibration-workbench"
    for directory in (videos, posters, metrics, frame_maps, workbench):
        directory.mkdir(parents=True, exist_ok=True)
    dataset_root = project_root / "thor_new4_20260831_processed"
    archive = project_root / "movementcap_20260831_worldcalib_tabletop_final.tar.gz"
    capture = load_new_capture(dataset_root, archive)
    raw_manual = load_manual_calibration(
        manual_profile, video_id="take007-mocap-markers"
    )
    solved_manual = load_manual_calibration(
        manual_profile, video_id="take007-solved"
    )
    frame_map = frame_maps / "take007_rgb_to_cmm.csv"
    write_frame_map(capture, frame_map)

    raw_video = videos / "07_take007_labeled_mocap_markers.mp4"
    render_raw_markers(
        capture,
        raw_video,
        posters / "07_take007_labeled_mocap_markers.jpg",
        metrics / "07_take007_labeled_mocap_markers.json",
        manual=raw_manual,
    )
    solved_video = videos / "08_take007_aligned_hand_pose.mp4"
    world_poses, valid, _ = render_solved_pose(
        capture,
        solved_video,
        posters / "08_take007_aligned_hand_pose.jpg",
        metrics / "08_take007_aligned_hand_pose.json",
        manual=solved_manual,
    )
    # The browser workbench always starts from the explicit default profile so
    # exported XYZ values can be fed back to the renderer without double-counting.
    if manual_profile is None:
        workbench_world = world_poses
        workbench_valid = valid
    else:
        local_poses, workbench_valid, _ = load_solved_pose(capture)
        workbench_world, _ = align_solved_pose(
            capture,
            local_poses,
            workbench_valid,
            default_manual_calibration(),
        )
    workbench_paths = export_calibration_workbench(
        capture,
        workbench_world,
        workbench_valid,
        workbench,
    )
    if manual_profile is not None:
        applied_profile = workbench / "applied_manual_profile.json"
        shutil.copy2(Path(manual_profile), applied_profile)
        workbench_paths["applied_manual_profile"] = applied_profile
    no_glove_video = videos / "09_no_glove_world_calibration.mp4"
    no_glove_rgb = (
        dataset_root / "camera_glove_recording_20260831_155410"
        / "rgbd_unpack" / "RGB.mp4"
    )
    no_glove_intrinsics = no_glove_rgb.with_name("camera_1_intrinsics.json")
    no_glove_camera = load_camera_model(
        archive,
        no_glove_intrinsics,
        expected_recording=NO_GLOVE_RECORDING,
    )
    reference_verification = verify_calibration_reference_rgb(
        archive, no_glove_rgb, no_glove_camera
    )
    render_no_glove_calibration(
        no_glove_rgb,
        no_glove_camera,
        no_glove_video,
        posters / "09_no_glove_world_calibration.jpg",
        metrics / "09_no_glove_world_calibration.json",
        reference_verification=reference_verification,
    )
    return {
        "raw_markers": raw_video,
        "solved_pose": solved_video,
        "no_glove_calibration": no_glove_video,
        "frame_map": frame_map,
        **{f"workbench_{key}": value for key, value in workbench_paths.items()},
    }
