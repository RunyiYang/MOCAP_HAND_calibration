#!/usr/bin/env python3
"""Synchronize and overlay glove and CMAvatar hand poses on the RGB videos.

The implementation is intentionally tied to the delivered data contracts:

* MP4 frame ``i`` is paired with ROS1 color message ``i``.  The ROS bag index
  supplies the real device timestamp; MP4 PTS is not used as an absolute clock.
* ``camera_cmavatar_alignment.csv`` maps the color device clock to both the
  Thor monotonic clock (glove) and the CMAvatar Windows clock (mocap).  Its host
  fields describe a later frame-poll instant, so the measured capture-to-poll
  latency is removed row by row to recover the camera exposure instant.
* CMAvatar positions are read from ``Take_*_Human.cma`` in millimetres.
* The CS-400 result supplies the fixed mocap-world -> RGB-camera transform.
* Glove keypoints have orientation and articulation but no global translation.
  Their wrist is therefore anchored to the synchronized mocap wrist.  A single
  palm-only similarity rotation/scale can be fitted on a dedicated calibration
  take and frozen for other takes; fingertips are never used in registration.

The overlay is a synchronized comparison product, not an independent absolute
accuracy measurement.  Glove-to-mocap differences always share a mocap wrist
anchor.  Use ``--registration-segment`` to avoid fitting the evaluation take.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import cv2
import numpy as np


COLOR_TOPIC = "/cam/sensor_2/frameType_2"
ROS_OP_CONNECTION = 7
ROS_OP_INDEX_DATA = 4
ROS_BAG_MAGIC = b"#ROSBAG V2.0\n"

MOCAP_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)
MOCAP_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)

GLOVE_NAMES = (
    "wrist",
    "thumb_mcp",
    "thumb_pip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)
GLOVE_CHAINS = (
    (0, 1, 2, 3),
    (0, 4, 5, 6, 7),
    (0, 8, 9, 10, 11),
    (0, 12, 13, 14, 15),
    (0, 16, 17, 18, 19),
)

# The fixed registration sees only the four non-thumb MCP vectors.  Wrist is
# the common origin and fingertips remain holdout observations.
GLOVE_PALM_INDICES = np.asarray((4, 8, 12, 16), dtype=np.int32)
MOCAP_PALM_INDICES = np.asarray((5, 9, 13, 17), dtype=np.int32)

# Quantitative comparison deliberately excludes the ambiguous 20-vs-21 thumb
# topology.  These are wrist plus all four joints on the four non-thumb fingers.
GLOVE_COMMON_INDICES = np.asarray(tuple(range(4, 20)), dtype=np.int32)
MOCAP_COMMON_INDICES = np.asarray(tuple(range(5, 21)), dtype=np.int32)
GLOVE_TIP_INDICES = np.asarray((7, 11, 15, 19), dtype=np.int32)
MOCAP_TIP_INDICES = np.asarray((8, 12, 16, 20), dtype=np.int32)
GLOVE_INTERMEDIATE_INDICES = np.asarray((5, 6, 9, 10, 13, 14, 17, 18), dtype=np.int32)
MOCAP_INTERMEDIATE_INDICES = np.asarray((6, 7, 10, 11, 14, 15, 18, 19), dtype=np.int32)

MOCAP_COLOR_BGR = (25, 205, 255)
GLOVE_COLOR_BGR = (75, 245, 90)
CONNECTOR_COLOR_BGR = (255, 145, 45)

POSE_MODE_GLOVE_WRIST = "glove-wrist-preserved"
POSE_MODE_MOCAP_ROOT = "mocap-root-fusion"
POSE_MODES = (POSE_MODE_GLOVE_WRIST, POSE_MODE_MOCAP_ROOT)
RENDER_LAYER_COMPARISON = "comparison"
RENDER_LAYER_SOLVED_POSE = "solved-pose-only"
RENDER_LAYERS = (RENDER_LAYER_COMPARISON, RENDER_LAYER_SOLVED_POSE)
GLOVE_REGISTRATION_PROFILE_SCHEMA = "gt_calib.glove_mocap_registration.v1"
LEGACY_MOCAP_ROOT_PROFILE_SCHEMA = "gt_calib.mocap_root_conditioned_articulation.v1"
PRE_EXPOSURE_MOCAP_ROOT_PROFILE_SCHEMA = (
    "gt_calib.mocap_root_conditioned_articulation_profile.v1"
)
MOCAP_ROOT_PROFILE_SCHEMA = (
    "gt_calib.mocap_root_conditioned_articulation_profile.v2"
)
MOCAP_ROOT_METRICS_SCHEMA = (
    "gt_calib.mocap_root_conditioned_articulation_metrics.v2"
)
REGISTRATION_PROFILE_ARTIFACT_TYPE = "registration_profile"
EVALUATION_METRICS_ARTIFACT_TYPE = "evaluation_metrics"
MOCAP_ROOT_FIT_JOINT_NAMES = (
    "index_mcp",
    "middle_mcp",
    "ring_mcp",
    "pinky_mcp",
)
MOCAP_ROOT_QUATERNION_SEMANTICS = (
    "empirically verified active root-local-to-mocap-world"
)
MOCAP_ROOT_CANONICAL_ROTATION_FIELD = (
    "rotation_mocap_root_local_from_glove_local"
)


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass(frozen=True)
class ClockMapping:
    device_s: np.ndarray
    # These are exposure-time clocks.  The camera timecode rows are emitted
    # when Thor polls an already-buffered frame, so the raw host timestamps are
    # later than the frame's device/acquisition timestamp by about 255 ms.
    monotonic_s: np.ndarray
    cmavatar_s: np.ndarray
    capture_to_poll_s: np.ndarray
    poll_monotonic_s: np.ndarray
    poll_cmavatar_s: np.ndarray

    def _map_targets(
        self,
        device_s: np.ndarray,
        monotonic_source_s: np.ndarray,
        cmavatar_source_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(device_s, dtype=np.float64)
        origin = self.device_s[0]
        relative_values = values - origin
        relative_device = self.device_s - origin
        # color_device_timestamp_us has microsecond resolution, while float64
        # epoch seconds have an approximately 0.238 us ULP for this dataset.
        # Half a microsecond absorbs that representation noise without treating
        # a genuinely adjacent (one-microsecond) device tick as the endpoint.
        # Snap tolerance-close values explicitly, then interpolate only the
        # in-domain subset: np.interp must never silently clamp an out-of-domain
        # timestamp.
        tolerance_s = 0.5e-6
        query = relative_values.reshape(-1).copy()
        finite = np.isfinite(query)
        near_first = finite & (np.abs(query) <= tolerance_s)
        near_last = finite & (
            np.abs(query - relative_device[-1]) <= tolerance_s
        )
        query[near_first] = 0.0
        query[near_last] = relative_device[-1]
        inside = finite & (query >= 0.0) & (query <= relative_device[-1])

        monotonic = np.full(query.shape, np.nan, dtype=np.float64)
        cmavatar = np.full(query.shape, np.nan, dtype=np.float64)
        if np.any(inside):
            monotonic[inside] = np.interp(
                query[inside], relative_device, monotonic_source_s
            )
            cmavatar[inside] = np.interp(
                query[inside], relative_device, cmavatar_source_s
            )
        return (
            monotonic.reshape(values.shape),
            cmavatar.reshape(values.shape),
            inside.reshape(values.shape),
        )

    def map(self, device_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._map_targets(
            device_s,
            self.monotonic_s,
            self.cmavatar_s,
        )

    def map_poll_time(
        self, device_s: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the legacy callback/poll clocks for diagnostic comparison."""

        return self._map_targets(
            device_s,
            self.poll_monotonic_s,
            self.poll_cmavatar_s,
        )


@dataclass(frozen=True)
class MocapSequence:
    frame_counters: np.ndarray
    # High-resolution PTP cadence affinely anchored to the CMA Windows
    # TimeStamp column.  This avoids treating its millisecond text rounding as
    # the native 120 Hz sample clock.
    times_s: np.ndarray
    display_times_s: np.ndarray
    ptp_times_s: np.ndarray
    timestamp_fit: Mapping[str, Any]
    # (frames, side[left=0,right=1], 21, xyz), millimetres in mocap world.
    points_mm: np.ndarray
    # (frames, side[left=0,right=1], wxyz), active wrist local-to-world.
    wrist_quaternions_wxyz: np.ndarray


@dataclass(frozen=True)
class GloveSequence:
    times_s: np.ndarray
    local_mm: np.ndarray
    world_mm: np.ndarray


@dataclass(frozen=True)
class PreviewCalibration:
    path: Path
    payload: Mapping[str, Any]
    world_to_color: np.ndarray
    intrinsics: np.ndarray
    distortion: np.ndarray


@dataclass(frozen=True)
class SimilarityRegistration:
    scale: float
    rotation: np.ndarray
    fit_frame_count: int
    anchor_residual_median_mm: float
    anchor_residual_p95_mm: float


@dataclass(frozen=True)
class PreparedTake:
    segment_dir: Path
    video: VideoInfo
    rgb_device_s: np.ndarray
    monotonic_s: np.ndarray
    cmavatar_s: np.ndarray
    clock_inside: np.ndarray
    clock_mapping: ClockMapping
    mocap_source_times_s: np.ndarray
    mocap_frame_counters: np.ndarray
    mocap_timestamp_fit: Mapping[str, Any]
    mocap_mm: np.ndarray
    mocap_valid: np.ndarray
    mocap_wrist_rotations: np.ndarray
    glove_world_mm: Mapping[str, np.ndarray]
    glove_valid: Mapping[str, np.ndarray]
    glove_in_world_mm: Mapping[str, np.ndarray]
    registrations: Mapping[str, SimilarityRegistration]
    registration_source_segment: str
    registration_fitted_on_target: bool
    pose_mode: str
    max_interpolation_gap_ms: float
    calibration: PreviewCalibration
    sync_report: Mapping[str, Any]


def _read_u32(handle: Any) -> int:
    value = handle.read(4)
    if len(value) != 4:
        raise EOFError
    return struct.unpack("<I", value)[0]


def _ros_header_fields(raw: bytes) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    offset = 0
    while offset < len(raw):
        if offset + 4 > len(raw):
            raise ValueError("Truncated ROS bag record field length")
        length = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        field = raw[offset : offset + length]
        offset += length
        if len(field) != length or b"=" not in field:
            raise ValueError("Malformed ROS bag record field")
        key, value = field.split(b"=", 1)
        result[key.decode("ascii")] = value
    return result


def _field_int(fields: Mapping[str, bytes], name: str) -> int:
    if name not in fields:
        raise KeyError(f"ROS bag record has no {name!r} field")
    return int.from_bytes(fields[name], "little", signed=False)


def _ros_time_seconds(raw: bytes) -> float:
    if len(raw) != 8:
        raise ValueError("ROS time field must contain sec/nsec uint32 values")
    seconds, nanoseconds = struct.unpack("<II", raw)
    return float(seconds) + float(nanoseconds) * 1e-9


def _read_ros_record(handle: Any, *, read_data: bool) -> tuple[dict[str, bytes], bytes | int]:
    header_length = _read_u32(handle)
    header = _ros_header_fields(handle.read(header_length))
    data_length = _read_u32(handle)
    if read_data:
        data = handle.read(data_length)
        if len(data) != data_length:
            raise ValueError("Truncated ROS bag record data")
        return header, data
    handle.seek(data_length, 1)
    return header, data_length


def ros1_topic_index_timestamps(bag_path: Path, topic: str = COLOR_TOPIC) -> np.ndarray:
    """Read one topic's message timestamps without decompressing ROS1 chunks."""

    source = Path(bag_path)
    size = source.stat().st_size
    with source.open("rb") as handle:
        if handle.read(len(ROS_BAG_MAGIC)) != ROS_BAG_MAGIC:
            raise ValueError(f"{source} is not a ROSBAG V2.0 file")
        bag_header, _ = _read_ros_record(handle, read_data=False)
        index_position = _field_int(bag_header, "index_pos")
        records_start = handle.tell()
        if not (records_start < index_position < size):
            raise ValueError(f"Invalid ROS bag index position in {source}")

        handle.seek(index_position)
        connection_topics: dict[int, str] = {}
        while handle.tell() < size:
            try:
                header, _ = _read_ros_record(handle, read_data=False)
            except EOFError:
                break
            if _field_int(header, "op") != ROS_OP_CONNECTION:
                continue
            # Header keys are strings.  Keep compatibility with bags that put
            # topic only in the connection header, as these Orbbec bags do.
            raw_topic = header.get("topic", b"")
            connection_topics[_field_int(header, "conn")] = raw_topic.decode(
                "utf-8", errors="replace"
            )

        candidates = [key for key, value in connection_topics.items() if value == topic]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one {topic!r} connection in {source}, found {len(candidates)}"
            )
        connection_id = candidates[0]

        timestamps: list[float] = []
        handle.seek(records_start)
        while handle.tell() < index_position:
            header_length_raw = handle.read(4)
            if not header_length_raw:
                break
            if len(header_length_raw) != 4:
                raise ValueError("Truncated ROS bag record header length")
            header_length = struct.unpack("<I", header_length_raw)[0]
            header = _ros_header_fields(handle.read(header_length))
            data_length = _read_u32(handle)
            op = _field_int(header, "op")
            if op != ROS_OP_INDEX_DATA or _field_int(header, "conn") != connection_id:
                handle.seek(data_length, 1)
                continue
            data = handle.read(data_length)
            count = _field_int(header, "count")
            if len(data) != count * 12:
                raise ValueError("Unexpected ROS1 index-data entry size")
            for offset in range(0, len(data), 12):
                timestamps.append(_ros_time_seconds(data[offset : offset + 8]))

    values = np.asarray(timestamps, dtype=np.float64)
    if len(values) < 2 or np.any(np.diff(values) <= 0.0):
        raise ValueError(f"{topic!r} timestamps are missing or non-monotonic in {source}")
    return values


def video_info(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    if min(width, height, frame_count) <= 0 or fps <= 0.0:
        raise ValueError(f"Invalid video metadata: {path}")
    return VideoInfo(Path(path), width, height, fps, frame_count)


def load_clock_mapping(path: Path) -> ClockMapping:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError(f"Not enough camera alignment rows in {path}")
    device_ns = np.asarray(
        [int(row["color_device_timestamp_us"]) * 1000 for row in rows],
        dtype=np.int64,
    )
    poll_monotonic_ns = np.asarray(
        [int(row["thor_monotonic_ns"]) for row in rows], dtype=np.int64
    )
    poll_cmavatar_ns = np.asarray(
        [int(row["cmavatar_windows_estimated_ns"]) for row in rows],
        dtype=np.int64,
    )
    poll_realtime_ns = np.asarray(
        [int(row["thor_realtime_ns"]) for row in rows], dtype=np.int64
    )
    capture_to_poll_ns = poll_realtime_ns - device_ns
    capture_to_poll = capture_to_poll_ns.astype(np.float64) * 1e-9
    if (
        np.any(~np.isfinite(capture_to_poll))
        or np.any(capture_to_poll < 0.0)
        or np.any(capture_to_poll > 1.0)
    ):
        raise ValueError(
            f"Implausible camera capture-to-poll latency in {path}; "
            "cannot recover exposure timestamps"
        )

    # cmavatar_windows_estimated_ns and thor_monotonic_ns both describe the
    # host poll instant.  Shift each row back by its measured queue latency so
    # the resulting clocks describe color_device_timestamp_us (the frame
    # acquisition instant).  Mapping the raw poll clocks makes motion capture
    # lead the image by roughly 7--8 RGB frames in this dataset.
    monotonic = (poll_monotonic_ns - capture_to_poll_ns).astype(np.float64) * 1e-9
    cmavatar = (poll_cmavatar_ns - capture_to_poll_ns).astype(np.float64) * 1e-9
    device = device_ns.astype(np.float64) * 1e-9
    poll_monotonic = poll_monotonic_ns.astype(np.float64) * 1e-9
    poll_cmavatar = poll_cmavatar_ns.astype(np.float64) * 1e-9
    order = np.argsort(device)
    device = device[order]
    monotonic = monotonic[order]
    cmavatar = cmavatar[order]
    capture_to_poll = capture_to_poll[order]
    poll_monotonic = poll_monotonic[order]
    poll_cmavatar = poll_cmavatar[order]
    unique = np.r_[True, np.diff(device) > 0.0]
    device = device[unique]
    monotonic = monotonic[unique]
    cmavatar = cmavatar[unique]
    capture_to_poll = capture_to_poll[unique]
    poll_monotonic = poll_monotonic[unique]
    poll_cmavatar = poll_cmavatar[unique]
    if len(device) < 2:
        raise ValueError("Camera device timestamps are not distinct")
    if np.any(np.diff(monotonic) <= 0.0) or np.any(np.diff(cmavatar) <= 0.0):
        raise ValueError(f"Exposure-time clock mapping is not monotonic in {path}")
    return ClockMapping(
        device,
        monotonic,
        cmavatar,
        capture_to_poll,
        poll_monotonic,
        poll_cmavatar,
    )


def _cma_entities(side: str) -> tuple[str, ...]:
    if side == "left":
        prefix, hand = "Skeleton_0", "LeftHand"
    elif side == "right":
        prefix, hand = "Skeleton_1", "RightHand"
    else:
        raise ValueError(f"Unknown hand side: {side}")
    return (
        f"{prefix}_{hand}",
        *(f"{prefix}_{hand}{finger}{index}" for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky") for index in range(1, 5)),
    )


def _parse_cmavatar_timestamp(value: str) -> float:
    parsed = datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S.%f")
    return parsed.replace(tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()


def quaternion_wxyz_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    """Convert active local-to-world wxyz quaternions to rotation matrices."""

    values = np.asarray(quaternions, dtype=np.float64)
    if values.shape[-1] != 4:
        raise ValueError(f"Quaternion array must end in 4 values, got {values.shape}")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    normalized = np.divide(
        values,
        norms,
        out=np.full_like(values, np.nan),
        where=np.isfinite(norms) & (norms > 1e-12),
    )
    w, x, y, z = np.moveaxis(normalized, -1, 0)
    result = np.empty(values.shape[:-1] + (3, 3), dtype=np.float64)
    result[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    result[..., 0, 1] = 2.0 * (x * y - z * w)
    result[..., 0, 2] = 2.0 * (x * z + y * w)
    result[..., 1, 0] = 2.0 * (x * y + z * w)
    result[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    result[..., 1, 2] = 2.0 * (y * z - x * w)
    result[..., 2, 0] = 2.0 * (x * z - y * w)
    result[..., 2, 1] = 2.0 * (y * z + x * w)
    result[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return result


def _interpolation_brackets(
    source_times: np.ndarray,
    target_times: np.ndarray,
    *,
    max_gap_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve interpolation indices with direct, symmetric sample matches.

    A target that exactly equals any source timestamp uses that one sample
    directly (``low == high``, ``alpha == span == 0``).  In particular, both
    the first and final source timestamps are valid and do not depend on the
    size of a neighbouring gap.
    """

    source = np.asarray(source_times, dtype=np.float64)
    target = np.asarray(target_times, dtype=np.float64)
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("Interpolation timestamps must be one-dimensional")
    if len(source) == 0 or np.any(~np.isfinite(source)):
        raise ValueError("Source timestamps must be finite and non-empty")
    if len(source) > 1 and np.any(np.diff(source) <= 0.0):
        raise ValueError("Source timestamps must be strictly increasing")
    if max_gap_s is not None and max_gap_s <= 0.0:
        raise ValueError("max_gap_s must be positive")

    finite_target = np.isfinite(target)
    insertion = np.searchsorted(source, target, side="left")
    candidate = np.clip(insertion, 0, len(source) - 1)
    exact = (
        finite_target
        & (insertion < len(source))
        & (target == source[candidate])
    )

    if len(source) == 1:
        low = np.zeros(target.shape, dtype=np.int64)
        high = np.zeros(target.shape, dtype=np.int64)
        span = np.zeros(target.shape, dtype=np.float64)
        alpha = np.zeros(target.shape, dtype=np.float64)
        return low, high, alpha, span, exact

    high = np.clip(insertion, 1, len(source) - 1).astype(np.int64, copy=False)
    low = high - 1
    low = np.where(exact, candidate, low).astype(np.int64, copy=False)
    high = np.where(exact, candidate, high).astype(np.int64, copy=False)
    span = source[high] - source[low]
    alpha = np.divide(
        target - source[low],
        span,
        out=np.zeros(target.shape, dtype=np.float64),
        where=span > 0.0,
    )
    valid = exact | (
        finite_target & (insertion > 0) & (insertion < len(source))
    )
    if max_gap_s is not None:
        valid &= span <= max_gap_s
    return low, high, alpha, span, valid


def interpolate_quaternions(
    source_times: np.ndarray,
    quaternions_wxyz: np.ndarray,
    target_times: np.ndarray,
    *,
    max_gap_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """SLERP active wxyz quaternions, with sign continuity and gap rejection."""

    source = np.asarray(source_times, dtype=np.float64)
    target = np.asarray(target_times, dtype=np.float64)
    quaternions = np.asarray(quaternions_wxyz, dtype=np.float64).copy()
    if quaternions.shape[0] != len(source) or quaternions.shape[-1] != 4:
        raise ValueError("Quaternion samples do not match source timestamps")
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    quaternions = np.divide(
        quaternions,
        norms,
        out=np.full_like(quaternions, np.nan),
        where=np.isfinite(norms) & (norms > 1e-12),
    )
    for index in range(1, len(quaternions)):
        dot = np.sum(quaternions[index - 1] * quaternions[index], axis=-1)
        flip = np.isfinite(dot) & (dot < 0.0)
        quaternions[index][flip] *= -1.0

    low, high, alpha, _, valid = _interpolation_brackets(
        source,
        target,
        max_gap_s=max_gap_s,
    )

    q0 = quaternions[low]
    q1 = quaternions[high]
    endpoints_finite = np.all(np.isfinite(q0), axis=(-1, -2)) & np.all(
        np.isfinite(q1), axis=(-1, -2)
    )
    valid &= endpoints_finite
    dot = np.sum(q0 * q1, axis=-1)
    flip = dot < 0.0
    q1 = np.where(flip[..., None], -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    fraction = alpha[:, None]
    near = sin_theta < 1e-8
    weight0 = np.divide(
        np.sin((1.0 - fraction) * theta),
        sin_theta,
        out=1.0 - fraction + np.zeros_like(theta),
        where=~near,
    )
    weight1 = np.divide(
        np.sin(fraction * theta),
        sin_theta,
        out=fraction + np.zeros_like(theta),
        where=~near,
    )
    result = weight0[..., None] * q0 + weight1[..., None] * q1
    result_norm = np.linalg.norm(result, axis=-1, keepdims=True)
    result = np.divide(
        result,
        result_norm,
        out=np.full_like(result, np.nan),
        where=np.isfinite(result_norm) & (result_norm > 1e-12),
    )
    result[~valid] = np.nan
    return result, valid


def load_mocap_human(path: Path) -> MocapSequence:
    source = Path(path)
    with source.open("rb") as handle:
        header_line = handle.readline().decode("utf-8-sig").rstrip("\r\n")
        fields = header_line.split("\t")
        ptp_time_index = fields.index("PtpTimeStamp")
        columns: dict[str, list[list[int]]] = {}
        quaternion_columns: dict[str, list[int]] = {}
        for side in ("left", "right"):
            root_entity = _cma_entities(side)[0]
            columns[side] = [
                [fields.index(f"{entity}_Pos_{axis}(mm)") for axis in "XYZ"]
                for entity in _cma_entities(side)
            ]
            quaternion_columns[side] = [
                fields.index(f"{root_entity}_GlobalQ_{axis}")
                for axis in ("R", "X", "Y", "Z")
            ]

        counters: list[int] = []
        display_times: list[float] = []
        ptp_times: list[float] = []
        points: list[np.ndarray] = []
        wrist_quaternions: list[np.ndarray] = []
        for raw_line in handle:
            text = raw_line.decode("utf-8").rstrip("\r\n")
            if not text:
                continue
            values = text.split("\t")
            counters.append(int(values[0]))
            display_times.append(_parse_cmavatar_timestamp(values[1]))
            ptp_times.append(float(values[ptp_time_index]))
            side_points = []
            side_quaternions = []
            for side in ("left", "right"):
                side_points.append(
                    np.asarray(
                        [[float(values[index]) for index in xyz] for xyz in columns[side]],
                        dtype=np.float64,
                    )
                )
                side_quaternions.append(
                    np.asarray(
                        [float(values[index]) for index in quaternion_columns[side]],
                        dtype=np.float64,
                    )
                )
            points.append(np.stack(side_points))
            wrist_quaternions.append(np.stack(side_quaternions))

    frame_counters = np.asarray(counters, dtype=np.int64)
    display_times_s = np.asarray(display_times, dtype=np.float64)
    ptp_times_s = np.asarray(ptp_times, dtype=np.float64)
    ptp_centered = ptp_times_s - float(np.mean(ptp_times_s))
    display_centered = display_times_s - float(np.mean(display_times_s))
    denominator = float(np.dot(ptp_centered, ptp_centered))
    if denominator <= 0.0:
        raise ValueError(f"CMAvatar PTP timestamps are degenerate in {source}")
    ptp_to_windows_slope = float(
        np.dot(ptp_centered, display_centered) / denominator
    )
    times_s = (
        float(np.mean(display_times_s)) + ptp_to_windows_slope * ptp_centered
    )
    display_residual_ms = (display_times_s - times_s) * 1000.0
    frame_elapsed_s = (frame_counters - frame_counters[0]) / 120.0
    ptp_frame_residual_us = (
        (ptp_times_s - ptp_times_s[0]) - frame_elapsed_s
    ) * 1e6
    timestamp_fit: dict[str, Any] = {
        "source": "PtpTimeStamp affine fit to millisecond Windows TimeStamp",
        "ptp_to_windows_slope": ptp_to_windows_slope,
        "display_timestamp_fit_absolute_residual_ms": {
            "median": float(np.median(np.abs(display_residual_ms))),
            "p95": float(np.percentile(np.abs(display_residual_ms), 95)),
            "max": float(np.max(np.abs(display_residual_ms))),
        },
        "ptp_vs_120hz_frame_counter_absolute_residual_us": {
            "median": float(np.median(np.abs(ptp_frame_residual_us))),
            "p95": float(np.percentile(np.abs(ptp_frame_residual_us), 95)),
            "max": float(np.max(np.abs(ptp_frame_residual_us))),
        },
        "semantic_status": (
            "high-resolution relative cadence observed; absolute PTP epoch contract "
            "was not delivered, so Windows TimeStamp remains the affine anchor"
        ),
    }
    points_mm = np.asarray(points, dtype=np.float64)
    wrist_quaternions_wxyz = np.asarray(wrist_quaternions, dtype=np.float64)
    if points_mm.ndim != 4 or points_mm.shape[1:] != (2, 21, 3):
        raise ValueError(f"Unexpected CMAvatar hand shape in {source}: {points_mm.shape}")
    if len(frame_counters) < 2 or not np.all(np.diff(frame_counters) == 1):
        raise ValueError(f"CMAvatar frame counters are not contiguous in {source}")
    if (
        np.any(np.diff(display_times_s) <= 0.0)
        or np.any(np.diff(ptp_times_s) <= 0.0)
        or np.any(np.diff(times_s) <= 0.0)
    ):
        raise ValueError(f"CMAvatar timestamps are not increasing in {source}")
    if not (0.999 <= ptp_to_windows_slope <= 1.001):
        raise ValueError(f"Implausible CMAvatar PTP/Windows clock slope in {source}")
    if timestamp_fit["display_timestamp_fit_absolute_residual_ms"]["p95"] > 5.0:
        raise ValueError(f"CMAvatar PTP/Windows timestamp fit is unstable in {source}")
    if wrist_quaternions_wxyz.shape != (len(points_mm), 2, 4):
        raise ValueError(
            f"Unexpected CMAvatar wrist quaternion shape in {source}: "
            f"{wrist_quaternions_wxyz.shape}"
        )
    return MocapSequence(
        frame_counters,
        times_s,
        display_times_s,
        ptp_times_s,
        timestamp_fit,
        points_mm,
        wrist_quaternions_wxyz,
    )


def load_glove_keypoints(path: Path) -> GloveSequence:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        fields = next(reader)
        time_index = fields.index("host_monotonic_ns")
        local_indices = [
            [fields.index(f"local_{name}_{axis}_m") for axis in "xyz"]
            for name in GLOVE_NAMES
        ]
        world_indices = [
            [fields.index(f"world_{name}_{axis}_m") for axis in "xyz"]
            for name in GLOVE_NAMES
        ]
        times: list[float] = []
        local: list[np.ndarray] = []
        world: list[np.ndarray] = []
        for row in reader:
            times.append(int(row[time_index]) * 1e-9)
            local.append(
                np.asarray(
                    [[float(row[index]) * 1000.0 for index in xyz] for xyz in local_indices],
                    dtype=np.float64,
                )
            )
            world.append(
                np.asarray(
                    [[float(row[index]) * 1000.0 for index in xyz] for xyz in world_indices],
                    dtype=np.float64,
                )
            )
    times_s = np.asarray(times, dtype=np.float64)
    local_mm = np.asarray(local, dtype=np.float64)
    world_mm = np.asarray(world, dtype=np.float64)
    if local_mm.shape != world_mm.shape or local_mm.shape[1:] != (20, 3):
        raise ValueError(f"Unexpected glove keypoint shape in {source}")
    if len(times_s) < 2 or np.any(np.diff(times_s) <= 0.0):
        raise ValueError(f"Glove timestamps are not strictly increasing in {source}")
    wrist_error = max(
        float(np.max(np.abs(local_mm[:, 0]))),
        float(np.max(np.abs(world_mm[:, 0]))),
    )
    if wrist_error > 1e-6:
        raise ValueError(f"Glove wrist is not origin-centred in {source}")
    return GloveSequence(times_s, local_mm, world_mm)


def interpolate_series(
    source_times: np.ndarray,
    values: np.ndarray,
    target_times: np.ndarray,
    *,
    max_gap_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_times, dtype=np.float64)
    target = np.asarray(target_times, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    if data.ndim == 0 or data.shape[0] != len(source):
        raise ValueError("Samples do not match source timestamps")
    low, high, alpha, _, valid = _interpolation_brackets(
        source,
        target,
        max_gap_s=max_gap_s,
    )
    shape = (len(target),) + (1,) * (data.ndim - 1)
    result = data[low] * (1.0 - alpha.reshape(shape)) + data[high] * alpha.reshape(shape)
    result[~valid] = np.nan
    return result, valid


def load_preview_calibration(path: Path) -> PreviewCalibration:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "movementcap.camera_to_mocap_world.v1":
        raise ValueError(f"Unsupported camera/world calibration schema in {source}")
    transforms = payload["transforms"]
    world_to_color = np.asarray(transforms["world_to_color_camera"], dtype=np.float64)
    color = payload["orbbec_profiles_from_bag"]["color"]
    intrinsics = np.asarray(color["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
    distortion = np.asarray(
        color["distortion_orbbec_k1_k2_k3_k4_k5_k6_p1_p2"], dtype=np.float64
    )
    if world_to_color.shape != (4, 4) or intrinsics.shape != (4,) or distortion.shape != (8,):
        raise ValueError(f"Malformed camera/world calibration arrays in {source}")
    return PreviewCalibration(source, payload, world_to_color, intrinsics, distortion)


def distort_normalized(
    x: np.ndarray, y: np.ndarray, coefficients: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    k1, k2, k3, k4, k5, k6, p1, p2 = np.asarray(coefficients, dtype=np.float64)
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (
        1.0 + k4 * r2 + k5 * r4 + k6 * r6
    )
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xd, yd


def project_world_to_rgb(
    points_world_mm: np.ndarray, calibration: PreviewCalibration
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world_mm, dtype=np.float64)
    camera = points @ calibration.world_to_color[:3, :3].T + calibration.world_to_color[:3, 3]
    positive = camera[..., 2] > 1e-6
    safe_z = np.where(positive, camera[..., 2], np.nan)
    x = camera[..., 0] / safe_z
    y = camera[..., 1] / safe_z
    xd, yd = distort_normalized(x, y, calibration.distortion)
    fx, fy, cx, cy = calibration.intrinsics
    pixels = np.stack((fx * xd + cx, fy * yd + cy), axis=-1)
    return pixels, positive


def fit_fixed_palm_registration(
    glove_root_relative_mm: np.ndarray,
    mocap_world_mm: np.ndarray,
    valid: np.ndarray,
    *,
    mocap_wrist_rotations: np.ndarray | None = None,
) -> SimilarityRegistration:
    mask = np.asarray(valid, dtype=bool)
    if np.count_nonzero(mask) < 10:
        raise ValueError("Too few synchronized frames for palm registration")
    source = glove_root_relative_mm[mask][:, GLOVE_PALM_INDICES]
    selected_mocap = mocap_world_mm[mask]
    target = (
        selected_mocap[:, MOCAP_PALM_INDICES]
        - selected_mocap[:, 0, None, :]
    )
    if mocap_wrist_rotations is not None:
        rotations = np.asarray(mocap_wrist_rotations, dtype=np.float64)[mask]
        if rotations.shape != (len(source), 3, 3):
            raise ValueError(
                f"Unexpected mocap wrist rotation shape for registration: {rotations.shape}"
            )
        # Row-vector convention: local = (world - wrist) @ R_world_from_wrist.
        target = np.einsum("fji,fik->fjk", target, rotations)

    def solve(use: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        x = source[use].reshape(-1, 3)
        y = target[use].reshape(-1, 3)
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("Palm registration contains non-finite points")
        covariance = x.T @ y
        left, singular_values, right_t = np.linalg.svd(covariance)
        if singular_values[0] <= 1e-12 or singular_values[1] <= singular_values[0] * 1e-8:
            raise ValueError("Palm registration geometry is degenerate")
        rotation = right_t.T @ left.T
        if np.linalg.det(rotation) < 0.0:
            right_t[-1] *= -1.0
            rotation = right_t.T @ left.T
        rotated = x @ rotation.T
        denominator = float(np.sum(rotated * rotated))
        if denominator <= 1e-12:
            raise ValueError("Palm registration has zero source extent")
        scale = float(np.sum(rotated * y) / denominator)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("Palm registration produced an invalid scale")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-6
        ):
            raise ValueError("Palm registration did not produce a proper rotation")
        predicted = source[use] @ rotation.T * scale
        residual = np.linalg.norm(predicted - target[use], axis=-1)
        return scale, rotation, residual

    keep = np.ones(len(source), dtype=bool)
    scale, rotation, residual = solve(keep)
    frame_residual = np.median(residual, axis=1)
    median = float(np.median(frame_residual))
    mad = float(np.median(np.abs(frame_residual - median)))
    threshold = median + max(3.0 * 1.4826 * mad, 3.0)
    refined = frame_residual <= threshold
    if np.count_nonzero(refined) >= max(10, len(source) // 2):
        scale, rotation, _ = solve(refined)
        keep = refined
    predicted = source @ rotation.T * scale
    residual = np.linalg.norm(predicted - target, axis=-1)
    retained = residual[keep]
    return SimilarityRegistration(
        scale=scale,
        rotation=rotation,
        fit_frame_count=int(np.count_nonzero(keep)),
        anchor_residual_median_mm=float(np.median(retained)),
        anchor_residual_p95_mm=float(np.percentile(retained, 95)),
    )


def apply_registration(
    glove_root_relative_mm: np.ndarray,
    mocap_world_mm: np.ndarray,
    registration: SimilarityRegistration,
    *,
    mocap_wrist_rotations: np.ndarray | None = None,
) -> np.ndarray:
    relative = (
        np.asarray(glove_root_relative_mm)
        @ registration.rotation.T
        * registration.scale
    )
    if mocap_wrist_rotations is not None:
        rotations = np.asarray(mocap_wrist_rotations, dtype=np.float64)
        if rotations.shape != (len(relative), 3, 3):
            raise ValueError(
                f"Unexpected mocap wrist rotation shape for application: {rotations.shape}"
            )
        # Row-vector convention: world = local @ R_world_from_wrist.T.
        relative = np.einsum("fji,fki->fjk", relative, rotations)
    return relative + np.asarray(mocap_world_mm)[:, 0, None, :]


def _registration_to_payload(
    registration: SimilarityRegistration,
    *,
    pose_mode: str = POSE_MODE_GLOVE_WRIST,
) -> dict[str, Any]:
    payload = {
        "scale": registration.scale,
        # Retained for compatibility with existing profiles and consumers.
        "rotation_mocap_from_glove": registration.rotation.tolist(),
        "fit_frame_count": registration.fit_frame_count,
        "source_fit_anchor_residual_mm": {
            "median": registration.anchor_residual_median_mm,
            "p95": registration.anchor_residual_p95_mm,
        },
    }
    if pose_mode == POSE_MODE_MOCAP_ROOT:
        payload[MOCAP_ROOT_CANONICAL_ROTATION_FIELD] = registration.rotation.tolist()
    return payload


def _registration_from_payload(payload: Mapping[str, Any]) -> SimilarityRegistration:
    canonical_value = payload.get(MOCAP_ROOT_CANONICAL_ROTATION_FIELD)
    legacy_value = payload.get("rotation_mocap_from_glove")
    rotation_value = canonical_value if canonical_value is not None else legacy_value
    if rotation_value is None:
        raise ValueError(
            "Registration payload is missing both canonical and legacy rotation fields"
        )
    rotation = np.asarray(rotation_value, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("Registration rotation must be 3x3")
    if canonical_value is not None and legacy_value is not None:
        legacy_rotation = np.asarray(legacy_value, dtype=np.float64)
        if legacy_rotation.shape != (3, 3) or not np.allclose(
            rotation, legacy_rotation, atol=1e-12, rtol=0.0
        ):
            raise ValueError("Canonical and legacy registration rotations disagree")
    residual = payload.get("source_fit_anchor_residual_mm")
    if not isinstance(residual, Mapping):
        raise ValueError("Registration payload is missing anchor residual metrics")
    try:
        registration = SimilarityRegistration(
            scale=float(payload["scale"]),
            rotation=rotation,
            fit_frame_count=int(payload["fit_frame_count"]),
            anchor_residual_median_mm=float(residual["median"]),
            anchor_residual_p95_mm=float(residual["p95"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Registration payload contains invalid numeric fields") from error
    if not np.isfinite(registration.scale) or registration.scale <= 0.0:
        raise ValueError("Registration scale must be finite and positive")
    if registration.fit_frame_count <= 0:
        raise ValueError("Registration fit frame count must be positive")
    if (
        not np.isfinite(registration.anchor_residual_median_mm)
        or not np.isfinite(registration.anchor_residual_p95_mm)
        or registration.anchor_residual_median_mm < 0.0
        or registration.anchor_residual_p95_mm < registration.anchor_residual_median_mm
    ):
        raise ValueError("Registration anchor residual metrics are invalid")
    if not np.all(np.isfinite(rotation)):
        raise ValueError("Registration rotation contains non-finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError("Registration rotation is not a proper rotation matrix")
    return registration


def _require_profile_value(
    payload: Mapping[str, Any],
    key: str,
    expected: Any,
    *,
    context: str,
) -> None:
    value = payload.get(key)
    matches = value is expected if isinstance(expected, bool) else value == expected
    if not matches:
        raise ValueError(
            f"{context} requires {key}={expected!r}, got {value!r}"
        )


def _validate_root_registration_profile(
    payload: Mapping[str, Any],
    *,
    schema: str,
) -> None:
    context = "MOCAP root registration profile"
    if schema == MOCAP_ROOT_PROFILE_SCHEMA:
        _require_profile_value(
            payload,
            "artifact_type",
            REGISTRATION_PROFILE_ARTIFACT_TYPE,
            context=context,
        )
    elif "artifact_type" in payload:
        _require_profile_value(
            payload,
            "artifact_type",
            REGISTRATION_PROFILE_ARTIFACT_TYPE,
            context=context,
        )

    _require_profile_value(
        payload, "pose_mode", POSE_MODE_MOCAP_ROOT, context=context
    )
    _require_profile_value(
        payload,
        "fit_joint_names",
        list(MOCAP_ROOT_FIT_JOINT_NAMES),
        context=context,
    )
    _require_profile_value(
        payload, "target_finger_joints_used_in_fit", False, context=context
    )
    _require_profile_value(
        payload, "fingertips_used_in_fit", False, context=context
    )
    _require_profile_value(
        payload,
        "quaternion_semantics",
        MOCAP_ROOT_QUATERNION_SEMANTICS,
        context=context,
    )
    _require_profile_value(
        payload,
        "translation_policy",
        "evaluation mocap wrist copied per frame",
        context=context,
    )
    _require_profile_value(
        payload, "absolute_wrist_translation_evaluable", False, context=context
    )

    source_segment = payload.get("source_segment")
    if not isinstance(source_segment, str) or not source_segment.strip():
        raise ValueError(f"{context} requires a non-empty source_segment")
    try:
        max_gap_ms = float(payload["max_interpolation_gap_ms"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{context} has an invalid max_interpolation_gap_ms") from error
    if not np.isfinite(max_gap_ms) or max_gap_ms <= 0.0:
        raise ValueError(f"{context} max_interpolation_gap_ms must be positive")

    if schema == MOCAP_ROOT_PROFILE_SCHEMA:
        timestamp_alignment = payload.get("timestamp_alignment")
        if not isinstance(timestamp_alignment, Mapping):
            raise ValueError(f"{context} requires timestamp_alignment")
        required_timestamp_contract = {
            "camera_timestamp": "color_device_timestamp_us acquisition time",
            "host_timestamp": "thor poll/receive time",
            "capture_to_poll_latency_removed_row_by_row": True,
            "glove_target_clock": "exposure-corrected thor_monotonic_ns",
            "mocap_target_clock": (
                "exposure-corrected cmavatar_windows_estimated_ns"
            ),
            "mocap_sample_clock": (
                "PtpTimeStamp high-resolution cadence affinely anchored to "
                "millisecond Windows TimeStamp"
            ),
        }
        for key, expected in required_timestamp_contract.items():
            _require_profile_value(
                timestamp_alignment,
                key,
                expected,
                context=f"{context} timestamp_alignment",
            )

    root_conditioning = payload.get("root_conditioning")
    if not isinstance(root_conditioning, Mapping):
        raise ValueError(f"{context} requires root_conditioning")
    required_root_contract = {
        "conditional_on_mocap_wrist_translation": True,
        "conditional_on_mocap_wrist_orientation": True,
        "target_finger_joints_used_in_fit": False,
        "mocap_wrist_quaternion_fields": "GlobalQ_R/X/Y/Z",
        "quaternion_storage_order": "wxyz",
        "quaternion_semantics": MOCAP_ROOT_QUATERNION_SEMANTICS,
        "rotation_semantics": "active wrist-local to mocap-world",
        "interpolation": "sign-continuous SLERP",
        "glove_input": "root-relative local keypoints",
    }
    for key, expected in required_root_contract.items():
        _require_profile_value(
            root_conditioning,
            key,
            expected,
            context=f"{context} root_conditioning",
        )

    sides = payload.get("sides")
    if not isinstance(sides, Mapping) or set(sides) != {"left", "right"}:
        raise ValueError(f"{context} must contain exactly left and right sides")
    for side in ("left", "right"):
        side_payload = sides[side]
        if not isinstance(side_payload, Mapping):
            raise ValueError(f"{context} side {side!r} must be an object")
        if MOCAP_ROOT_CANONICAL_ROTATION_FIELD not in side_payload:
            raise ValueError(
                f"{context} side {side!r} is missing "
                f"{MOCAP_ROOT_CANONICAL_ROTATION_FIELD}"
            )
        if "rotation_mocap_from_glove" not in side_payload:
            raise ValueError(
                f"{context} side {side!r} is missing the legacy rotation alias"
            )


def _validate_legacy_glove_registration_profile(
    payload: Mapping[str, Any],
) -> None:
    context = "Legacy glove registration profile"
    pose_mode = payload.get("pose_mode")
    if pose_mode is not None and pose_mode != POSE_MODE_GLOVE_WRIST:
        raise ValueError(
            f"{context} requires pose_mode={POSE_MODE_GLOVE_WRIST!r}, "
            f"got {pose_mode!r}"
        )
    if "artifact_type" in payload:
        _require_profile_value(
            payload,
            "artifact_type",
            REGISTRATION_PROFILE_ARTIFACT_TYPE,
            context=context,
        )
    source_segment = payload.get("source_segment")
    if not isinstance(source_segment, str) or not source_segment.strip():
        raise ValueError(f"{context} requires a non-empty source_segment")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _glove_solver_provenance(segment: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for side in ("left", "right"):
        meta_path = (
            segment
            / "手套解算"
            / "solved"
            / "primary"
            / f"{side}_hand_pose.meta.json"
        )
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        result[side] = {
            "meta_path": str(meta_path),
            "solver_id": payload.get("solver_id"),
            "solver_version": payload.get("solver_version"),
            "config_sha256": payload.get("config_sha256"),
            "session_neutral_used": payload.get("session_neutral", {}).get("used"),
            "session_neutral_captured_at_utc": payload.get("session_neutral", {}).get(
                "captured_at_utc"
            ),
        }
    return result


def registration_profile_payload(prepared: PreparedTake) -> dict[str, Any]:
    """Serialize the fixed palm registration fitted on one source take."""

    if not prepared.registration_fitted_on_target:
        raise ValueError("Cannot export a registration profile from a holdout take")
    take_number = _take_number(prepared.segment_dir)
    human_cma = (
        prepared.segment_dir
        / "动捕"
        / f"Take_{take_number}"
        / f"Take_{take_number}_Human.cma"
    )
    input_files: dict[str, Any] = {
        "human_cma": {
            "path": str(human_cma),
            "sha256": _sha256_file(human_cma),
        }
    }
    for side in ("left", "right"):
        keypoints = (
            prepared.segment_dir
            / "手套解算"
            / "solved"
            / "primary"
            / f"{side}_hand_keypoints.csv"
        )
        input_files[f"{side}_glove_keypoints"] = {
            "path": str(keypoints),
            "sha256": _sha256_file(keypoints),
        }
    payload = {
        "schema": (
            MOCAP_ROOT_PROFILE_SCHEMA
            if prepared.pose_mode == POSE_MODE_MOCAP_ROOT
            else GLOVE_REGISTRATION_PROFILE_SCHEMA
        ),
        **(
            {"artifact_type": REGISTRATION_PROFILE_ARTIFACT_TYPE}
            if prepared.pose_mode == POSE_MODE_MOCAP_ROOT
            else {}
        ),
        "pose_mode": prepared.pose_mode,
        "source_segment": prepared.segment_dir.name,
        "camera_world_calibration": {
            "path": str(prepared.calibration.path),
            "sha256": _sha256_file(prepared.calibration.path),
        },
        "input_files": input_files,
        "glove_solver": _glove_solver_provenance(prepared.segment_dir),
        "fit_joint_names": list(MOCAP_ROOT_FIT_JOINT_NAMES),
        "target_finger_joints_used_in_fit": False,
        "fingertips_used_in_fit": False,
        "translation_policy": "evaluation mocap wrist copied per frame",
        "absolute_wrist_translation_evaluable": False,
        "max_interpolation_gap_ms": prepared.max_interpolation_gap_ms,
        "timestamp_alignment": {
            "camera_timestamp": "color_device_timestamp_us acquisition time",
            "host_timestamp": "thor poll/receive time",
            "capture_to_poll_latency_removed_row_by_row": True,
            "camera_capture_to_thor_poll_latency_ms": _percentiles(
                prepared.clock_mapping.capture_to_poll_s * 1000.0
            ),
            "glove_target_clock": "exposure-corrected thor_monotonic_ns",
            "mocap_target_clock": (
                "exposure-corrected cmavatar_windows_estimated_ns"
            ),
            "mocap_sample_clock": (
                "PtpTimeStamp high-resolution cadence affinely anchored to "
                "millisecond Windows TimeStamp"
            ),
            "mocap_timestamp_fit": prepared.mocap_timestamp_fit,
        },
        "sides": {
            side: _registration_to_payload(
                prepared.registrations[side], pose_mode=prepared.pose_mode
            )
            for side in ("left", "right")
        },
    }
    if prepared.pose_mode == POSE_MODE_MOCAP_ROOT:
        payload["quaternion_semantics"] = MOCAP_ROOT_QUATERNION_SEMANTICS
        payload["root_conditioning"] = {
            "conditional_on_mocap_wrist_translation": True,
            "conditional_on_mocap_wrist_orientation": True,
            "target_finger_joints_used_in_fit": False,
            "mocap_wrist_quaternion_fields": "GlobalQ_R/X/Y/Z",
            "quaternion_storage_order": "wxyz",
            "quaternion_semantics": MOCAP_ROOT_QUATERNION_SEMANTICS,
            "rotation_semantics": "active wrist-local to mocap-world",
            "interpolation": "sign-continuous SLERP",
            "glove_input": "root-relative local keypoints",
            "claim_boundary": (
                "calibrated glove articulation on a MOCAP wrist SE(3); "
                "not an independent glove wrist or full 6DoF measurement"
            ),
        }
    return payload


def write_registration_profile(path: Path, prepared: PreparedTake) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(registration_profile_payload(prepared), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return output


def load_registration_profile(
    path: Path,
) -> tuple[str, dict[str, SimilarityRegistration], Mapping[str, Any]]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Registration profile must be a JSON object: {source}")
    schema = payload.get("schema")
    if schema == GLOVE_REGISTRATION_PROFILE_SCHEMA:
        _validate_legacy_glove_registration_profile(payload)
    elif schema in {
        LEGACY_MOCAP_ROOT_PROFILE_SCHEMA,
        PRE_EXPOSURE_MOCAP_ROOT_PROFILE_SCHEMA,
        MOCAP_ROOT_PROFILE_SCHEMA,
    }:
        _validate_root_registration_profile(payload, schema=str(schema))
    else:
        raise ValueError(f"Unsupported registration profile schema in {source}")
    sides = payload.get("sides", {})
    if not isinstance(sides, Mapping) or set(sides) != {"left", "right"}:
        raise ValueError(f"Registration profile must contain left and right hands: {source}")
    if schema == GLOVE_REGISTRATION_PROFILE_SCHEMA:
        for side in ("left", "right"):
            side_payload = sides[side]
            if not isinstance(side_payload, Mapping):
                raise ValueError(f"Registration profile side {side!r} must be an object")
            if "rotation_mocap_from_glove" not in side_payload:
                raise ValueError(
                    f"Legacy glove registration side {side!r} is missing "
                    "rotation_mocap_from_glove"
                )
    registrations = {
        side: _registration_from_payload(sides[side]) for side in ("left", "right")
    }
    return str(payload["source_segment"]), registrations, payload


def _discover_segment(dataset_root: Path, key: str) -> Path:
    normalized = key.strip()
    if normalized.startswith("Take_"):
        matches = sorted(dataset_root.glob(f"*_{normalized}"))
    elif normalized.isdigit() and len(normalized) <= 3:
        matches = sorted(dataset_root.glob(f"{int(normalized):02d}_*"))
    else:
        candidate = dataset_root / normalized
        matches = [candidate] if candidate.is_dir() else []
    matches = [item for item in matches if item.is_dir() and not item.name.startswith("00_")]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one non-00 segment for {key!r} below {dataset_root}, found {len(matches)}"
        )
    return matches[0]


def _take_number(segment: Path) -> str:
    if "Take_" not in segment.name:
        raise ValueError(f"Segment name does not contain Take_: {segment.name}")
    return segment.name.split("Take_", 1)[1]


def _default_calibration(dataset_root: Path) -> Path:
    matches = sorted(
        dataset_root.glob(
            "movementcap_worldcalib_pointcloud_package_*/movementcap_ruler_worldcalib/"
            "results/manual_final/camera_to_world.json"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one delivered camera_to_world.json below {dataset_root}, found {len(matches)}"
        )
    return matches[0]


def prepare_take(
    segment_dir: Path,
    calibration_path: Path,
    *,
    registrations: Mapping[str, SimilarityRegistration] | None = None,
    registration_source_segment: str | None = None,
    max_interpolation_gap_ms: float = 25.0,
    pose_mode: str = POSE_MODE_GLOVE_WRIST,
) -> PreparedTake:
    segment = Path(segment_dir)
    if pose_mode not in POSE_MODES:
        raise ValueError(f"Unsupported pose mode: {pose_mode}")
    if max_interpolation_gap_ms <= 0.0:
        raise ValueError("max_interpolation_gap_ms must be positive")
    max_gap_s = max_interpolation_gap_ms * 1e-3
    if registrations is not None and registration_source_segment is None:
        raise ValueError(
            "registration_source_segment is required with external registrations"
        )
    if registrations is None and registration_source_segment not in (None, segment.name):
        raise ValueError(
            "registration_source_segment cannot name another take without external registrations"
        )
    if registrations is not None and set(registrations) != {"left", "right"}:
        raise ValueError("External registrations must contain left and right hands")
    source_segment = registration_source_segment or segment.name
    video = video_info(segment / "视频" / "RGB.mp4")
    rgb_device_s = ros1_topic_index_timestamps(
        segment / "原始BAG与内参" / "camera_1_rgb_depth.bag"
    )
    if len(rgb_device_s) != video.frame_count:
        raise ValueError(
            f"RGB bag messages ({len(rgb_device_s)}) do not match MP4 frames ({video.frame_count})"
        )
    clocks = load_clock_mapping(segment / "同步校验" / "camera_cmavatar_alignment.csv")
    monotonic_s, cmavatar_s, clock_inside = clocks.map(rgb_device_s)

    take_number = _take_number(segment)
    mocap = load_mocap_human(
        segment / "动捕" / f"Take_{take_number}" / f"Take_{take_number}_Human.cma"
    )
    mocap_mm, mocap_valid = interpolate_series(
        mocap.times_s,
        mocap.points_mm,
        cmavatar_s,
        max_gap_s=max_gap_s,
    )
    mocap_wrist_quaternions, mocap_rotation_valid = interpolate_quaternions(
        mocap.times_s,
        mocap.wrist_quaternions_wxyz,
        cmavatar_s,
        max_gap_s=max_gap_s,
    )
    mocap_wrist_rotations = quaternion_wxyz_to_matrix(mocap_wrist_quaternions)
    calibration = load_preview_calibration(calibration_path)
    applies_to = set(calibration.payload.get("applies_to", []))
    if segment.name not in applies_to:
        raise ValueError(
            f"Calibration {calibration.path} does not declare applicability to {segment.name}"
        )

    sync_report = json.loads(
        (segment / "同步校验" / "common_interval_sync_report.json").read_text(
            encoding="utf-8"
        )
    )
    common_start = int(sync_report["cmavatar_start_ns"]) * 1e-9
    common_end = int(sync_report["cmavatar_end_ns"]) * 1e-9
    mocap_valid &= (
        (cmavatar_s >= common_start)
        & (cmavatar_s <= common_end)
        & clock_inside
        & np.all(np.isfinite(mocap_mm), axis=(1, 2, 3))
        & mocap_rotation_valid
        & np.all(np.isfinite(mocap_wrist_rotations), axis=(1, 2, 3))
    )

    glove_world: dict[str, np.ndarray] = {}
    glove_valid: dict[str, np.ndarray] = {}
    glove_in_world: dict[str, np.ndarray] = {}
    active_registrations: dict[str, SimilarityRegistration] = {}
    for side_index, side in enumerate(("left", "right")):
        glove = load_glove_keypoints(
            segment / "手套解算" / "solved" / "primary" / f"{side}_hand_keypoints.csv"
        )
        interpolated, valid = interpolate_series(
            glove.times_s,
            (
                glove.local_mm
                if pose_mode == POSE_MODE_MOCAP_ROOT
                else glove.world_mm
            ),
            monotonic_s,
            max_gap_s=max_gap_s,
        )
        valid &= mocap_valid & np.all(np.isfinite(interpolated), axis=(1, 2))
        registration = (
            registrations[side]
            if registrations is not None
            else fit_fixed_palm_registration(
                interpolated,
                mocap_mm[:, side_index],
                valid,
                mocap_wrist_rotations=(
                    mocap_wrist_rotations[:, side_index]
                    if pose_mode == POSE_MODE_MOCAP_ROOT
                    else None
                ),
            )
        )
        glove_world[side] = interpolated
        glove_valid[side] = valid
        glove_in_world[side] = apply_registration(
            interpolated,
            mocap_mm[:, side_index],
            registration,
            mocap_wrist_rotations=(
                mocap_wrist_rotations[:, side_index]
                if pose_mode == POSE_MODE_MOCAP_ROOT
                else None
            ),
        )
        active_registrations[side] = registration

    return PreparedTake(
        segment_dir=segment,
        video=video,
        rgb_device_s=rgb_device_s,
        monotonic_s=monotonic_s,
        cmavatar_s=cmavatar_s,
        clock_inside=clock_inside,
        clock_mapping=clocks,
        mocap_source_times_s=mocap.times_s,
        mocap_frame_counters=mocap.frame_counters,
        mocap_timestamp_fit=mocap.timestamp_fit,
        mocap_mm=mocap_mm,
        mocap_valid=mocap_valid,
        mocap_wrist_rotations=mocap_wrist_rotations,
        glove_world_mm=glove_world,
        glove_valid=glove_valid,
        glove_in_world_mm=glove_in_world,
        registrations=active_registrations,
        registration_source_segment=source_segment,
        registration_fitted_on_target=source_segment == segment.name,
        pose_mode=pose_mode,
        max_interpolation_gap_ms=max_interpolation_gap_ms,
        calibration=calibration,
        sync_report=sync_report,
    )


def _finite_pixel(point: np.ndarray) -> bool:
    return bool(np.asarray(point).shape == (2,) and np.all(np.isfinite(point)))


def _draw_chain(
    image: np.ndarray,
    pixels: np.ndarray,
    chain: Sequence[int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    height, width = image.shape[:2]
    rectangle = (0, 0, width, height)
    for first, second in zip(chain, chain[1:]):
        if not (_finite_pixel(pixels[first]) and _finite_pixel(pixels[second])):
            continue
        p1 = tuple(np.rint(pixels[first]).astype(int))
        p2 = tuple(np.rint(pixels[second]).astype(int))
        visible, clipped_first, clipped_second = cv2.clipLine(rectangle, p1, p2)
        if visible:
            cv2.line(
                image,
                clipped_first,
                clipped_second,
                color,
                thickness,
                cv2.LINE_AA,
            )


def draw_hand_skeleton(
    image: np.ndarray,
    pixels: np.ndarray,
    chains: Sequence[Sequence[int]],
    color: tuple[int, int, int],
    *,
    thickness: int,
    radius: int,
    hollow: bool,
) -> None:
    for chain in chains:
        _draw_chain(image, pixels, chain, color, thickness)
    height, width = image.shape[:2]
    for point in pixels:
        if not _finite_pixel(point):
            continue
        x, y = np.rint(point).astype(int)
        if -radius <= x < width + radius and -radius <= y < height + radius:
            cv2.circle(
                image,
                (int(x), int(y)),
                radius,
                color,
                2 if hollow else -1,
                cv2.LINE_AA,
            )


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.58,
    color: tuple[int, int, int] = (245, 245, 245),
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (8, 8, 8),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def _frame_difference_mm(
    glove_world: np.ndarray, mocap_world: np.ndarray
) -> tuple[float, float]:
    glove_relative = glove_world - glove_world[0]
    mocap_relative = mocap_world - mocap_world[0]
    common = np.linalg.norm(
        glove_relative[GLOVE_COMMON_INDICES] - mocap_relative[MOCAP_COMMON_INDICES],
        axis=1,
    )
    tips = np.linalg.norm(
        glove_relative[GLOVE_TIP_INDICES] - mocap_relative[MOCAP_TIP_INDICES],
        axis=1,
    )
    return float(np.median(common)), float(np.median(tips))


def render_frame(
    prepared: PreparedTake,
    frame_index: int,
    frame: np.ndarray,
    *,
    render_layer: str = RENDER_LAYER_COMPARISON,
) -> np.ndarray:
    if render_layer not in RENDER_LAYERS:
        raise ValueError(f"Unsupported render layer: {render_layer}")
    solved_pose_only = render_layer == RENDER_LAYER_SOLVED_POSE
    output = frame.copy()
    valid = bool(prepared.mocap_valid[frame_index])
    side_stats: dict[str, tuple[float, float]] = {}
    side_pose_valid: dict[str, bool] = {"left": False, "right": False}
    if valid:
        for side_index, side in enumerate(("left", "right")):
            mocap_world = prepared.mocap_mm[frame_index, side_index]
            mocap_pixels, mocap_positive = project_world_to_rgb(
                mocap_world, prepared.calibration
            )
            mocap_pixels[~mocap_positive] = np.nan
            if not solved_pose_only:
                draw_hand_skeleton(
                    output,
                    mocap_pixels,
                    MOCAP_CHAINS,
                    MOCAP_COLOR_BGR,
                    thickness=5,
                    radius=5,
                    hollow=True,
                )

            if prepared.glove_valid[side][frame_index]:
                side_pose_valid[side] = True
                glove_world = prepared.glove_in_world_mm[side][frame_index]
                glove_pixels, glove_positive = project_world_to_rgb(
                    glove_world, prepared.calibration
                )
                glove_pixels[~glove_positive] = np.nan
                draw_hand_skeleton(
                    output,
                    glove_pixels,
                    GLOVE_CHAINS,
                    GLOVE_COLOR_BGR,
                    thickness=5 if solved_pose_only else 3,
                    radius=5 if solved_pose_only else 3,
                    hollow=solved_pose_only,
                )
                if not solved_pose_only:
                    for glove_index, mocap_index in zip(
                        GLOVE_TIP_INDICES, MOCAP_TIP_INDICES, strict=True
                    ):
                        _draw_chain(
                            output,
                            np.vstack(
                                (glove_pixels[glove_index], mocap_pixels[mocap_index])
                            ),
                            (0, 1),
                            CONNECTOR_COLOR_BGR,
                            1,
                        )
                    side_stats[side] = _frame_difference_mm(
                        glove_world, mocap_world
                    )

            wrist = (
                glove_pixels[0]
                if solved_pose_only and side_pose_valid[side]
                else mocap_pixels[0]
            )
            if _finite_pixel(wrist):
                x, y = np.rint(wrist).astype(int)
                _draw_text(
                    output,
                    "L" if side == "left" else "R",
                    (int(x + 9), int(y - 9)),
                    scale=0.55,
                )

    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (output.shape[1], 78), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, output, 0.42, 0.0, output)
    elapsed = prepared.rgb_device_s[frame_index] - prepared.rgb_device_s[0]
    _draw_text(
        output,
        f"{prepared.segment_dir.name}  frame {frame_index}/{prepared.video.frame_count - 1}  "
        f"bag-time {elapsed:6.3f}s",
        (16, 25),
        scale=0.58,
    )
    if solved_pose_only:
        _draw_text(
            output,
            "GLOVE SOLVED ARTICULATION 20-joint | MOCAP WRIST SE(3) ANCHOR",
            (16, 53),
            color=GLOVE_COLOR_BGR,
        )
        validity = "  ".join(
            f"{side[0].upper()}={'valid' if side_pose_valid[side] else 'unavailable'}"
            for side in ("left", "right")
        )
        _draw_text(
            output,
            f"{validity} | Take_000 canonical R+scale | VIZ-only, not independent glove 6DoF/GT",
            (16, 73),
            scale=0.48,
            color=(225, 225, 225) if any(side_pose_valid.values()) else (80, 160, 255),
        )
    else:
        _draw_text(output, "MOCAP 21-joint", (16, 53), color=MOCAP_COLOR_BGR)
        _draw_text(
            output,
            (
                (
                    "FUSED GLOVE LOCAL | MOCAP wrist SE3 | canonical calibration from "
                    if prepared.pose_mode == POSE_MODE_MOCAP_ROOT
                    else "GLOVE 20-joint | MOCAP wrist position | palm calibration from "
                )
                + f"{prepared.registration_source_segment}"
            ),
            (205, 53),
            color=GLOVE_COLOR_BGR,
        )
        if not valid:
            _draw_text(
                output,
                "outside synchronized CMAvatar interval",
                (16, 73),
                color=(80, 160, 255),
            )
        elif side_stats:
            details = "  ".join(
                f"{side[0].upper()} non-thumb median {stats[0]:.1f}mm / tips {stats[1]:.1f}mm"
                for side, stats in side_stats.items()
            )
            _draw_text(
                output, details, (16, 73), scale=0.48, color=(225, 225, 225)
            )
    return output


def add_visualization_metadata(metrics: dict[str, Any], render_layer: str) -> dict[str, Any]:
    if render_layer not in RENDER_LAYERS:
        raise ValueError(f"Unsupported render layer: {render_layer}")
    solved_pose_only = render_layer == RENDER_LAYER_SOLVED_POSE
    metrics["visualization"] = {
        "render_layer": render_layer,
        "draws_mocap_21_joint_skeleton": not solved_pose_only,
        "draws_glove_solved_20_joint_pose": True,
        "draws_fingertip_error_connectors": not solved_pose_only,
        "solved_pose_semantics": (
            "glove root-local articulation with synchronized MOCAP wrist translation "
            "and orientation; not independent glove wrist 6DoF or ground truth"
        ),
    }
    return metrics


def _scaled_size(video: VideoInfo, output_width: int) -> tuple[int, int]:
    if output_width <= 0 or output_width == video.width:
        return video.width, video.height
    height = int(round(video.height * output_width / video.width))
    if height % 2:
        height += 1
    return int(output_width), height


def _percentiles(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "min": float(np.min(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def _interpolation_time_diagnostics_ms(
    source_times_s: np.ndarray,
    target_times_s: np.ndarray,
    valid: np.ndarray,
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    source = np.asarray(source_times_s, dtype=np.float64)
    target = np.asarray(target_times_s, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if target.shape != mask.shape:
        raise ValueError("Target timestamps and validity mask must have the same shape")
    low, high, _, span_s, bracket_valid = _interpolation_brackets(source, target)
    bracket_ms = span_s * 1000.0
    nearest_ms = (
        np.minimum(
            np.abs(target - source[low]),
            np.abs(source[high] - target),
        )
        * 1000.0
    )
    selected = mask & bracket_valid
    return _percentiles(bracket_ms[selected]), _percentiles(nearest_ms[selected])


def comparison_metrics(
    prepared: PreparedTake,
    *,
    frame_interval: tuple[int, int] | None = None,
) -> dict[str, Any]:
    total_frames = prepared.video.frame_count
    if frame_interval is None:
        first_frame, stop_frame = 0, total_frames
        statistics_scope = "full_source_take"
    else:
        first_frame, stop_frame = (int(value) for value in frame_interval)
        if first_frame < 0 or stop_frame > total_frames or first_frame >= stop_frame:
            raise ValueError(
                f"Invalid metrics frame interval [{first_frame}, {stop_frame})"
            )
        statistics_scope = "selected_source_frame_interval"
    evaluation_mask = np.zeros(total_frames, dtype=bool)
    evaluation_mask[first_frame:stop_frame] = True

    root_fusion = prepared.pose_mode == POSE_MODE_MOCAP_ROOT
    sides: dict[str, Any] = {}
    for side_index, side in enumerate(("left", "right")):
        valid = prepared.glove_valid[side] & evaluation_mask
        glove = prepared.glove_in_world_mm[side][valid]
        mocap = prepared.mocap_mm[valid, side_index]
        glove_relative = glove - glove[:, 0, None, :]
        mocap_relative = mocap - mocap[:, 0, None, :]
        common = np.linalg.norm(
            glove_relative[:, GLOVE_COMMON_INDICES]
            - mocap_relative[:, MOCAP_COMMON_INDICES],
            axis=2,
        )
        mcps = np.linalg.norm(
            glove_relative[:, GLOVE_PALM_INDICES]
            - mocap_relative[:, MOCAP_PALM_INDICES],
            axis=2,
        )
        intermediate = np.linalg.norm(
            glove_relative[:, GLOVE_INTERMEDIATE_INDICES]
            - mocap_relative[:, MOCAP_INTERMEDIATE_INDICES],
            axis=2,
        )
        tips = np.linalg.norm(
            glove_relative[:, GLOVE_TIP_INDICES]
            - mocap_relative[:, MOCAP_TIP_INDICES],
            axis=2,
        )
        registration = prepared.registrations[side]
        synchronized_frames = int(
            np.count_nonzero(prepared.mocap_valid & evaluation_mask)
        )
        retained_frames = int(np.count_nonzero(valid))
        sides[side] = {
            "valid_rgb_frames": retained_frames,
            "synchronized_mocap_rgb_frames": synchronized_frames,
            "frames_rejected_by_glove_interpolation_or_finite_gate": (
                synchronized_frames - retained_frames
            ),
            "retained_fraction_of_synchronized_mocap_frames": (
                retained_frames / synchronized_frames if synchronized_frames else 0.0
            ),
            "registration": {
                "method": (
                    "fixed canonical rotation and uniform scale in mocap wrist-local frame; non-thumb MCP only"
                    if root_fusion
                    else "fixed rotation and uniform scale; non-thumb MCP only"
                ),
                "source_segment": prepared.registration_source_segment,
                "target_segment_used_in_fit": prepared.registration_fitted_on_target,
                "wrist_translation": "copied from synchronized mocap each frame",
                "wrist_orientation": (
                    "copied from synchronized mocap Human root GlobalQ each frame"
                    if root_fusion
                    else "preserved from glove world keypoints; not copied from mocap"
                ),
                "glove_pose_source": (
                    "root-relative local keypoints articulated by synchronized mocap wrist GlobalQ"
                    if root_fusion
                    else "world keypoints rotated by the solved glove wrist quaternion"
                ),
                "per_frame_rotation_refit": False,
                "preserves_glove_vs_mocap_wrist_orientation_difference": not root_fusion,
                "conditional_on_mocap_wrist_se3": root_fusion,
                "target_finger_joints_used_in_fit": False,
                "fingertips_used_in_fit": False,
                "scale": registration.scale,
                **(
                    {
                        MOCAP_ROOT_CANONICAL_ROTATION_FIELD: registration.rotation.tolist()
                    }
                    if root_fusion
                    else {}
                ),
                # Retained for compatibility with existing metrics consumers.
                "rotation_mocap_from_glove": registration.rotation.tolist(),
                "fit_frame_count": registration.fit_frame_count,
                "source_fit_anchor_residual_mm": {
                    "median": registration.anchor_residual_median_mm,
                    "p95": registration.anchor_residual_p95_mm,
                },
            },
            "root_normalized_non_thumb_joint_epe_mm": {
                "aggregation": "pooled_frame_joint_epe",
                **_percentiles(common),
            },
            "root_normalized_frame_mpjpe_mm": {
                "aggregation": "mean_of_16_non_thumb_joint_epe_per_frame",
                **_percentiles(np.mean(common, axis=1)),
            },
            "root_normalized_non_thumb_mcp_epe_mm": {
                "aggregation": "pooled_frame_joint_epe",
                **_percentiles(mcps),
            },
            "root_normalized_non_thumb_intermediate_joint_epe_mm": {
                "aggregation": "pooled_frame_joint_epe",
                **_percentiles(intermediate),
            },
            "root_normalized_non_thumb_fingertip_epe_mm": {
                "aggregation": "pooled_frame_joint_epe",
                **_percentiles(tips),
            },
        }
    actual_rate = (len(prepared.rgb_device_s) - 1) / (
        prepared.rgb_device_s[-1] - prepared.rgb_device_s[0]
    )
    nominal_end_s = (len(prepared.rgb_device_s) - 1) / prepared.video.fps
    bag_end_s = prepared.rgb_device_s[-1] - prepared.rgb_device_s[0]
    mocap_bracket_ms, nearest_mocap_ms = _interpolation_time_diagnostics_ms(
        prepared.mocap_source_times_s,
        prepared.cmavatar_s,
        prepared.mocap_valid & evaluation_mask,
    )
    capture_to_poll_ms = _percentiles(
        prepared.clock_mapping.capture_to_poll_s * 1000.0
    )
    fit_interpretation = (
        "Target segment contributed to palm registration; metrics are same-take consistency."
        if prepared.registration_fitted_on_target
        else (
            f"Registration was fitted on {prepared.registration_source_segment} and frozen; "
            "the target segment did not contribute to rotation or scale fitting."
        )
    )
    return {
        "schema": (
            MOCAP_ROOT_METRICS_SCHEMA
            if root_fusion
            else "gt_calib.glove_mocap_rgb_preview.v2"
        ),
        **(
            {"artifact_type": EVALUATION_METRICS_ARTIFACT_TYPE}
            if root_fusion
            else {}
        ),
        "segment": prepared.segment_dir.name,
        "evaluation_scope": {
            "statistics_scope": statistics_scope,
            "source_frame_first": first_frame,
            "source_frame_stop_exclusive": stop_frame,
            "source_frame_last_inclusive": stop_frame - 1,
            "evaluated_rgb_frames": stop_frame - first_frame,
            "source_take_total_rgb_frames": total_frames,
            "covers_full_source_take": (
                first_frame == 0 and stop_frame == total_frames
            ),
        },
        "evaluation_protocol": {
            "pose_mode": prepared.pose_mode,
            "registration_source_segment": prepared.registration_source_segment,
            "target_segment_used_in_registration_fit": prepared.registration_fitted_on_target,
            "is_cross_take_holdout": not prepared.registration_fitted_on_target,
            "metric_frame": (
                "mocap-wrist SE(3)-conditioned root-normalized articulation"
                if root_fusion
                else "mocap-wrist root-normalized"
            ),
            "conditional_on_mocap_wrist_translation": True,
            "conditional_on_mocap_wrist_orientation": root_fusion,
            "target_finger_joints_used_in_fit": False,
            "quaternion_semantics": (
                MOCAP_ROOT_QUATERNION_SEMANTICS if root_fusion else None
            ),
            "cross_take_scope": (
                (
                    "glove local articulation/session repeatability conditioned on each target's mocap wrist SE(3), including each take's captured session-neutral"
                    if root_fusion
                    else "whole-pipeline/session-repeatability, including each take's captured session-neutral"
                )
                if not prepared.registration_fitted_on_target
                else "same-take registration consistency"
            ),
            "absolute_wrist_translation_evaluable": False,
            "absolute_6dof_metrics": None,
            "absolute_6dof_unavailable_reason": (
                (
                    "glove wrist translation and orientation are not evaluated because synchronized mocap wrist SE(3) is copied into the fused articulation"
                    if root_fusion
                    else "glove wrist translation is zero and synchronized mocap wrist translation is copied into the overlay"
                )
            ),
        },
        "inputs": {
            "rgb_mp4": str(prepared.video.path),
            "rgb_bag": str(
                prepared.segment_dir / "原始BAG与内参" / "camera_1_rgb_depth.bag"
            ),
            "calibration": str(prepared.calibration.path),
            "glove_solver": _glove_solver_provenance(prepared.segment_dir),
        },
        "video": {
            "width": prepared.video.width,
            "height": prepared.video.height,
            "nominal_fps": prepared.video.fps,
            "frame_count": prepared.video.frame_count,
            "bag_index_color_message_count": int(len(prepared.rgb_device_s)),
            "actual_mean_rate_hz": actual_rate,
            "bag_duration_minus_frame_index_over_nominal_fps_ms": (
                bag_end_s - nominal_end_s
            )
            * 1000.0,
        },
        "synchronization": {
            "common_interval_pass": bool(prepared.sync_report["pass_within_common_interval"]),
            "timestamp_semantics": (
                "color_device_timestamp_us is the frame acquisition time; "
                "Thor poll/receive latency is subtracted row by row before mapping "
                "to glove monotonic and CMAvatar Windows time"
            ),
            "camera_capture_to_thor_poll_latency_ms": capture_to_poll_ms,
            "nearest_mocap_sample_delta_ms": nearest_mocap_ms,
            # Deprecated compatibility aliases for existing report/site
            # consumers.  They are nearest-sample quantization distances, not
            # synchronization error or GT accuracy.
            "exposure_to_nearest_mocap_sample_absolute_error_ms": nearest_mocap_ms,
            "camera_to_mocap_absolute_error_ms": nearest_mocap_ms,
            "deprecated_metric_aliases": {
                "exposure_to_nearest_mocap_sample_absolute_error_ms": {
                    "replacement": "nearest_mocap_sample_delta_ms",
                    "semantics": (
                        "Deprecated compatibility alias: nearest 120 Hz MOCAP "
                        "sample quantization distance, not synchronization error "
                        "or GT accuracy."
                    ),
                },
                "camera_to_mocap_absolute_error_ms": {
                    "replacement": "nearest_mocap_sample_delta_ms",
                    "semantics": (
                        "Deprecated compatibility alias: nearest 120 Hz MOCAP "
                        "sample quantization distance at corrected acquisition "
                        "time, not synchronization error or GT accuracy."
                    ),
                },
            },
            "mocap_bracketing_source_span_ms": mocap_bracket_ms,
            "synchronized_rgb_frames": int(
                np.count_nonzero(prepared.mocap_valid & evaluation_mask)
            ),
            "max_interpolation_gap_ms": prepared.max_interpolation_gap_ms,
            "large_interpolation_gaps_are_rejected": True,
            "gap_definition": "time span between the two bracketing source samples",
            "interpolation_target": "corrected RGB frame acquisition timestamp",
            "mocap_sample_timebase": prepared.mocap_timestamp_fit,
        },
        "mocap_joint_quality": {
            "per_frame_confidence_available_in_human_cma": False,
            "marker_visibility_available_in_human_cma": False,
            "occlusion_or_gap_fill_flags_available_in_human_cma": False,
            "reconstruction_residual_available_in_human_cma": False,
        },
        "spatial_calibration": {
            "schema": prepared.calibration.payload.get("schema"),
            "status": prepared.calibration.payload.get("status"),
            "method": prepared.calibration.payload.get("method"),
            "fixed_camera_assumption": prepared.calibration.payload.get(
                "fixed_camera_assumption"
            ),
            "world_to_color_camera": prepared.calibration.world_to_color.tolist(),
            "review_notes": prepared.calibration.payload.get("review_notes", []),
        },
        "sides": sides,
        "interpretation_limits": [
            (
                "This is MOCAP-wrist-6DoF-conditioned glove articulation: both wrist translation and orientation are shared from MOCAP."
                if root_fusion
                else "Glove has no global wrist translation; mocap wrist translation is shared."
            ),
            fit_interpretation,
            (
                "One fixed canonical rotation/scale is used; target finger joints never enter the fit, but target MOCAP wrist GlobalQ conditions every frame."
                if root_fusion
                else "One fixed take-level rotation/scale is used; per-frame rotation is not refitted, so wrist-orientation differences remain visible."
            ),
            (
                "The canonical scale maps two solver templates and is not a verified anatomical bone-length ratio."
                if root_fusion
                else "The fitted scale is a registration gain under orientation mismatch, not a pure anatomical bone-length ratio."
            ),
            "Thumb is drawn but excluded from metrics because glove has 20 points and mocap has 21.",
            "CS-400 calibration is manual visual marker classification plus metric multi-frame depth.",
            "The recorded depth-to-color matrix is used as delivered even though its 3x3 block is slightly non-orthogonal; an official SDK D2C extrinsic or held-out 2D labels are still required for an independent pixel-accuracy claim.",
        ],
    }


def render_take(
    prepared: PreparedTake,
    output_path: Path,
    *,
    output_width: int,
    start_frame: int,
    max_frames: int | None,
    snapshot_count: int,
    render_layer: str = RENDER_LAYER_COMPARISON,
) -> tuple[Path, Path, Path | None]:
    if render_layer not in RENDER_LAYERS:
        raise ValueError(f"Unsupported render layer: {render_layer}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = _scaled_size(prepared.video, output_width)
    first = max(int(start_frame), 0)
    stop = prepared.video.frame_count
    if max_frames is not None:
        stop = min(stop, first + max(int(max_frames), 0))
    if first >= stop:
        raise ValueError(f"Empty render frame interval [{first}, {stop})")

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
        candidates = np.arange(first, stop, dtype=np.int64)
        if render_layer == RENDER_LAYER_SOLVED_POSE:
            pose_valid = np.logical_or.reduce(
                tuple(prepared.glove_valid[side][first:stop] for side in ("left", "right"))
            )
            valid_candidates = candidates[pose_valid]
            if len(valid_candidates):
                candidates = valid_candidates
        sample_count = min(snapshot_count, len(candidates))
        snapshot_indices = set(
            candidates[
                np.rint(np.linspace(0, len(candidates) - 1, sample_count)).astype(int)
            ]
        )
    snapshots: list[np.ndarray] = []
    frame_index = first
    try:
        while frame_index < stop:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"RGB decoding stopped at frame {frame_index}")
            rendered = render_frame(
                prepared, frame_index, frame, render_layer=render_layer
            )
            if (width, height) != (prepared.video.width, prepared.video.height):
                rendered = cv2.resize(rendered, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(rendered)
            if frame_index in snapshot_indices:
                snapshots.append(rendered.copy())
            frame_index += 1
            if frame_index == first + 1 or (frame_index - first) % 300 == 0:
                print(
                    f"{prepared.segment_dir.name}: rendered {frame_index - first}/{stop - first}",
                    flush=True,
                )
    finally:
        capture.release()
        writer.release()

    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            add_visualization_metadata(
                comparison_metrics(prepared, frame_interval=(first, stop)),
                render_layer,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    contact_path: Path | None = None
    if snapshots:
        columns = min(3, len(snapshots))
        rows = int(math.ceil(len(snapshots) / columns))
        cell_height, cell_width = snapshots[0].shape[:2]
        contact = np.zeros((rows * cell_height, columns * cell_width, 3), dtype=np.uint8)
        for index, snapshot in enumerate(snapshots):
            row, column = divmod(index, columns)
            contact[
                row * cell_height : (row + 1) * cell_height,
                column * cell_width : (column + 1) * cell_width,
            ] = snapshot
        contact_path = output.with_suffix(".contact.jpg")
        if not cv2.imwrite(str(contact_path), contact):
            raise RuntimeError(f"Could not write contact sheet: {contact_path}")
    return output, metrics_path, contact_path


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    default_dataset = project_root / "同步整理_20260829_三段"
    parser = argparse.ArgumentParser(
        description="Overlay synchronized glove and CMAvatar hand skeletons on RGB video."
    )
    parser.add_argument("--dataset-root", type=Path, default=default_dataset)
    parser.add_argument(
        "--segment",
        action="append",
        default=[],
        help="01, 02, 03, Take_000, or an exact segment directory name; repeatable.",
    )
    parser.add_argument("--calibration", type=Path)
    parser.add_argument(
        "--pose-mode",
        choices=POSE_MODES,
        help=(
            "glove-wrist-preserved keeps the glove world wrist orientation; "
            "mocap-root-fusion uses MOCAP wrist SE(3) with glove local articulation. "
            "Default: profile value, otherwise glove-wrist-preserved."
        ),
    )
    parser.add_argument(
        "--render-layer",
        choices=RENDER_LAYERS,
        default=RENDER_LAYER_COMPARISON,
        help=(
            "comparison draws MOCAP, solved glove pose, and fingertip connectors; "
            "solved-pose-only draws only the solved 20-joint glove articulation "
            "while retaining the explicit MOCAP wrist SE(3) anchor label."
        ),
    )
    registration_group = parser.add_mutually_exclusive_group()
    registration_group.add_argument(
        "--registration-segment",
        help=(
            "Fit fixed palm rotation/scale on this segment and freeze it for all "
            "requested segments, e.g. 01."
        ),
    )
    registration_group.add_argument(
        "--registration-profile",
        type=Path,
        help="Load a previously exported fixed palm registration JSON profile.",
    )
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs")
    parser.add_argument("--output-width", type=int, default=1280)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--snapshot-count", type=int, default=6)
    parser.add_argument(
        "--max-interpolation-gap-ms",
        type=float,
        help=(
            "Reject interpolation across a larger source timestamp gap. "
            "Default: profile value or 25 ms."
        ),
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Write metrics JSON without rendering video.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    calibration = (
        args.calibration.expanduser().resolve()
        if args.calibration is not None
        else _default_calibration(dataset_root)
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_prepared: PreparedTake | None = None
    fixed_registrations: Mapping[str, SimilarityRegistration] | None = None
    registration_source_segment: str | None = None
    profile_payload: Mapping[str, Any] | None = None
    if args.registration_profile is not None:
        registration_source_segment, fixed_registrations, profile_payload = (
            load_registration_profile(args.registration_profile.expanduser().resolve())
        )
    profile_pose_mode = (
        str(profile_payload.get("pose_mode", POSE_MODE_GLOVE_WRIST))
        if profile_payload is not None
        else None
    )
    if (
        args.pose_mode is not None
        and profile_pose_mode is not None
        and args.pose_mode != profile_pose_mode
    ):
        raise ValueError(
            f"Requested pose mode {args.pose_mode} does not match profile mode "
            f"{profile_pose_mode}"
        )
    pose_mode = args.pose_mode or profile_pose_mode or POSE_MODE_GLOVE_WRIST
    max_interpolation_gap_ms = (
        args.max_interpolation_gap_ms
        if args.max_interpolation_gap_ms is not None
        else float(
            profile_payload.get("max_interpolation_gap_ms", 25.0)
            if profile_payload is not None
            else 25.0
        )
    )
    if args.registration_segment is not None:
        source_segment = _discover_segment(dataset_root, args.registration_segment)
        print(f"Fitting registration on {source_segment.name}", flush=True)
        source_prepared = prepare_take(
            source_segment,
            calibration,
            max_interpolation_gap_ms=max_interpolation_gap_ms,
            pose_mode=pose_mode,
        )
        registration_source_segment = source_segment.name
        fixed_registrations = source_prepared.registrations
        profile_name = (
            f"{source_segment.name}_mocap_root_fusion_registration_profile.json"
            if pose_mode == POSE_MODE_MOCAP_ROOT
            else f"{source_segment.name}_registration_profile.json"
        )
        profile_path = write_registration_profile(
            output_dir / profile_name,
            source_prepared,
        )
        print(f"Wrote {profile_path}", flush=True)

    segment_keys = args.segment or ["01", "02", "03"]
    for key in segment_keys:
        segment = _discover_segment(dataset_root, key)
        print(f"Preparing {segment.name}", flush=True)
        if source_prepared is not None and segment == source_prepared.segment_dir:
            prepared = source_prepared
        else:
            prepared = prepare_take(
                segment,
                calibration,
                registrations=fixed_registrations,
                registration_source_segment=registration_source_segment,
                max_interpolation_gap_ms=max_interpolation_gap_ms,
                pose_mode=pose_mode,
            )
        if fixed_registrations is None:
            protocol_suffix = "same_take_fit"
        elif prepared.registration_fitted_on_target:
            protocol_suffix = "calibration_fit"
        else:
            source_take = prepared.registration_source_segment.split("Take_", 1)[-1]
            protocol_suffix = f"holdout_from_Take_{source_take}"
        if args.render_layer == RENDER_LAYER_SOLVED_POSE:
            comparison_name = "solved_hand_pose"
        else:
            comparison_name = (
                "mocap_root_fused_glove"
                if pose_mode == POSE_MODE_MOCAP_ROOT
                else "glove_vs_mocap"
            )
        stem = f"{segment.name}_{comparison_name}_{protocol_suffix}"
        metrics_path = output_dir / f"{stem}.metrics.json"
        if args.analyze_only:
            metrics_path.write_text(
                json.dumps(
                    add_visualization_metadata(
                        comparison_metrics(prepared), args.render_layer
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"Wrote {metrics_path}", flush=True)
            continue
        outputs = render_take(
            prepared,
            output_dir / f"{stem}.mp4",
            output_width=args.output_width,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            snapshot_count=args.snapshot_count,
            render_layer=args.render_layer,
        )
        print("Wrote " + ", ".join(str(path) for path in outputs if path), flush=True)


if __name__ == "__main__":
    main()
