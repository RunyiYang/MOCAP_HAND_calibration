#!/usr/bin/env python3
"""Render synchronized CMAvatar hands directly on raw metric depth frames.

This is deliberately independent of the RGB overlay path:

* ``mono16`` depth is decoded from the ROS1 BAG without using ``Depth.mp4``.
* Every depth frame uses its own Orbbec device timestamp.  The approximately
  1.31 ms depth-after-color offset is therefore preserved, not rounded away.
* CMAvatar positions are interpolated at that depth acquisition timestamp.
* The delivered, rigid ``world_to_depth_camera`` SE(3) is used exactly as
  recorded.  No hand pixels or depth surfaces are used to refit it here.
* The preview uses one fixed millimetre colour scale across all frames.

The result is a synchronization, projection, and coverage diagnostic.  A
mocap joint is an anatomical/marker-model location while a depth pixel is the
first visible surface on a camera ray, so their difference is not reported as
an accuracy score.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import cv2
import numpy as np

import gt_calib_viz as viz
import mocap_video_overlay as rgb_overlay


DEPTH_TOPIC = "/cam/sensor_3/frameType_3"
DEPTH_PROFILE_TOPIC = "/cam/streamProfileType_3"
PROFILE_MESSAGE_TYPE = "custom_msg/OBStreamProfileInfo"
DEPTH_OVERLAY_SCHEMA = "gt_calib.depth_mocap_overlay.v1"
LEFT_COLOR_BGR = (225, 65, 245)
RIGHT_COLOR_BGR = (245, 220, 45)
OUTLINE_BGR = (10, 10, 10)


@dataclass(frozen=True)
class DepthCalibration:
    path: Path
    payload: Mapping[str, Any]
    world_to_depth: np.ndarray
    intrinsics: np.ndarray
    distortion: np.ndarray
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class DepthFrame:
    message_index: int
    message_timestamp_ns: int
    device_timestamp_us: int
    frame_number: int
    encoding: str
    width: int
    height: int
    depth_mm: np.ndarray


@dataclass(frozen=True)
class DepthFrameMetadata:
    """Lightweight provenance retained after a raw depth frame is rendered."""

    message_index: int
    message_timestamp_ns: int
    device_timestamp_us: int
    frame_number: int


@dataclass(frozen=True)
class OrbbecVideoProfile:
    message_timestamp_ns: int
    stream_type: int
    pixel_format: int
    rotation_matrix: tuple[float, ...]
    translation_mm: tuple[float, ...]
    width: int
    height: int
    fps: int
    intrinsics: tuple[float, ...]
    distortion: tuple[float, ...]
    distortion_model: int


@dataclass(frozen=True)
class PreparedDepthMocap:
    segment_dir: Path
    bag_path: Path
    depth_refs: Sequence[rgb_overlay.Ros1MessageRef]
    depth_timestamp_ns: np.ndarray
    rgb_timestamp_ns: np.ndarray
    depth_device_s: np.ndarray
    rgb_device_s: np.ndarray
    cmavatar_s: np.ndarray
    clock_inside: np.ndarray
    clock_mapping: viz.ClockMapping
    mocap_path: Path
    mocap: viz.MocapSequence
    mocap_mm: np.ndarray
    mocap_valid: np.ndarray
    interpolation: rgb_overlay.InterpolationAudit
    calibration: DepthCalibration
    camera_contract: Mapping[str, Any]
    depth_rgb_pairing: Mapping[str, Any]
    sync_report: Mapping[str, Any]
    max_interpolation_gap_ms: float


@dataclass(frozen=True)
class FrameCoverage:
    raw_valid_pixel_fraction: float
    joint_count: int
    positive_depth_joint_count: int
    in_frame_joint_count: int
    raw_depth_valid_at_joint_pixel_count: int
    wrist_count: int
    in_frame_wrist_count: int
    raw_depth_valid_at_wrist_pixel_count: int


def _read_ros_string(payload: bytes, offset: int) -> tuple[str, int]:
    if offset + 4 > len(payload):
        raise ValueError("Truncated ROS string length")
    length = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    stop = offset + length
    if stop > len(payload):
        raise ValueError("Truncated ROS string")
    return payload[offset:stop].decode("utf-8", errors="replace"), stop


def decode_orbbec_depth_image(
    serialized: bytes,
    *,
    message_index: int,
    message_timestamp_ns: int,
) -> DepthFrame:
    """Decode the recorder's actual Orbbec ``sensor_msgs/Image`` wire order."""

    if len(serialized) < 73:
        raise ValueError("Truncated sensor_msgs/Image payload")
    _, stamp_seconds, stamp_nanoseconds = struct.unpack_from("<III", serialized, 0)
    header_timestamp_ns = int(stamp_seconds) * 1_000_000_000 + int(stamp_nanoseconds)
    if header_timestamp_ns != message_timestamp_ns:
        raise ValueError("Orbbec depth header time disagrees with BAG message time")
    _, offset = _read_ros_string(serialized, 12)
    if offset + 8 > len(serialized):
        raise ValueError("Truncated sensor_msgs/Image dimensions")
    height, width = struct.unpack_from("<II", serialized, offset)
    encoding, offset = _read_ros_string(serialized, offset + 8)
    if offset + 13 > len(serialized):
        raise ValueError("Truncated sensor_msgs/Image payload header")
    is_bigendian = bool(serialized[offset])
    offset += 1
    step = struct.unpack_from("<I", serialized, offset)[0]
    offset += 4
    metadata_size, packed_data_size = struct.unpack_from("<II", serialized, offset)
    offset += 8
    packed_stop = offset + packed_data_size
    if packed_stop + 32 != len(serialized):
        raise ValueError(
            "Orbbec depth packed-data length does not leave four uint64 metadata fields"
        )
    if metadata_size > packed_data_size:
        raise ValueError("Orbbec depth metadata exceeds packed-data size")
    image_data = serialized[offset + metadata_size : packed_stop]
    frame_number, device_timestamp_us, _, _ = struct.unpack_from(
        "<QQQQ", serialized, packed_stop
    )
    if (message_timestamp_ns + 500) // 1000 != int(device_timestamp_us):
        raise ValueError(
            "Orbbec depth device timestamp does not match BAG time at microsecond precision"
        )
    if encoding.lower() not in {"mono16", "16uc1"}:
        raise ValueError(f"Expected mono16 depth, found {encoding!r}")
    if step < width * 2 or step % 2:
        raise ValueError(f"Invalid mono16 row step {step} for width {width}")
    if len(image_data) != height * step:
        raise ValueError(
            f"Depth byte count {len(image_data)} does not match height*step {height * step}"
        )
    dtype = np.dtype(">u2" if is_bigendian else "<u2")
    row_values = step // 2
    depth_mm = (
        np.frombuffer(image_data, dtype=dtype)
        .reshape(int(height), int(row_values))[:, : int(width)]
        .astype(np.uint16, copy=True)
    )
    return DepthFrame(
        message_index=int(message_index),
        message_timestamp_ns=int(message_timestamp_ns),
        device_timestamp_us=int(device_timestamp_us),
        frame_number=int(frame_number),
        encoding=encoding,
        width=int(width),
        height=int(height),
        depth_mm=depth_mm,
    )


def _read_indexed_ros1_message(
    bag_path: Path,
    ref: rgb_overlay.Ros1MessageRef,
) -> bytes:
    """Read one indexed message and revalidate its connection and timestamp."""

    with Path(bag_path).open("rb") as handle:
        handle.seek(ref.chunk_data_offset)
        compressed = handle.read(ref.chunk_data_length)
    if len(compressed) != ref.chunk_data_length:
        raise ValueError("Truncated ROS1 compressed chunk")
    chunk = rgb_overlay._decompress_ros1_chunk(
        compressed,
        ref.compression,
        ref.uncompressed_size,
    )
    nested = BytesIO(chunk)
    nested.seek(ref.record_offset)
    header, serialized = viz._read_ros_record(nested, read_data=True)
    if viz._field_int(header, "op") != rgb_overlay.ROS_OP_MESSAGE_DATA:
        raise ValueError("ROS1 index offset does not point to message data")
    if viz._field_int(header, "conn") != ref.connection_id:
        raise ValueError("ROS1 index offset points to another connection")
    if rgb_overlay._ros_time_ns(header["time"]) != ref.timestamp_ns:
        raise ValueError("ROS1 message timestamp disagrees with INDEX_DATA")
    return serialized


def decode_orbbec_video_profile(
    serialized: bytes,
    *,
    message_timestamp_ns: int,
) -> OrbbecVideoProfile:
    """Decode the recorder's ``custom_msg/OBStreamProfileInfo`` wire layout."""

    if len(serialized) < 121:
        raise ValueError("Truncated Orbbec stream profile")
    _, stamp_seconds, stamp_nanoseconds = struct.unpack_from("<III", serialized, 0)
    header_timestamp_ns = int(stamp_seconds) * 1_000_000_000 + int(stamp_nanoseconds)
    if header_timestamp_ns != message_timestamp_ns:
        raise ValueError("Orbbec profile header time disagrees with BAG message time")
    _, offset = _read_ros_string(serialized, 12)
    stream_type, pixel_format = struct.unpack_from("<BB", serialized, offset)
    offset += 2
    rotation = struct.unpack_from("<9f", serialized, offset)
    offset += 9 * 4
    translation = struct.unpack_from("<3f", serialized, offset)
    offset += 3 * 4
    width, height, fps = struct.unpack_from("<3H", serialized, offset)
    offset += 3 * 2
    intrinsics = struct.unpack_from("<4f", serialized, offset)
    offset += 4 * 4
    distortion = struct.unpack_from("<8f", serialized, offset)
    offset += 8 * 4
    (distortion_model,) = struct.unpack_from("<B", serialized, offset)
    offset += 1
    if offset != len(serialized):
        raise ValueError(
            f"Unexpected Orbbec profile trailing bytes: parsed {offset}, payload {len(serialized)}"
        )
    return OrbbecVideoProfile(
        message_timestamp_ns=int(message_timestamp_ns),
        stream_type=int(stream_type),
        pixel_format=int(pixel_format),
        rotation_matrix=tuple(float(value) for value in rotation),
        translation_mm=tuple(float(value) for value in translation),
        width=int(width),
        height=int(height),
        fps=int(fps),
        intrinsics=tuple(float(value) for value in intrinsics),
        distortion=tuple(float(value) for value in distortion),
        distortion_model=int(distortion_model),
    )


def load_orbbec_depth_profile(bag_path: Path) -> tuple[OrbbecVideoProfile, int]:
    refs, message_type = rgb_overlay._ros1_topic_message_refs(
        bag_path,
        DEPTH_PROFILE_TOPIC,
        expected_message_type=PROFILE_MESSAGE_TYPE,
    )
    if not refs:
        raise ValueError("Raw RGB-D BAG contains no depth stream profile")
    profiles = [
        decode_orbbec_video_profile(
            _read_indexed_ros1_message(bag_path, ref),
            message_timestamp_ns=ref.timestamp_ns,
        )
        for ref in refs
    ]
    first = profiles[0]
    if any(profile != first for profile in profiles[1:]):
        raise ValueError("Depth stream profile changes within one recording")
    return first, len(profiles)


def iter_bag_depth_frames(
    bag_path: Path,
    refs: Sequence[rgb_overlay.Ros1MessageRef],
    start: int,
    stop: int,
) -> Iterator[DepthFrame]:
    """Yield a half-open, ordered depth interval while caching one ROS chunk."""

    if not (0 <= start <= stop <= len(refs)):
        raise ValueError(f"Invalid depth frame interval [{start}, {stop})")
    cached_chunk_key: tuple[int, int] | None = None
    cached_chunk: bytes | None = None
    with Path(bag_path).open("rb") as handle:
        for message_index in range(start, stop):
            ref = refs[message_index]
            chunk_key = (ref.chunk_data_offset, ref.chunk_data_length)
            if cached_chunk_key != chunk_key or cached_chunk is None:
                handle.seek(ref.chunk_data_offset)
                compressed = handle.read(ref.chunk_data_length)
                if len(compressed) != ref.chunk_data_length:
                    raise ValueError("Truncated ROS1 compressed chunk")
                cached_chunk = rgb_overlay._decompress_ros1_chunk(
                    compressed,
                    ref.compression,
                    ref.uncompressed_size,
                )
                cached_chunk_key = chunk_key
            nested = BytesIO(cached_chunk)
            nested.seek(ref.record_offset)
            header, serialized = viz._read_ros_record(nested, read_data=True)
            if viz._field_int(header, "op") != rgb_overlay.ROS_OP_MESSAGE_DATA:
                raise ValueError("ROS1 depth index offset does not point to message data")
            if viz._field_int(header, "conn") != ref.connection_id:
                raise ValueError("ROS1 depth index offset points to another connection")
            indexed_timestamp_ns = rgb_overlay._ros_time_ns(header["time"])
            if indexed_timestamp_ns != ref.timestamp_ns:
                raise ValueError("ROS1 depth timestamp disagrees with INDEX_DATA")
            yield decode_orbbec_depth_image(
                serialized,
                message_index=message_index,
                message_timestamp_ns=ref.timestamp_ns,
            )


def load_depth_calibration(path: Path) -> DepthCalibration:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "movementcap.camera_to_mocap_world.v1":
        raise ValueError(f"Unsupported camera/world calibration schema in {source}")
    transform = np.asarray(
        payload["transforms"]["world_to_depth_camera"], dtype=np.float64
    )
    inverse = np.asarray(
        payload["transforms"]["depth_camera_to_world"], dtype=np.float64
    )
    depth = payload["orbbec_profiles_from_bag"]["depth"]
    intrinsics = np.asarray(depth["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
    distortion = np.asarray(
        depth["distortion_orbbec_k1_k2_k3_k4_k5_k6_p1_p2"], dtype=np.float64
    )
    width, height = (int(value) for value in depth["resolution"])
    fps = float(depth["fps"])
    if transform.shape != (4, 4) or inverse.shape != (4, 4):
        raise ValueError("Depth/world transforms must be 4x4")
    if intrinsics.shape != (4,) or distortion.shape != (8,):
        raise ValueError("Malformed depth camera model")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12):
        raise ValueError("world_to_depth_camera has a non-homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9):
        raise ValueError("world_to_depth_camera rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9):
        raise ValueError("world_to_depth_camera rotation is not right-handed")
    if not np.allclose(transform @ inverse, np.eye(4), atol=1e-8):
        raise ValueError("Delivered depth/world transforms are not inverses")
    if min(width, height) <= 0 or fps <= 0.0:
        raise ValueError("Invalid depth resolution or frame rate")
    return DepthCalibration(
        path=source,
        payload=payload,
        world_to_depth=transform,
        intrinsics=intrinsics,
        distortion=distortion,
        width=width,
        height=height,
        fps=fps,
    )


def validate_recorded_depth_camera_contract(
    segment_dir: Path,
    bag_path: Path,
    calibration: DepthCalibration,
) -> dict[str, Any]:
    """Fail closed unless this take's delivered and BAG-native camera models agree."""

    segment = Path(segment_dir)
    intrinsics_path = segment / "原始BAG与内参" / "camera_1_intrinsics.json"
    delivered = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    recorded_profile, profile_message_count = load_orbbec_depth_profile(bag_path)
    camera_param = delivered["camera_param"]
    depth_intrinsic = camera_param["depth_intrinsic"]
    depth_distortion = camera_param["depth_distortion"]
    json_intrinsics = np.asarray(
        [
            depth_intrinsic["fx"],
            depth_intrinsic["fy"],
            depth_intrinsic["cx"],
            depth_intrinsic["cy"],
        ],
        dtype=np.float64,
    )
    json_distortion = np.asarray(
        [
            depth_distortion["k1"],
            depth_distortion["k2"],
            depth_distortion["k3"],
            depth_distortion["k4"],
            depth_distortion["k5"],
            depth_distortion["k6"],
            depth_distortion["p1"],
            depth_distortion["p2"],
        ],
        dtype=np.float64,
    )
    consistency = calibration.payload["camera_consistency"]
    expected_serials = list(consistency["serial_numbers"])
    serial = str(delivered["device"]["serial_number"])
    details = [
        row
        for row in consistency["details"]
        if row.get("recording") == segment.name
    ]
    expected_profile = calibration.payload["orbbec_profiles_from_bag"]["depth"]
    gates = {
        "intrinsics_json_schema": (
            delivered.get("schema") == "multi_camera_datacollect.orbbec_camera_intrinsics.v1"
        ),
        "bag_filename_matches": delivered.get("bag_path") == Path(bag_path).name,
        "single_expected_serial": len(expected_serials) == 1,
        "serial_matches_calibration": serial in expected_serials,
        "segment_has_calibration_detail": len(details) == 1,
        "json_resolution_matches": (
            [int(depth_intrinsic["width"]), int(depth_intrinsic["height"])]
            == [calibration.width, calibration.height]
        ),
        "json_intrinsics_match": bool(
            np.allclose(json_intrinsics, calibration.intrinsics, rtol=0.0, atol=1e-6)
        ),
        "json_distortion_matches": bool(
            np.allclose(json_distortion, calibration.distortion, rtol=0.0, atol=1e-6)
        ),
        "bag_profile_resolution_matches": (
            [recorded_profile.width, recorded_profile.height]
            == [calibration.width, calibration.height]
            == list(expected_profile["resolution"])
        ),
        "bag_profile_fps_matches": (
            abs(recorded_profile.fps - calibration.fps) <= 1e-12
            and recorded_profile.fps == int(expected_profile["fps"])
        ),
        "bag_profile_intrinsics_match": bool(
            np.allclose(
                recorded_profile.intrinsics,
                calibration.intrinsics,
                rtol=0.0,
                atol=1e-6,
            )
        ),
        "bag_profile_distortion_matches": bool(
            np.allclose(
                recorded_profile.distortion,
                calibration.distortion,
                rtol=0.0,
                atol=1e-6,
            )
        ),
    }
    if details:
        gates["segment_detail_serial_and_depth_intrinsics_match"] = bool(
            details[0].get("serial_number") == serial
            and np.allclose(
                details[0].get("depth", []),
                calibration.intrinsics,
                rtol=0.0,
                atol=1e-6,
            )
        )
    else:
        gates["segment_detail_serial_and_depth_intrinsics_match"] = False
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError(
            f"Recorded depth-camera contract disagrees with calibration for {segment.name}: {failed}"
        )
    return {
        "pass": True,
        "serial_number": serial,
        "device_name": delivered["device"].get("name"),
        "firmware_version": delivered["device"].get("firmware_version"),
        "intrinsics_json": str(intrinsics_path.resolve()),
        "bag_depth_profile_topic": DEPTH_PROFILE_TOPIC,
        "bag_depth_profile_message_type": PROFILE_MESSAGE_TYPE,
        "bag_depth_profile_message_count": profile_message_count,
        "bag_depth_profile_timestamp_ns": recorded_profile.message_timestamp_ns,
        "bag_depth_profile_stream_type": recorded_profile.stream_type,
        "bag_depth_profile_pixel_format": recorded_profile.pixel_format,
        "bag_depth_profile_distortion_model": recorded_profile.distortion_model,
        "resolution": [recorded_profile.width, recorded_profile.height],
        "fps": recorded_profile.fps,
        "intrinsics_fx_fy_cx_cy": list(recorded_profile.intrinsics),
        "distortion_orbbec_k1_k2_k3_k4_k5_k6_p1_p2": list(
            recorded_profile.distortion
        ),
        "gates": gates,
    }


def validate_depth_rgb_pairing(
    depth_timestamp_ns: np.ndarray,
    rgb_timestamp_ns: np.ndarray,
) -> dict[str, Any]:
    """Prove paired messages are same-index and any extra depth is a tail frame."""

    depth_ns = np.asarray(depth_timestamp_ns, dtype=np.int64)
    rgb_ns = np.asarray(rgb_timestamp_ns, dtype=np.int64)
    if len(depth_ns) not in {len(rgb_ns), len(rgb_ns) + 1}:
        raise ValueError("Unexpected RGB/depth message-count relation")
    if min(len(depth_ns), len(rgb_ns)) < 2:
        raise ValueError("Not enough RGB-D timestamps to validate pairing")
    if np.any(np.diff(depth_ns) <= 0) or np.any(np.diff(rgb_ns) <= 0):
        raise ValueError("RGB-D timestamps must be strictly increasing")

    positions = np.searchsorted(rgb_ns, depth_ns, side="left")
    right = np.clip(positions, 0, len(rgb_ns) - 1)
    left = np.clip(positions - 1, 0, len(rgb_ns) - 1)
    choose_left = np.abs(depth_ns - rgb_ns[left]) <= np.abs(rgb_ns[right] - depth_ns)
    nearest = np.where(choose_left, left, right)
    paired_count = len(rgb_ns)
    same_index_nearest = bool(
        np.array_equal(nearest[:paired_count], np.arange(paired_count))
    )
    extra_count = len(depth_ns) - len(rgb_ns)
    extra_tail_validated = bool(
        extra_count == 0
        or (
            extra_count == 1
            and depth_ns[-1] > rgb_ns[-1]
            and nearest[-1] == len(rgb_ns) - 1
            and depth_ns[-1] > depth_ns[-2]
        )
    )
    paired_delta_ms = (depth_ns[:paired_count] - rgb_ns) / 1e6
    paired_delta_plausible = bool(
        np.all((paired_delta_ms > 0.0) & (paired_delta_ms < 5.0))
    )
    if not (same_index_nearest and extra_tail_validated and paired_delta_plausible):
        raise ValueError(
            "RGB-D same-index/tail pairing failed: "
            f"same_index={same_index_nearest}, extra_tail={extra_tail_validated}, "
            f"delta_plausible={paired_delta_plausible}"
        )
    return {
        "pass": True,
        "same_index_is_nearest_for_all_paired_messages": same_index_nearest,
        "paired_message_count": paired_count,
        "extra_depth_message_count": extra_count,
        "extra_depth_is_validated_tail_frame": extra_tail_validated,
        "extra_depth_tail_timestamp_ns": int(depth_ns[-1]) if extra_count else None,
        "paired_depth_minus_rgb_ms": rgb_overlay._distribution_ms(
            paired_delta_ms * 1e-3
        ),
    }


def project_world_to_depth(
    points_world_mm: np.ndarray,
    calibration: DepthCalibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project mocap-world points through the frozen raw-depth camera model."""

    points = np.asarray(points_world_mm, dtype=np.float64)
    camera = (
        points @ calibration.world_to_depth[:3, :3].T
        + calibration.world_to_depth[:3, 3]
    )
    depth_z_mm = camera[..., 2]
    positive = np.isfinite(camera).all(axis=-1) & (depth_z_mm > 1e-6)
    safe_z = np.where(positive, depth_z_mm, np.nan)
    x = camera[..., 0] / safe_z
    y = camera[..., 1] / safe_z
    xd, yd = viz.distort_normalized(x, y, calibration.distortion)
    fx, fy, cx, cy = calibration.intrinsics
    pixels = np.stack((fx * xd + cx, fy * yd + cy), axis=-1)
    return pixels, positive, depth_z_mm


def metric_depth_preview(
    depth_mm: np.ndarray,
    *,
    min_depth_mm: float,
    max_depth_mm: float,
) -> np.ndarray:
    """Colourize raw depth with a fixed, frame-independent millimetre scale."""

    depth = np.asarray(depth_mm)
    if depth.ndim != 2:
        raise ValueError("Depth preview input must be a 2D image")
    if not (0.0 < min_depth_mm < max_depth_mm):
        raise ValueError("Depth preview bounds must satisfy 0 < min < max")
    valid = depth > 0
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    scaled = (depth.astype(np.float32) - min_depth_mm) / (
        max_depth_mm - min_depth_mm
    )
    normalized[valid] = np.clip(scaled[valid] * 255.0, 0.0, 255.0).astype(
        np.uint8
    )
    # Turbo's red end denotes nearer geometry after inversion; zero depth is
    # explicitly black instead of sharing either endpoint colour.
    preview = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    return preview


def prepare_depth_mocap(
    segment_dir: Path,
    calibration_path: Path,
    *,
    max_interpolation_gap_ms: float = 25.0,
) -> PreparedDepthMocap:
    segment = Path(segment_dir)
    if max_interpolation_gap_ms <= 0.0:
        raise ValueError("max_interpolation_gap_ms must be positive")
    bag_path = segment / "原始BAG与内参" / "camera_1_rgb_depth.bag"
    depth_refs, message_type = rgb_overlay._ros1_topic_message_refs(
        bag_path, DEPTH_TOPIC
    )
    if message_type != "sensor_msgs/Image":
        raise ValueError(f"Unexpected depth message type {message_type!r}")
    depth_ns = np.asarray([ref.timestamp_ns for ref in depth_refs], dtype=np.int64)
    if len(depth_ns) < 2 or np.any(np.diff(depth_ns) <= 0):
        raise ValueError("Depth BAG timestamps are missing or non-monotonic")
    rgb_refs, rgb_message_type = rgb_overlay._ros1_topic_message_refs(
        bag_path, viz.COLOR_TOPIC
    )
    if rgb_message_type != "sensor_msgs/Image":
        raise ValueError(f"Unexpected RGB message type {rgb_message_type!r}")
    rgb_ns = np.asarray([ref.timestamp_ns for ref in rgb_refs], dtype=np.int64)
    depth_device_s = depth_ns.astype(np.float64) * 1e-9
    rgb_device_s = rgb_ns.astype(np.float64) * 1e-9
    depth_rgb_pairing = validate_depth_rgb_pairing(depth_ns, rgb_ns)

    clock_mapping = viz.load_clock_mapping(
        segment / "同步校验" / "camera_cmavatar_alignment.csv"
    )
    _, cmavatar_s, clock_inside = clock_mapping.map(depth_device_s)
    take_number = viz._take_number(segment)
    mocap_path = (
        segment / "动捕" / f"Take_{take_number}" / f"Take_{take_number}_Human.cma"
    )
    mocap = viz.load_mocap_human(mocap_path)
    max_gap_s = max_interpolation_gap_ms * 1e-3
    interpolation = rgb_overlay.interpolation_audit(
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
        raise AssertionError("Depth interpolation audit disagrees with position interpolation")

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
    rgb_overlay.synchronized_interval(mocap_valid)

    calibration = load_depth_calibration(calibration_path)
    applies_to = set(calibration.payload.get("applies_to", []))
    if segment.name not in applies_to:
        raise ValueError(
            f"Calibration {calibration.path} does not declare applicability to {segment.name}"
        )
    if not bool(calibration.payload.get("camera_consistency", {}).get("passed")):
        raise ValueError("Delivered camera-consistency check did not pass")
    camera_contract = validate_recorded_depth_camera_contract(
        segment,
        bag_path,
        calibration,
    )
    return PreparedDepthMocap(
        segment_dir=segment,
        bag_path=bag_path,
        depth_refs=depth_refs,
        depth_timestamp_ns=depth_ns,
        rgb_timestamp_ns=rgb_ns,
        depth_device_s=depth_device_s,
        rgb_device_s=rgb_device_s,
        cmavatar_s=cmavatar_s,
        clock_inside=clock_inside,
        clock_mapping=clock_mapping,
        mocap_path=mocap_path,
        mocap=mocap,
        mocap_mm=mocap_mm,
        mocap_valid=mocap_valid,
        interpolation=interpolation,
        calibration=calibration,
        camera_contract=camera_contract,
        depth_rgb_pairing=depth_rgb_pairing,
        sync_report=sync_report,
        max_interpolation_gap_ms=max_interpolation_gap_ms,
    )


def _inside_frame(pixels: np.ndarray, width: int, height: int) -> np.ndarray:
    return (
        np.isfinite(pixels).all(axis=-1)
        & (pixels[..., 0] >= 0.0)
        & (pixels[..., 0] <= width - 1)
        & (pixels[..., 1] >= 0.0)
        & (pixels[..., 1] <= height - 1)
    )


def render_depth_mocap_frame(
    prepared: PreparedDepthMocap,
    source_frame: int,
    output_frame: int,
    depth_mm: np.ndarray,
    *,
    first_source_frame: int,
    min_depth_mm: float,
    max_depth_mm: float,
) -> tuple[np.ndarray, FrameCoverage]:
    if not prepared.mocap_valid[source_frame]:
        raise ValueError(f"Depth source frame {source_frame} is not synchronized")
    calibration = prepared.calibration
    if depth_mm.shape != (calibration.height, calibration.width):
        raise ValueError(
            f"Depth shape {depth_mm.shape} does not match calibrated "
            f"{(calibration.height, calibration.width)}"
        )
    output = metric_depth_preview(
        depth_mm,
        min_depth_mm=min_depth_mm,
        max_depth_mm=max_depth_mm,
    )
    pixels, positive, _ = project_world_to_depth(
        prepared.mocap_mm[source_frame], calibration
    )
    pixels[~positive] = np.nan
    in_frame = positive & _inside_frame(pixels, calibration.width, calibration.height)
    sampled_depth_valid = np.zeros(in_frame.shape, dtype=bool)
    if np.any(in_frame):
        rounded = np.rint(pixels[in_frame]).astype(np.int64)
        sampled_depth_valid[in_frame] = depth_mm[rounded[:, 1], rounded[:, 0]] > 0

    panel = output.copy()
    cv2.rectangle(panel, (0, 0), (output.shape[1], 91), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.66, output, 0.34, 0.0, output)
    for side_index, (label, color) in enumerate(
        (("L", LEFT_COLOR_BGR), ("R", RIGHT_COLOR_BGR))
    ):
        side_pixels = pixels[side_index]
        viz.draw_hand_skeleton(
            output,
            side_pixels,
            viz.MOCAP_CHAINS,
            OUTLINE_BGR,
            thickness=7,
            radius=7,
            hollow=False,
        )
        viz.draw_hand_skeleton(
            output,
            side_pixels,
            viz.MOCAP_CHAINS,
            color,
            thickness=3,
            radius=4,
            hollow=False,
        )
        wrist = side_pixels[0]
        if viz._finite_pixel(wrist):
            x, y = np.rint(wrist).astype(int)
            viz._draw_text(
                output,
                f"{label} MOCAP",
                (int(x + 7), int(y - 8)),
                scale=0.42,
                color=color,
            )

    audit = prepared.interpolation
    low = int(audit.low_indices[source_frame])
    high = int(audit.high_indices[source_frame])
    elapsed_s = (
        prepared.depth_device_s[source_frame]
        - prepared.depth_device_s[first_source_frame]
    )
    viz._draw_text(
        output,
        "RAW METRIC DEPTH <-> MOCAP | frozen world-to-depth SE(3)",
        (12, 22),
        scale=0.48,
    )
    viz._draw_text(
        output,
        (
            f"{prepared.segment_dir.name} | out {output_frame} | depth {source_frame} "
            f"| t={elapsed_s:7.3f}s | fixed scale {min_depth_mm:.0f}-{max_depth_mm:.0f} mm"
        ),
        (12, 48),
        scale=0.41,
    )
    viz._draw_text(
        output,
        (
            f"CMAvatar {prepared.mocap.frame_counters[low]} -> "
            f"{prepared.mocap.frame_counters[high]} | alpha={audit.alpha[source_frame]:.3f} "
            f"| nearest dt={audit.nearest_delta_s[source_frame] * 1000.0:.2f}ms"
        ),
        (12, 73),
        scale=0.41,
        color=(175, 245, 175),
    )
    viz._draw_text(
        output,
        "black = raw depth invalid | skeleton/surface difference is not an accuracy score",
        (12, output.shape[0] - 12),
        scale=0.36,
        color=(215, 215, 215),
    )

    wrist_mask = np.zeros(in_frame.shape, dtype=bool)
    wrist_mask[:, 0] = True
    return output, FrameCoverage(
        raw_valid_pixel_fraction=float(np.mean(depth_mm > 0)),
        joint_count=int(in_frame.size),
        positive_depth_joint_count=int(np.count_nonzero(positive)),
        in_frame_joint_count=int(np.count_nonzero(in_frame)),
        raw_depth_valid_at_joint_pixel_count=int(np.count_nonzero(sampled_depth_valid)),
        wrist_count=2,
        in_frame_wrist_count=int(np.count_nonzero(in_frame & wrist_mask)),
        raw_depth_valid_at_wrist_pixel_count=int(
            np.count_nonzero(sampled_depth_valid & wrist_mask)
        ),
    )


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _artifact(path: Path) -> dict[str, Any]:
    source = Path(path)
    return {
        "path": str(source.resolve()),
        "bytes": source.stat().st_size,
        "sha256": rgb_overlay._sha256(source),
    }


def depth_frame_mapping_rows(
    prepared: PreparedDepthMocap,
    first: int,
    stop: int,
    coverage: Sequence[FrameCoverage],
    decoded_frames: Sequence[DepthFrameMetadata],
) -> list[dict[str, int | float]]:
    """Build an auditable raw-depth-frame to CMAvatar interpolation map."""

    count = stop - first
    if first < 0 or stop > len(prepared.depth_refs) or count <= 0:
        raise ValueError(f"Invalid depth frame interval [{first}, {stop})")
    if stop > len(prepared.rgb_timestamp_ns):
        raise ValueError("Depth frame-map interval extends into an unmatched RGB tail")
    if len(coverage) != count or len(decoded_frames) != count:
        raise ValueError("Depth frame-map inputs do not cover the selected interval")
    if not np.all(prepared.mocap_valid[first:stop]):
        raise ValueError("Depth frame-map interval contains invalid MOCAP frames")

    rows: list[dict[str, int | float]] = []
    for output_frame, source_frame in enumerate(range(first, stop)):
        metadata = decoded_frames[output_frame]
        frame_coverage = coverage[output_frame]
        if metadata.message_index != source_frame:
            raise ValueError("Decoded depth metadata is not source-frame contiguous")
        low = int(prepared.interpolation.low_indices[source_frame])
        high = int(prepared.interpolation.high_indices[source_frame])
        wrists, positive, _ = project_world_to_depth(
            prepared.mocap_mm[source_frame, :, 0], prepared.calibration
        )
        in_frame = positive & _inside_frame(
            wrists, prepared.calibration.width, prepared.calibration.height
        )
        rgb_timestamp_ns = int(prepared.rgb_timestamp_ns[source_frame])
        depth_timestamp_ns = int(prepared.depth_timestamp_ns[source_frame])
        depth_device_s = float(prepared.depth_device_s[source_frame])
        rows.append(
            {
                "output_frame": output_frame,
                "source_depth_frame": source_frame,
                "bag_message_timestamp_ns": metadata.message_timestamp_ns,
                "depth_device_timestamp_us": metadata.device_timestamp_us,
                "depth_frame_number": metadata.frame_number,
                "source_bag_elapsed_s": depth_device_s
                - float(prepared.depth_device_s[0]),
                "matched_clip_bag_elapsed_s": depth_device_s
                - float(prepared.depth_device_s[first]),
                "same_index_rgb_device_timestamp_us": (rgb_timestamp_ns + 500)
                // 1000,
                "depth_minus_rgb_device_timestamp_ms": (
                    depth_timestamp_ns - rgb_timestamp_ns
                )
                / 1e6,
                "cmavatar_target_timestamp_ns": int(
                    round(float(prepared.cmavatar_s[source_frame]) * 1e9)
                ),
                "mocap_low_frame_counter": int(prepared.mocap.frame_counters[low]),
                "mocap_high_frame_counter": int(prepared.mocap.frame_counters[high]),
                "mocap_low_timestamp_ns": int(
                    round(float(prepared.mocap.times_s[low]) * 1e9)
                ),
                "mocap_high_timestamp_ns": int(
                    round(float(prepared.mocap.times_s[high]) * 1e9)
                ),
                "interpolation_alpha": float(
                    prepared.interpolation.alpha[source_frame]
                ),
                "bracket_span_ms": float(
                    prepared.interpolation.bracket_span_s[source_frame] * 1000.0
                ),
                "nearest_mocap_sample_delta_ms": float(
                    prepared.interpolation.nearest_delta_s[source_frame] * 1000.0
                ),
                "raw_valid_pixel_fraction": frame_coverage.raw_valid_pixel_fraction,
                "projected_joint_inside_count": frame_coverage.in_frame_joint_count,
                "projected_joint_nonzero_depth_count": (
                    frame_coverage.raw_depth_valid_at_joint_pixel_count
                ),
                "left_wrist_depth_x": float(wrists[0, 0]),
                "left_wrist_depth_y": float(wrists[0, 1]),
                "left_wrist_inside_frame": int(in_frame[0]),
                "right_wrist_depth_x": float(wrists[1, 0]),
                "right_wrist_depth_y": float(wrists[1, 1]),
                "right_wrist_inside_frame": int(in_frame[1]),
            }
        )
    return rows


def write_depth_frame_mapping(
    path: Path,
    prepared: PreparedDepthMocap,
    first: int,
    stop: int,
    coverage: Sequence[FrameCoverage],
    decoded_frames: Sequence[DepthFrameMetadata],
) -> Path:
    rows = depth_frame_mapping_rows(
        prepared, first, stop, coverage, decoded_frames
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def depth_overlay_metrics(
    prepared: PreparedDepthMocap,
    first: int,
    stop: int,
    coverage: Sequence[FrameCoverage],
    decoded_frames: Sequence[DepthFrameMetadata],
    artifacts: Mapping[str, Path],
    *,
    min_depth_mm: float,
    max_depth_mm: float,
) -> dict[str, Any]:
    if stop <= first or len(coverage) != stop - first or len(decoded_frames) != stop - first:
        raise ValueError("Metrics inputs do not cover the selected depth interval")
    selected = slice(first, stop)
    bracket = rgb_overlay._distribution_ms(
        prepared.interpolation.bracket_span_s[selected]
    )
    nearest = rgb_overlay._distribution_ms(
        prepared.interpolation.nearest_delta_s[selected]
    )
    depth_minus_rgb_ms = prepared.depth_rgb_pairing["paired_depth_minus_rgb_ms"]
    calibration = prepared.calibration
    rotation = calibration.world_to_depth[:3, :3]
    total_joints = sum(item.joint_count for item in coverage)
    positive_joints = sum(item.positive_depth_joint_count for item in coverage)
    in_frame_joints = sum(item.in_frame_joint_count for item in coverage)
    depth_valid_joints = sum(
        item.raw_depth_valid_at_joint_pixel_count for item in coverage
    )
    total_wrists = sum(item.wrist_count for item in coverage)
    in_frame_wrists = sum(item.in_frame_wrist_count for item in coverage)
    depth_valid_wrists = sum(
        item.raw_depth_valid_at_wrist_pixel_count for item in coverage
    )
    decoded_timestamp_us = [item.device_timestamp_us for item in decoded_frames]
    decoded_frame_numbers = [item.frame_number for item in decoded_frames]
    decoded_frame_numbers_increasing = bool(
        len(decoded_frame_numbers) < 2 or np.all(np.diff(decoded_frame_numbers) > 0)
    )
    decoded_timestamps_increasing = bool(
        len(decoded_timestamp_us) < 2 or np.all(np.diff(decoded_timestamp_us) > 0)
    )
    decoded_timestamps_match_index = bool(
        all(
            (frame.message_timestamp_ns + 500) // 1000
            == frame.device_timestamp_us
            for frame in decoded_frames
        )
    )
    if not (
        decoded_frame_numbers_increasing
        and decoded_timestamps_increasing
        and decoded_timestamps_match_index
    ):
        raise ValueError("Decoded depth frame-number/timestamp gate failed")
    input_paths = {
        "raw_rgbd_bag": prepared.bag_path,
        "camera_intrinsics_json": prepared.segment_dir
        / "原始BAG与内参"
        / "camera_1_intrinsics.json",
        "clock_mapping": prepared.segment_dir
        / "同步校验"
        / "camera_cmavatar_alignment.csv",
        "common_interval_sync_report": prepared.segment_dir
        / "同步校验"
        / "common_interval_sync_report.json",
        "mocap_human": prepared.mocap_path,
        "camera_to_world": calibration.path,
    }
    return {
        "schema": DEPTH_OVERLAY_SCHEMA,
        "artifact_type": "evaluation_metrics",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "segment": prepared.segment_dir.name,
        "status": "raw_metric_depth_timestamp_mapped_mocap_projection_visualization",
        "claim": (
            "Raw mono16 depth frames use their own device timestamps, CMAvatar is "
            "interpolated at those timestamps, and both 21-joint hands are projected "
            "with the frozen delivered world-to-depth SE(3). Metrics describe timing "
            "and coverage only; skeleton-to-visible-surface distance is not accuracy."
        ),
        "inputs": {
            name: _artifact(path) for name, path in input_paths.items()
        },
        "depth_frame_contract": {
            "topic": DEPTH_TOPIC,
            "encoding": "mono16",
            "units": "millimetres",
            "raw_bag_depth_messages": len(prepared.depth_refs),
            "raw_bag_rgb_messages": len(prepared.rgb_device_s),
            "depth_minus_rgb_message_count": len(prepared.depth_refs)
            - len(prepared.rgb_device_s),
            "rgb_depth_pairing": prepared.depth_rgb_pairing,
            "same_index_is_nearest_for_all_paired_messages": prepared.depth_rgb_pairing[
                "same_index_is_nearest_for_all_paired_messages"
            ],
            "extra_depth_tail_is_not_forced_onto_rgb": prepared.depth_rgb_pairing[
                "extra_depth_is_validated_tail_frame"
            ],
            "source_frame_first": first,
            "source_frame_last_inclusive": stop - 1,
            "published_clip_frames": stop - first,
            "frame_mapping_rows": stop - first,
            "frame_mapping_output_and_source_frames_contiguous": True,
            "published_clip_all_frames_mocap_valid": bool(
                np.all(prepared.mocap_valid[selected])
            ),
            "trimmed_prefix_depth_frames": first,
            "trimmed_suffix_depth_frames": len(prepared.depth_refs) - stop,
            "covers_full_synchronized_depth_interval": (
                (first, stop) == rgb_overlay.synchronized_interval(prepared.mocap_valid)
            ),
            "resolution": [calibration.width, calibration.height],
            "fps": calibration.fps,
            "preview_fixed_metric_range_mm": [min_depth_mm, max_depth_mm],
            "decoded_frame_numbers_strictly_increasing": decoded_frame_numbers_increasing,
            "decoded_device_timestamps_strictly_increasing": decoded_timestamps_increasing,
            "decoded_device_timestamps_match_bag_index": decoded_timestamps_match_index,
            "raw_valid_pixel_fraction": {
                "median": float(
                    np.median([item.raw_valid_pixel_fraction for item in coverage])
                ),
                "p05": float(
                    np.percentile(
                        [item.raw_valid_pixel_fraction for item in coverage], 5
                    )
                ),
                "p95": float(
                    np.percentile(
                        [item.raw_valid_pixel_fraction for item in coverage], 95
                    )
                ),
            },
        },
        "temporal_alignment": {
            "time_key": "depth Image.timestamp_usec / BAG index timestamp",
            "mapping": (
                "depth device acquisition timestamp -> delivered exposure-time "
                "ClockMapping -> CMAvatar time -> linear 120 Hz position interpolation"
            ),
            "depth_timestamp_used_directly_not_rgb_timestamp_substituted": True,
            "paired_depth_minus_rgb_device_timestamp_ms": depth_minus_rgb_ms,
            "mocap_bracket_span_ms": bracket,
            "nearest_mocap_sample_delta_ms": nearest,
            "max_allowed_bracket_span_ms": prepared.max_interpolation_gap_ms,
            "mocap_sample_timebase": prepared.mocap.timestamp_fit,
            "clock_model_estimated_uncertainty_ms": float(
                prepared.sync_report["clock_model"]["estimated_clock_uncertainty_ms"]
            ),
            "all_published_frames_have_valid_interpolation": bool(
                np.all(prepared.mocap_valid[selected])
            ),
        },
        "spatial_projection_coverage": {
            "calibration_schema": calibration.payload["schema"],
            "calibration_status": calibration.payload["status"],
            "calibration_method": calibration.payload["method"],
            "fixed_camera_assumption": calibration.payload["fixed_camera_assumption"],
            "recorded_camera_contract": prepared.camera_contract,
            "world_to_depth_camera_frozen_no_refit": True,
            "world_to_depth_camera": calibration.world_to_depth.tolist(),
            "rotation_determinant": float(np.linalg.det(rotation)),
            "rotation_orthogonality_max_abs_error": float(
                np.max(np.abs(rotation.T @ rotation - np.eye(3)))
            ),
            "projected_joint_samples": total_joints,
            "positive_camera_depth_fraction": _fraction(positive_joints, total_joints),
            "inside_depth_frame_fraction": _fraction(in_frame_joints, total_joints),
            "raw_depth_valid_at_projected_joint_pixel_fraction": _fraction(
                depth_valid_joints, in_frame_joints
            ),
            "wrist_samples": total_wrists,
            "wrist_inside_depth_frame_fraction": _fraction(
                in_frame_wrists, total_wrists
            ),
            "raw_depth_valid_at_projected_wrist_pixel_fraction": _fraction(
                depth_valid_wrists, in_frame_wrists
            ),
            "skeleton_to_surface_error_measured": False,
            "independent_depth_joint_labels_available": False,
        },
        "artifacts": {name: _artifact(path) for name, path in artifacts.items()},
        "interpretation_limits": [
            "A depth pixel is the first visible surface on a ray; a CMAvatar joint is not that surface, so their range difference is not an accuracy metric.",
            "Projection coverage and nonzero-depth coverage are diagnostics, not independent pose ground truth.",
            "The fixed world-to-depth transform was manually established from the CS-400/table recording and is reused under the fixed-camera assumption.",
            "The delivered clock uncertainty is a software-model estimate, not a shared hardware-trigger measurement.",
            "Human.cma provides no per-joint visibility, residual, occlusion, or gap-fill quality field.",
        ],
    }


def _render_depth_mocap_video_staged(
    prepared: PreparedDepthMocap,
    output_path: Path,
    *,
    snapshot_count: int = 6,
    h264: bool = True,
    min_depth_mm: float = 350.0,
    max_depth_mm: float = 1800.0,
    max_frames: int | None = None,
) -> dict[str, Path]:
    synchronized_first, synchronized_stop = rgb_overlay.synchronized_interval(
        prepared.mocap_valid
    )
    first = synchronized_first
    stop = synchronized_stop
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        stop = min(stop, first + max_frames)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    calibration = prepared.calibration
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        calibration.fps,
        (calibration.width, calibration.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create depth overlay video: {output}")
    snapshot_indices: set[int] = set()
    if snapshot_count > 0:
        snapshot_indices = set(
            np.rint(
                np.linspace(first, stop - 1, min(snapshot_count, stop - first))
            ).astype(int)
        )
    snapshots: list[np.ndarray] = []
    coverage: list[FrameCoverage] = []
    decoded_frames: list[DepthFrameMetadata] = []
    try:
        for depth_frame in iter_bag_depth_frames(
            prepared.bag_path, prepared.depth_refs, first, stop
        ):
            if depth_frame.width != calibration.width or depth_frame.height != calibration.height:
                raise ValueError("Decoded depth frame does not match calibrated profile")
            rendered, frame_coverage = render_depth_mocap_frame(
                prepared,
                depth_frame.message_index,
                depth_frame.message_index - first,
                depth_frame.depth_mm,
                first_source_frame=first,
                min_depth_mm=min_depth_mm,
                max_depth_mm=max_depth_mm,
            )
            writer.write(rendered)
            coverage.append(frame_coverage)
            # Do not retain the 640 x 576 uint16 image after it has been encoded.
            # Full takes contain roughly 1,800 frames, so retaining only immutable
            # provenance keeps a three-take delivery comfortably memory bounded.
            decoded_frames.append(
                DepthFrameMetadata(
                    message_index=depth_frame.message_index,
                    message_timestamp_ns=depth_frame.message_timestamp_ns,
                    device_timestamp_us=depth_frame.device_timestamp_us,
                    frame_number=depth_frame.frame_number,
                )
            )
            if depth_frame.message_index in snapshot_indices:
                snapshots.append(rendered.copy())
            rendered_count = depth_frame.message_index - first + 1
            if rendered_count == 1 or rendered_count % 300 == 0:
                print(
                    f"{prepared.segment_dir.name}: rendered depth {rendered_count}/{stop - first}",
                    flush=True,
                )
    finally:
        writer.release()
    if len(decoded_frames) != stop - first:
        raise RuntimeError(
            f"Decoded {len(decoded_frames)} depth frames, expected {stop - first}"
        )

    contact_path = output.with_suffix(".contact.jpg")
    written_contact = rgb_overlay._write_contact_sheet(contact_path, snapshots)
    artifacts: dict[str, Path] = {"mp4v_preview": output}
    if written_contact is not None:
        artifacts["contact_sheet"] = written_contact
    if h264:
        h264_path = output.with_name(output.stem + "_h264.mp4")
        rgb_overlay.transcode_h264(output, h264_path)
        artifacts["h264_delivery_video"] = h264_path
    alignment_path = output.with_suffix(".alignment.csv")
    artifacts["frame_mapping_csv"] = write_depth_frame_mapping(
        alignment_path,
        prepared,
        first,
        stop,
        coverage,
        decoded_frames,
    )
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            depth_overlay_metrics(
                prepared,
                first,
                stop,
                coverage,
                decoded_frames,
                artifacts,
                min_depth_mm=min_depth_mm,
                max_depth_mm=max_depth_mm,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts["metrics_json"] = metrics_path
    return artifacts


def render_depth_mocap_video(
    prepared: PreparedDepthMocap,
    output_path: Path,
    *,
    snapshot_count: int = 6,
    h264: bool = True,
    min_depth_mm: float = 350.0,
    max_depth_mm: float = 1800.0,
    max_frames: int | None = None,
) -> dict[str, Path]:
    """Render in a staging directory and publish metrics last as a commit marker.

    A consumer must validate the hashes recorded in metrics before accepting the
    media.  If rendering or transcoding fails, no final path changes.  If the
    same-filesystem replace phase is interrupted, the previous metrics remains
    and fails closed against any already-replaced asset until the next run.
    """

    published_output = Path(output_path).resolve()
    published_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{published_output.stem}.staging-",
        dir=published_output.parent,
    ) as temporary:
        staged_output = Path(temporary) / published_output.name
        staged = _render_depth_mocap_video_staged(
            prepared,
            staged_output,
            snapshot_count=snapshot_count,
            h264=h264,
            min_depth_mm=min_depth_mm,
            max_depth_mm=max_depth_mm,
            max_frames=max_frames,
        )
        staged_metrics = staged["metrics_json"]
        payload = json.loads(staged_metrics.read_text(encoding="utf-8"))
        published: dict[str, Path] = {}
        for name, staged_path in staged.items():
            if name == "metrics_json":
                continue
            final_path = published_output.parent / staged_path.name
            recorded = payload["artifacts"][name]
            if staged_path.stat().st_size != int(recorded["bytes"]):
                raise ValueError(f"Staged artifact size changed before publish: {staged_path}")
            if rgb_overlay._sha256(staged_path) != recorded["sha256"]:
                raise ValueError(f"Staged artifact hash changed before publish: {staged_path}")
            recorded["path"] = str(final_path.resolve())
            published[name] = final_path

        staged_metrics.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for name, final_path in published.items():
            os.replace(staged[name], final_path)
        final_metrics = published_output.with_suffix(".metrics.json")
        os.replace(staged_metrics, final_metrics)
        published["metrics_json"] = final_metrics
        return published


def _gap_stem_token(max_interpolation_gap_ms: float) -> str:
    value = float(max_interpolation_gap_ms)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("max_interpolation_gap_ms must be finite and positive")
    if abs(value - 25.0) <= 1e-9:
        return "strict25"
    encoded = np.format_float_positional(value, trim="-").replace(".", "p")
    return f"gap{encoded}"


def depth_output_stem(
    segment_name: str,
    max_interpolation_gap_ms: float,
    *,
    max_frames: int | None = None,
) -> str:
    stem = f"{segment_name}_depth_mocap_aligned_{_gap_stem_token(max_interpolation_gap_ms)}"
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        stem += f"_smoke{max_frames}"
    return stem


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Render synchronized CMAvatar hands on raw metric depth frames."
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
        default=project_root / "outputs" / "depth_mocap_alignment_review",
    )
    parser.add_argument("--snapshot-count", type=int, default=6)
    parser.add_argument("--max-interpolation-gap-ms", type=float, default=25.0)
    parser.add_argument("--min-depth-mm", type=float, default=350.0)
    parser.add_argument("--max-depth-mm", type=float, default=1800.0)
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional smoke-test limit; omitted means the full synchronized interval.",
    )
    parser.add_argument("--no-h264", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    calibration_path = (
        args.calibration.expanduser().resolve()
        if args.calibration is not None
        else viz._default_calibration(dataset_root)
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for key in args.segment or ("01", "02", "03"):
        segment = viz._discover_segment(dataset_root, key)
        print(f"Preparing raw depth for {segment.name}", flush=True)
        prepared = prepare_depth_mocap(
            segment,
            calibration_path,
            max_interpolation_gap_ms=args.max_interpolation_gap_ms,
        )
        stem = depth_output_stem(
            segment.name,
            args.max_interpolation_gap_ms,
            max_frames=args.max_frames,
        )
        artifacts = render_depth_mocap_video(
            prepared,
            output_dir / f"{stem}.mp4",
            snapshot_count=args.snapshot_count,
            h264=not args.no_h264,
            min_depth_mm=args.min_depth_mm,
            max_depth_mm=args.max_depth_mm,
            max_frames=args.max_frames,
        )
        print(
            json.dumps(
                {name: str(path) for name, path in artifacts.items()},
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
