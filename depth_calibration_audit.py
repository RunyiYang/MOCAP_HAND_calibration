#!/usr/bin/env python3
"""Quantitatively audit the delivered depth-camera-to-MOCAP calibration.

The delivered calibration is a fixed transform, not a per-frame camera
trajectory.  This tool keeps three questions separate:

* whether the depth-camera transform is a valid, reproducible SE(3),
* whether the physical camera stayed fixed between recordings, and
* whether the recorded depth-to-colour linear block is itself a rigid pose.

Raw ``mono16`` depth is decoded directly from the ROS1 BAG index.  The
8-bit ``Depth.mp4`` preview is never used for metric geometry.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import json
import math
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping, Sequence
import warnings

import cv2
import numpy as np

import gt_calib_viz as viz
import mocap_video_overlay as bagio


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "同步整理_20260829_三段"
CALIBRATION_PROJECT = (
    DATASET_ROOT
    / "movementcap_worldcalib_pointcloud_package_20260830"
    / "movementcap_ruler_worldcalib"
)
DEFAULT_CALIBRATION = (
    CALIBRATION_PROJECT / "results" / "manual_final" / "camera_to_world.json"
)
DEFAULT_CONFIG = CALIBRATION_PROJECT / "configs" / "manual_cs400_points.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "depth_calibration_audit"

DEPTH_TOPIC = "/cam/sensor_3/frameType_3"
ROS_OP_MESSAGE_DATA = 2

REFERENCE_WINDOWS = (
    ("first", 0, 36),
    ("middle", 36, 73),
    ("last", 73, 109),
)
RGB_REGISTRATION_FRACTIONS = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class DepthFrame:
    message_index: int
    message_timestamp_ns: int
    device_timestamp_us: int
    frame_number: int
    image_mm: np.ndarray


@dataclass(frozen=True)
class PlaneFit:
    normal: np.ndarray
    d_mm: float
    candidate_points: np.ndarray
    residual_mm: np.ndarray
    refined_inlier_count: int


@dataclass(frozen=True)
class DepthPoseFit:
    plane: PlaneFit
    world_origin_in_depth_mm: np.ndarray
    basis_depth_from_world: np.ndarray
    depth_to_world: np.ndarray
    measured_axis_angle_deg: float


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _distribution(values: np.ndarray | Sequence[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data):
        raise ValueError("Cannot summarize an empty distribution")
    return {
        "count": int(len(data)),
        "min": float(np.min(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
    }


def _read_ros_string(payload: bytes, offset: int) -> tuple[str, int]:
    if offset + 4 > len(payload):
        raise ValueError("Truncated ROS string length")
    length = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    stop = offset + length
    if stop > len(payload):
        raise ValueError("Truncated ROS string")
    return payload[offset:stop].decode("utf-8", errors="replace"), stop


def decode_orbbec_depth(
    serialized: bytes,
    *,
    message_index: int,
    message_timestamp_ns: int,
) -> DepthFrame:
    """Decode the recorder's actual Image wire layout as raw millimetres."""

    if len(serialized) < 73:
        raise ValueError("Truncated sensor_msgs/Image payload")
    _, stamp_seconds, stamp_nanoseconds = struct.unpack_from("<III", serialized, 0)
    header_timestamp_ns = int(stamp_seconds) * 1_000_000_000 + int(stamp_nanoseconds)
    if header_timestamp_ns != message_timestamp_ns:
        raise ValueError("Depth Image header time disagrees with BAG message time")
    _, offset = _read_ros_string(serialized, 12)
    if offset + 8 > len(serialized):
        raise ValueError("Truncated depth dimensions")
    height, width = struct.unpack_from("<II", serialized, offset)
    encoding, offset = _read_ros_string(serialized, offset + 8)
    if offset + 13 > len(serialized):
        raise ValueError("Truncated depth payload header")
    is_bigendian = bool(serialized[offset])
    offset += 1
    step = struct.unpack_from("<I", serialized, offset)[0]
    offset += 4
    metadata_size, packed_size = struct.unpack_from("<II", serialized, offset)
    offset += 8
    packed_stop = offset + packed_size
    if packed_stop + 32 != len(serialized):
        raise ValueError("Depth payload does not leave four uint64 metadata fields")
    if metadata_size > packed_size:
        raise ValueError("Depth metadata exceeds packed data")
    pixels = serialized[offset + metadata_size : packed_stop]
    frame_number, device_timestamp_us, _, _ = struct.unpack_from(
        "<QQQQ", serialized, packed_stop
    )
    if encoding.lower() not in {"mono16", "16uc1"}:
        raise ValueError(f"Expected mono16 metric depth, found {encoding!r}")
    if step % 2 or len(pixels) != int(height) * int(step):
        raise ValueError("Depth payload byte count does not match height*step")
    dtype = np.dtype(">u2" if is_bigendian else "<u2")
    row_values = int(step) // dtype.itemsize
    image = np.frombuffer(pixels, dtype=dtype).reshape(int(height), row_values)
    image = image[:, : int(width)].astype(np.uint16, copy=True)
    if (message_timestamp_ns + 500) // 1000 != int(device_timestamp_us):
        raise ValueError("Depth device timestamp disagrees with BAG time")
    return DepthFrame(
        message_index=int(message_index),
        message_timestamp_ns=int(message_timestamp_ns),
        device_timestamp_us=int(device_timestamp_us),
        frame_number=int(frame_number),
        image_mm=image,
    )


def read_depth_frames(
    bag_path: Path,
    sample_indices: Sequence[int],
) -> tuple[list[DepthFrame], int]:
    """Random-access selected raw depth frames through the ROS1 BAG index."""

    refs, _ = bagio._ros1_topic_message_refs(Path(bag_path), DEPTH_TOPIC)
    requested = sorted(set(int(value) for value in sample_indices))
    if not requested or requested[0] < 0 or requested[-1] >= len(refs):
        raise ValueError("Depth sample index is outside the message range")
    result: list[DepthFrame] = []
    cached_key: tuple[int, int] | None = None
    cached_chunk: bytes | None = None
    with Path(bag_path).open("rb") as handle:
        for index in requested:
            ref = refs[index]
            key = (ref.chunk_data_offset, ref.chunk_data_length)
            if key != cached_key or cached_chunk is None:
                handle.seek(ref.chunk_data_offset)
                compressed = handle.read(ref.chunk_data_length)
                if len(compressed) != ref.chunk_data_length:
                    raise ValueError("Truncated ROS1 depth chunk")
                cached_chunk = bagio._decompress_ros1_chunk(
                    compressed,
                    ref.compression,
                    ref.uncompressed_size,
                )
                cached_key = key
            nested = BytesIO(cached_chunk)
            nested.seek(ref.record_offset)
            header, serialized = viz._read_ros_record(nested, read_data=True)
            if viz._field_int(header, "op") != ROS_OP_MESSAGE_DATA:
                raise ValueError("Depth index does not point to MESSAGE_DATA")
            if viz._field_int(header, "conn") != ref.connection_id:
                raise ValueError("Depth index points to a different connection")
            result.append(
                decode_orbbec_depth(
                    serialized,
                    message_index=index,
                    message_timestamp_ns=ref.timestamp_ns,
                )
            )
    return result, len(refs)


def temporal_median_mm(frames: Sequence[DepthFrame] | np.ndarray) -> np.ndarray:
    if isinstance(frames, np.ndarray):
        stack = np.asarray(frames, dtype=np.uint16)
    else:
        stack = np.stack([frame.image_mm for frame in frames])
    work = stack.astype(np.float32)
    work[work == 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(work, axis=0)
    return np.nan_to_num(median, nan=0.0).astype(np.uint16)


def undistort_normalized(
    x_distorted: np.ndarray | float,
    y_distorted: np.ndarray | float,
    coefficients: np.ndarray,
    *,
    iterations: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    xd = np.asarray(x_distorted, dtype=np.float64)
    yd = np.asarray(y_distorted, dtype=np.float64)
    x = xd.copy()
    y = yd.copy()
    k1, k2, k3, k4, k5, k6, p1, p2 = np.asarray(
        coefficients, dtype=np.float64
    )
    for _ in range(iterations):
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (
            1.0 + k4 * r2 + k5 * r4 + k6 * r6
        )
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (xd - dx) / radial
        y = (yd - dy) / radial
    return x, y


def distort_normalized(
    x: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    k1, k2, k3, k4, k5, k6, p1, p2 = np.asarray(
        coefficients, dtype=np.float64
    )
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (
        1.0 + k4 * r2 + k5 * r4 + k6 * r6
    )
    return (
        x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
        y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
    )


def pixel_rays(
    u: np.ndarray | float,
    v: np.ndarray | float,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    fx, fy, cx, cy = np.asarray(intrinsics, dtype=np.float64)
    xd = (np.asarray(u, dtype=np.float64) - cx) / fx
    yd = (np.asarray(v, dtype=np.float64) - cy) / fy
    x, y = undistort_normalized(xd, yd, distortion)
    return np.stack([x, y, np.ones_like(x)], axis=-1)


def project_camera_points(
    points: np.ndarray,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    x = points[..., 0] / points[..., 2]
    y = points[..., 1] / points[..., 2]
    xd, yd = distort_normalized(x, y, distortion)
    fx, fy, cx, cy = np.asarray(intrinsics, dtype=np.float64)
    return np.stack([fx * xd + cx, fy * yd + cy], axis=-1)


def depth_points_for_plane(
    median_depth_mm: np.ndarray,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    fit_config: Mapping[str, Any],
    *,
    v_min: int | None = None,
    u_range: tuple[int, int] | None = None,
) -> np.ndarray:
    vv, uu = np.mgrid[: median_depth_mm.shape[0], : median_depth_mm.shape[1]]
    z = median_depth_mm.astype(np.float64)
    minimum_v = int(
        fit_config["depth_image_v_min"] if v_min is None else v_min
    )
    mask = (
        (z > float(fit_config["depth_min_mm"]))
        & (z < float(fit_config["depth_max_mm"]))
        & (vv > minimum_v)
    )
    if u_range is not None:
        mask &= (uu >= int(u_range[0])) & (uu < int(u_range[1]))
    rays = pixel_rays(uu[mask], vv[mask], intrinsics, distortion)
    return rays * z[mask, None]


def fit_plane_ransac(
    points: np.ndarray,
    *,
    threshold_mm: float,
    refine_threshold_mm: float,
    seed: int,
) -> PlaneFit:
    """Reproduce the delivered table fit while retaining untruncated residuals."""

    values = np.asarray(points, dtype=np.float64)
    if len(values) < 100:
        raise ValueError("Too few points for a stable table-plane fit")
    rng = np.random.default_rng(seed)
    sample = values[rng.choice(len(values), min(30_000, len(values)), replace=False)]
    best_count = -1
    best_normal: np.ndarray | None = None
    best_d = 0.0
    for _ in range(500):
        tri = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        length = float(np.linalg.norm(normal))
        if length < 1e-9:
            continue
        normal /= length
        d_mm = -float(normal @ tri[0])
        count = int(np.count_nonzero(np.abs(sample @ normal + d_mm) < threshold_mm))
        if count > best_count:
            best_count = count
            best_normal = normal
            best_d = d_mm
    if best_normal is None:
        raise RuntimeError("RANSAC failed to find a table plane")
    first_residual = np.abs(values @ best_normal + best_d)
    inliers = values[first_residual < refine_threshold_mm]
    centroid = np.mean(inliers, axis=0)
    _, _, right_t = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = right_t[-1]
    if float(normal @ centroid) > 0.0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    d_mm = -float(normal @ centroid)
    residual = np.abs(values @ normal + d_mm)
    return PlaneFit(
        normal=normal,
        d_mm=d_mm,
        candidate_points=values,
        residual_mm=residual,
        refined_inlier_count=int(len(inliers)),
    )


def reconstruct_marker(
    pixel: Iterable[float],
    marker_height_mm: float,
    plane_normal: np.ndarray,
    plane_d_mm: float,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    u, v = (float(value) for value in pixel)
    ray = pixel_rays(u, v, intrinsics, distortion)
    scale = (marker_height_mm - plane_d_mm) / float(plane_normal @ ray)
    center = ray * scale
    foot = center - marker_height_mm * plane_normal
    return center, foot


def _line_intersection(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> np.ndarray:
    first = p2 - p1
    first /= np.linalg.norm(first)
    second = q2 - q1
    second /= np.linalg.norm(second)
    coefficients = np.linalg.lstsq(
        np.column_stack([first, -second]), q1 - p1, rcond=None
    )[0]
    return 0.5 * (p1 + coefficients[0] * first + q1 + coefficients[1] * second)


def _rotate_about_axis(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64)
    unit /= np.linalg.norm(unit)
    value = np.asarray(vector, dtype=np.float64)
    return (
        value * math.cos(angle)
        + np.cross(unit, value) * math.sin(angle)
        + unit * float(unit @ value) * (1.0 - math.cos(angle))
    )


def _homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def pose_from_plane_and_markers(
    plane: PlaneFit,
    marker_pixels: Mapping[str, Sequence[float]],
    marker_height_mm: float,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
) -> DepthPoseFit:
    feet: dict[str, np.ndarray] = {}
    for name, pixel in marker_pixels.items():
        _, feet[name] = reconstruct_marker(
            pixel,
            marker_height_mm,
            plane.normal,
            plane.d_mm,
            intrinsics,
            distortion,
        )
    origin = _line_intersection(
        feet["long_far"],
        feet["long_near"],
        feet["short_near"],
        feet["short_far"],
    )
    x_measured = 0.5 * (feet["long_far"] + feet["long_near"]) - origin
    x_measured -= plane.normal * float(plane.normal @ x_measured)
    x_measured /= np.linalg.norm(x_measured)
    y_measured = 0.5 * (feet["short_near"] + feet["short_far"]) - origin
    y_measured -= plane.normal * float(plane.normal @ y_measured)
    y_measured /= np.linalg.norm(y_measured)
    measured_angle = math.degrees(
        math.acos(np.clip(float(x_measured @ y_measured), -1.0, 1.0))
    )
    y_from_x = np.cross(plane.normal, x_measured)
    y_from_x /= np.linalg.norm(y_from_x)
    if float(y_from_x @ y_measured) < 0.0:
        raise ValueError("Marker labels imply a reflected world frame")
    residual = math.atan2(
        float(plane.normal @ np.cross(y_from_x, y_measured)),
        float(y_from_x @ y_measured),
    )
    x_axis = _rotate_about_axis(x_measured, plane.normal, 0.5 * residual)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(plane.normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    basis_depth_from_world = np.column_stack([x_axis, y_axis, plane.normal])
    depth_to_world_rotation = basis_depth_from_world.T
    depth_to_world_translation = -depth_to_world_rotation @ origin
    return DepthPoseFit(
        plane=plane,
        world_origin_in_depth_mm=origin,
        basis_depth_from_world=basis_depth_from_world,
        depth_to_world=_homogeneous(
            depth_to_world_rotation, depth_to_world_translation
        ),
        measured_axis_angle_deg=measured_angle,
    )


def rotation_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    delta = np.asarray(first) @ np.asarray(second).T
    cosine = np.clip((float(np.trace(delta)) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def fit_depth_pose(
    median_depth_mm: np.ndarray,
    calibration: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    v_min: int | None = None,
    u_range: tuple[int, int] | None = None,
) -> DepthPoseFit:
    profile = calibration["orbbec_profiles_from_bag"]["depth"]
    intrinsics = np.asarray(profile["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
    distortion = np.asarray(
        profile["distortion_orbbec_k1_k2_k3_k4_k5_k6_p1_p2"],
        dtype=np.float64,
    )
    fit_config = config["table_plane_fit"]
    points = depth_points_for_plane(
        median_depth_mm,
        intrinsics,
        distortion,
        fit_config,
        v_min=v_min,
        u_range=u_range,
    )
    plane = fit_plane_ransac(
        points,
        threshold_mm=float(fit_config["ransac_threshold_mm"]),
        refine_threshold_mm=float(fit_config["refine_threshold_mm"]),
        seed=int(fit_config["random_seed"]),
    )
    return pose_from_plane_and_markers(
        plane,
        config["depth_marker_centers_px"],
        float(config["marker_center_height_above_table_mm"]),
        intrinsics,
        distortion,
    )


def _pose_delta(fit: DepthPoseFit, reference: DepthPoseFit) -> dict[str, float]:
    return {
        "plane_normal_angle_deg": math.degrees(
            math.acos(
                np.clip(float(fit.plane.normal @ reference.plane.normal), -1.0, 1.0)
            )
        ),
        "plane_d_delta_mm": float(fit.plane.d_mm - reference.plane.d_mm),
        "world_origin_in_depth_delta_mm": float(
            np.linalg.norm(
                fit.world_origin_in_depth_mm - reference.world_origin_in_depth_mm
            )
        ),
        "depth_to_world_rotation_delta_deg": rotation_angle_deg(
            fit.depth_to_world[:3, :3], reference.depth_to_world[:3, :3]
        ),
        "depth_camera_origin_in_world_delta_mm": float(
            np.linalg.norm(fit.depth_to_world[:3, 3] - reference.depth_to_world[:3, 3])
        ),
    }


def _plane_summary(plane: PlaneFit) -> dict[str, Any]:
    bounded = plane.residual_mm[plane.residual_mm < 30.0]
    return {
        "normal_up": plane.normal,
        "d_mm": plane.d_mm,
        "candidate_point_count": int(len(plane.candidate_points)),
        "refined_inlier_count": plane.refined_inlier_count,
        "inlier_ratio_below_5mm": float(np.mean(plane.residual_mm < 5.0)),
        "all_candidate_absolute_residual_mm": _distribution(plane.residual_mm),
        "legacy_below_30mm_absolute_residual_mm": _distribution(bounded),
        "candidate_count_at_or_above_30mm": int(
            np.count_nonzero(plane.residual_mm >= 30.0)
        ),
        "legacy_metric_warning": (
            "The delivered P95 was computed only after residuals >=30 mm were "
            "discarded; use all_candidate_absolute_residual_mm for an untruncated view."
        ),
    }


def marker_click_sensitivity(
    reference: DepthPoseFit,
    calibration: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    trials: int,
) -> dict[str, Any]:
    profile = calibration["orbbec_profiles_from_bag"]["depth"]
    intrinsics = np.asarray(profile["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
    distortion = np.asarray(
        profile["distortion_orbbec_k1_k2_k3_k4_k5_k6_p1_p2"],
        dtype=np.float64,
    )
    marker_height = float(config["marker_center_height_above_table_mm"])
    base = {
        name: np.asarray(pixel, dtype=np.float64)
        for name, pixel in config["depth_marker_centers_px"].items()
    }
    generator = np.random.default_rng(20260831)
    scenarios: dict[str, Any] = {}
    for sigma in (0.5, 1.0, 2.0):
        origin_errors: list[float] = []
        rotation_errors: list[float] = []
        for _ in range(trials):
            perturbed = {
                name: pixel + generator.normal(0.0, sigma, size=2)
                for name, pixel in base.items()
            }
            fit = pose_from_plane_and_markers(
                reference.plane,
                perturbed,
                marker_height,
                intrinsics,
                distortion,
            )
            origin_errors.append(
                float(
                    np.linalg.norm(
                        fit.world_origin_in_depth_mm
                        - reference.world_origin_in_depth_mm
                    )
                )
            )
            rotation_errors.append(
                rotation_angle_deg(
                    fit.depth_to_world[:3, :3], reference.depth_to_world[:3, :3]
                )
            )
        scenarios[f"gaussian_sigma_{sigma:g}px"] = {
            "origin_delta_mm": _distribution(origin_errors),
            "rotation_delta_deg": _distribution(rotation_errors),
        }

    height_effects: dict[str, Any] = {}
    for delta_mm in (-1.0, 1.0):
        fit = pose_from_plane_and_markers(
            reference.plane,
            base,
            marker_height + delta_mm,
            intrinsics,
            distortion,
        )
        height_effects[f"{delta_mm:+.0f}mm"] = _pose_delta(fit, reference)
    return {
        "monte_carlo_seed": 20260831,
        "trials_per_scenario": int(trials),
        "interpretation": (
            "Sensitivity to assumed independent pixel noise; not a measured "
            "confidence interval for the manual clicks."
        ),
        "pixel_scenarios": scenarios,
        "marker_height_sensitivity": height_effects,
    }


def closest_rotation(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left, singular, right_t = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return rotation, singular


def d2c_audit(calibration: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    transforms = calibration["transforms"]
    delivered = np.asarray(
        transforms["depth_camera_to_color_camera"], dtype=np.float64
    )
    linear = delivered[:3, :3]
    rotation, singular = closest_rotation(linear)
    rigid = delivered.copy()
    rigid[:3, :3] = rotation
    orthogonality_error = float(
        np.max(np.abs(linear.T @ linear - np.eye(3, dtype=np.float64)))
    )

    color = calibration["orbbec_profiles_from_bag"]["color"]
    intrinsics = np.asarray(color["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
    distortion = np.asarray(
        color["distortion_orbbec_k1_k2_k3_k4_k5_k6_p1_p2"],
        dtype=np.float64,
    )
    xx, yy, zz = np.meshgrid(
        np.linspace(-400.0, 400.0, 9),
        np.linspace(-100.0, 300.0, 5),
        np.linspace(700.0, 1300.0, 7),
        indexing="ij",
    )
    workspace = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    affine_points = workspace @ linear.T + delivered[:3, 3]
    rigid_points = workspace @ rotation.T + delivered[:3, 3]
    affine_pixels = project_camera_points(affine_points, intrinsics, distortion)
    rigid_pixels = project_camera_points(rigid_points, intrinsics, distortion)
    pixel_delta = np.linalg.norm(affine_pixels - rigid_pixels, axis=1)

    depth_to_world = np.asarray(transforms["depth_camera_to_world"], dtype=np.float64)
    rigid_color_to_world = depth_to_world @ np.linalg.inv(rigid)
    delivered_color_origin = np.asarray(
        calibration["camera_origins_in_world_mm"]["color_camera"], dtype=np.float64
    )
    candidate = {
        "schema": "gt_calib.d2c_rigid_candidate.v1",
        "status": "candidate_only_requires_independent_rgb_depth_validation",
        "convention": transforms["convention"],
        "source": "polar decomposition of delivered Orbbec forward linear block",
        "must_not_replace_delivered_projection_silently": True,
        "depth_camera_to_color_camera_rigid_candidate": rigid,
        "color_camera_to_depth_camera_rigid_candidate": np.linalg.inv(rigid),
        "color_camera_to_world_rigid_candidate": rigid_color_to_world,
        "world_to_color_camera_rigid_candidate": np.linalg.inv(rigid_color_to_world),
        "color_camera_origin_in_world_mm": rigid_color_to_world[:3, 3],
    }
    result = {
        "source": "BAG /cam/streamProfileType_3 via delivered calibration JSON",
        "hardware_align_enabled": False,
        "delivered_forward_mapping": delivered,
        "determinant": float(np.linalg.det(linear)),
        "singular_values": singular,
        "max_abs_orthogonality_error": orthogonality_error,
        "is_valid_rotation_at_1e-6": bool(
            abs(float(np.linalg.det(linear)) - 1.0) <= 1e-6
            and orthogonality_error <= 1e-6
        ),
        "nearest_so3_candidate": rotation,
        "affine_vs_rigid_workspace_projection_delta_px": _distribution(pixel_delta),
        "candidate_color_origin_delta_from_delivered_mm": float(
            np.linalg.norm(rigid_color_to_world[:3, 3] - delivered_color_origin)
        ),
        "decision": (
            "Keep the delivered affine block only as the current forward projection "
            "baseline. The rigid candidate is not promoted without independent RGB-D "
            "correspondences."
        ),
    }
    return result, candidate


def _video_frame(path: Path, fraction: float) -> tuple[np.ndarray, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open RGB video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    index = int(round((frame_count - 1) * fraction))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not decode frame {index} from {path}")
    return frame, index, frame_count


def _register_rgb_pair(
    reference: np.ndarray,
    target: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    if reference.shape != target.shape:
        raise ValueError("RGB registration requires equal-sized images")
    height, width = reference.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    # Two side-background bands retain the room/cage while excluding the
    # central actor and most of the tabletop interaction region.
    mask[: min(720, height), : min(700, width)] = 255
    mask[: min(720, height), min(1250, width) :] = 255
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02)
    first_keypoints, first_descriptors = sift.detectAndCompute(
        cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), mask
    )
    second_keypoints, second_descriptors = sift.detectAndCompute(
        cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), mask
    )
    if first_descriptors is None or second_descriptors is None:
        raise RuntimeError("SIFT found no static-background descriptors")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        first_descriptors, second_descriptors, k=2
    )
    good = [first for first, second in pairs if first.distance < 0.7 * second.distance]
    if len(good) < 20:
        raise RuntimeError("Too few static-background feature matches")
    source = np.float32([first_keypoints[item.queryIdx].pt for item in good])
    destination = np.float32([second_keypoints[item.trainIdx].pt for item in good])
    affine, inliers = cv2.estimateAffinePartial2D(
        source,
        destination,
        method=cv2.RANSAC,
        ransacReprojThreshold=1.5,
        maxIters=10_000,
        confidence=0.999,
        refineIters=50,
    )
    if affine is None or inliers is None:
        raise RuntimeError("Static-background RANSAC failed")
    selected = inliers.ravel().astype(bool)
    prediction = source @ affine[:, :2].T + affine[:, 2]
    reprojection = np.linalg.norm(prediction - destination, axis=1)[selected]
    scale = float(math.hypot(affine[0, 0], affine[1, 0]))
    rotation_deg = math.degrees(math.atan2(affine[1, 0], affine[0, 0]))
    translation = affine[:, 2]
    metrics = {
        "reference_keypoints": int(len(first_keypoints)),
        "target_keypoints": int(len(second_keypoints)),
        "ratio_test_matches": int(len(good)),
        "ransac_inliers": int(np.count_nonzero(selected)),
        "ransac_inlier_ratio": float(np.mean(selected)),
        "affine_partial_2d": affine,
        "translation_xy_px": translation,
        "translation_magnitude_px": float(np.linalg.norm(translation)),
        "rotation_deg": rotation_deg,
        "scale": scale,
        "absolute_scale_error": abs(scale - 1.0),
        "inlier_reprojection_error_px": _distribution(reprojection),
    }
    selected_matches = [item for item, keep in zip(good, selected, strict=True) if keep]
    selected_matches.sort(key=lambda item: item.distance)
    diagnostic = cv2.drawMatches(
        reference,
        first_keypoints,
        target,
        second_keypoints,
        selected_matches[:60],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return metrics, diagnostic


def rgb_fixed_camera_audit(
    dataset_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    recordings = {
        key: next(path for path in dataset_root.iterdir() if path.name.startswith(f"{key}_"))
        for key in ("00", "01", "02", "03")
    }
    rows: list[np.ndarray] = []
    results: list[dict[str, Any]] = []
    for key in ("01", "02", "03"):
        per_fraction: list[dict[str, Any]] = []
        middle_diagnostic: np.ndarray | None = None
        for fraction in RGB_REGISTRATION_FRACTIONS:
            reference, reference_index, reference_count = _video_frame(
                recordings["00"] / "视频" / "RGB.mp4", fraction
            )
            target, target_index, target_count = _video_frame(
                recordings[key] / "视频" / "RGB.mp4", fraction
            )
            metrics, diagnostic = _register_rgb_pair(reference, target)
            metrics.update(
                {
                    "fraction": fraction,
                    "reference_frame_index": reference_index,
                    "reference_frame_count": reference_count,
                    "target_frame_index": target_index,
                    "target_frame_count": target_count,
                }
            )
            per_fraction.append(metrics)
            if fraction == 0.5:
                middle_diagnostic = diagnostic
        if middle_diagnostic is None:
            raise AssertionError("Middle RGB registration diagnostic was not generated")
        canvas = cv2.resize(middle_diagnostic, (1280, 360), interpolation=cv2.INTER_AREA)
        cv2.rectangle(canvas, (0, 0), (1280, 50), (9, 21, 36), -1)
        median_translation = float(
            np.median([item["translation_magnitude_px"] for item in per_fraction])
        )
        max_translation = float(
            np.max([item["translation_magnitude_px"] for item in per_fraction])
        )
        max_rotation = float(
            np.max(np.abs([item["rotation_deg"] for item in per_fraction]))
        )
        max_scale_error = float(
            np.max([item["absolute_scale_error"] for item in per_fraction])
        )
        label = (
            f"00 -> {recordings[key].name} | median/max translation "
            f"{median_translation:.3f}/{max_translation:.3f}px | "
            f"max rotation {max_rotation:.4f}deg"
        )
        cv2.putText(
            canvas,
            label,
            (24, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (230, 240, 250),
            2,
            cv2.LINE_AA,
        )
        rows.append(canvas)
        passed = bool(
            max_translation <= 0.5
            and max_rotation <= 0.1
            and max_scale_error <= 0.001
            and min(item["ransac_inliers"] for item in per_fraction) >= 100
        )
        results.append(
            {
                "take": recordings[key].name,
                "frame_pairs": per_fraction,
                "summary": {
                    "translation_magnitude_px_median": median_translation,
                    "translation_magnitude_px_max": max_translation,
                    "absolute_rotation_deg_max": max_rotation,
                    "absolute_scale_error_max": max_scale_error,
                    "minimum_ransac_inliers": int(
                        min(item["ransac_inliers"] for item in per_fraction)
                    ),
                },
                "acceptance_pass": passed,
            }
        )
    diagnostic_path = output_dir / "rgb_fixed_camera_registration.jpg"
    if not cv2.imwrite(
        str(diagnostic_path),
        np.vstack(rows),
        [cv2.IMWRITE_JPEG_QUALITY, 91],
    ):
        raise IOError(f"Could not write {diagnostic_path}")
    return {
        "method": (
            "SIFT + ratio test + partial-affine RANSAC on two static side-background "
            "bands; 00 compared with 01/02/03 at 10%, 50%, and 90% of each clip."
        ),
        "opencv_version": cv2.__version__,
        "thresholds": {
            "translation_magnitude_px_max": 0.5,
            "absolute_rotation_deg_max": 0.1,
            "absolute_scale_error_max": 0.001,
            "minimum_ransac_inliers": 100,
        },
        "diagnostic_image": str(diagnostic_path.relative_to(PROJECT_ROOT)),
        "takes": results,
        "all_takes_pass": all(item["acceptance_pass"] for item in results),
        "scope_limit": (
            "Strong fixed-view evidence from static RGB background, not an independent "
            "metric 3D camera-pose accuracy measurement."
        ),
    }


def _draw_bar(
    image: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    value: float,
    maximum: float,
    color: tuple[int, int, int],
) -> None:
    cv2.rectangle(image, (x, y), (x + width, y + 22), (41, 55, 73), -1)
    fill = int(round(width * min(max(value / maximum, 0.0), 1.0)))
    cv2.rectangle(image, (x, y), (x + fill, y + 22), color, -1)


def write_summary_image(report: Mapping[str, Any], output_path: Path) -> None:
    image = np.full((900, 1600, 3), (17, 28, 43), dtype=np.uint8)
    white = (242, 246, 250)
    muted = (164, 181, 200)
    cyan = (230, 188, 72)
    amber = (73, 181, 245)
    green = (106, 210, 128)
    cv2.putText(
        image,
        "DEPTH CAMERA POSE AUDIT",
        (70, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.45,
        white,
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "fixed SE(3), raw-depth repeatability, cross-take evidence, D2C boundary",
        (72, 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        muted,
        1,
        cv2.LINE_AA,
    )

    repeatability = report["reference_00"]["temporal_repeatability"]
    cv2.putText(image, "00 DEPTH REPEATABILITY", (75, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.78, white, 2)
    cv2.putText(
        image,
        f"max rotation  {repeatability['max_rotation_delta_deg']:.3f} deg",
        (75, 265),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        green,
        2,
    )
    cv2.putText(
        image,
        f"max origin     {repeatability['max_origin_delta_mm']:.3f} mm",
        (75, 307),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        green,
        2,
    )

    rgb = report["cross_take_fixed_camera"]["rgb_background_registration"]
    cv2.putText(image, "RGB FIXED-VIEW CHECK", (575, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.78, white, 2)
    for row, take in enumerate(rgb["takes"]):
        summary = take["summary"]
        cv2.putText(
            image,
            f"Take {row + 1}: max {summary['translation_magnitude_px_max']:.3f}px / {summary['absolute_rotation_deg_max']:.4f}deg",
            (575, 263 + row * 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            green if take["acceptance_pass"] else amber,
            2,
        )

    d2c = report["depth_to_color"]
    cv2.putText(image, "D2C RIGIDITY", (1110, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.78, white, 2)
    cv2.putText(
        image,
        f"det  {d2c['determinant']:.6f}",
        (1110, 263),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        amber,
        2,
    )
    cv2.putText(
        image,
        f"orth err  {d2c['max_abs_orthogonality_error']:.6f}",
        (1110, 305),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        amber,
        2,
    )
    cv2.putText(
        image,
        "NOT A 6DoF POSE",
        (1110, 350),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.63,
        amber,
        2,
    )

    cv2.line(image, (70, 405), (1530, 405), (49, 66, 88), 2)
    cv2.putText(image, "CROSS-TAKE RAW-DEPTH TABLE DIAGNOSTIC", (75, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.82, white, 2)
    cv2.putText(
        image,
        "Plane result is ROI-sensitive; shown as a review signal, not a pose correction.",
        (75, 505),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        muted,
        1,
    )
    planes = report["cross_take_fixed_camera"]["raw_depth_table_plane"]
    for row, take in enumerate(planes["takes"]):
        y = 560 + row * 82
        angle = float(take["relative_to_00"]["plane_normal_angle_deg"])
        delta = abs(float(take["relative_to_00"]["plane_d_delta_mm"]))
        cv2.putText(image, f"Take {row + 1}", (75, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.63, white, 2)
        _draw_bar(image, x=220, y=y, width=430, value=angle, maximum=1.5, color=amber)
        cv2.putText(image, f"{angle:.3f} deg", (670, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.61, amber, 2)
        _draw_bar(image, x=880, y=y, width=430, value=delta, maximum=12.0, color=cyan)
        cv2.putText(image, f"{delta:.2f} mm", (1330, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.61, cyan, 2)

    cv2.putText(
        image,
        "DECISION: use delivered depth SE(3); fixed camera supported by RGB background; keep color pose in REVIEW.",
        (75, 845),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.67,
        white,
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(output_path), image):
        raise IOError(f"Could not write {output_path}")


def _discover_recording(dataset_root: Path, prefix: str) -> Path:
    matches = sorted(
        path for path in dataset_root.iterdir() if path.is_dir() and path.name.startswith(prefix)
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one recording with prefix {prefix!r}, found {len(matches)}")
    return matches[0]


def run_audit(
    *,
    dataset_root: Path,
    calibration_path: Path,
    config_path: Path,
    output_dir: Path,
    formal_sample_count: int,
    monte_carlo_trials: int,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    calibration_path = Path(calibration_path).resolve()
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))

    reference_dir = _discover_recording(dataset_root, "00_")
    reference_bag = reference_dir / "原始BAG与内参" / "camera_1_rgb_depth.bag"
    reference_refs, _ = bagio._ros1_topic_message_refs(reference_bag, DEPTH_TOPIC)
    reference_frames, reference_count = read_depth_frames(
        reference_bag, range(len(reference_refs))
    )
    if reference_count != 109:
        raise ValueError(f"Expected 109 reference depth frames, found {reference_count}")
    reference_stack = np.stack([frame.image_mm for frame in reference_frames])
    full_fit = fit_depth_pose(
        temporal_median_mm(reference_stack), calibration, config
    )
    delivered_depth_to_world = np.asarray(
        calibration["transforms"]["depth_camera_to_world"], dtype=np.float64
    )
    delivered_origin = np.asarray(
        calibration["world_origin_in_depth_camera_mm"], dtype=np.float64
    )
    reproduction = {
        "depth_to_world_max_abs_matrix_delta": float(
            np.max(np.abs(full_fit.depth_to_world - delivered_depth_to_world))
        ),
        "world_origin_in_depth_max_abs_delta_mm": float(
            np.max(np.abs(full_fit.world_origin_in_depth_mm - delivered_origin))
        ),
    }
    reproduction["pass_at_1e-9"] = bool(
        reproduction["depth_to_world_max_abs_matrix_delta"] <= 1e-9
        and reproduction["world_origin_in_depth_max_abs_delta_mm"] <= 1e-9
    )

    window_results: list[dict[str, Any]] = []
    for name, start, stop in REFERENCE_WINDOWS:
        fit = fit_depth_pose(
            temporal_median_mm(reference_stack[start:stop]), calibration, config
        )
        window_results.append(
            {
                "window": name,
                "frame_first": start,
                "frame_stop_exclusive": stop,
                "frame_count": stop - start,
                "relative_to_full_109": _pose_delta(fit, full_fit),
                "table_plane": _plane_summary(fit.plane),
            }
        )
    max_rotation = max(
        item["relative_to_full_109"]["depth_to_world_rotation_delta_deg"]
        for item in window_results
    )
    max_origin = max(
        item["relative_to_full_109"]["world_origin_in_depth_delta_mm"]
        for item in window_results
    )
    repeatability = {
        "windows": window_results,
        "max_rotation_delta_deg": max_rotation,
        "max_origin_delta_mm": max_origin,
        "thresholds": {"rotation_delta_deg_max": 0.1, "origin_delta_mm_max": 1.0},
        "pass": bool(max_rotation <= 0.1 and max_origin <= 1.0),
        "scope_limit": "Conditional repeatability using the same manual marker pixels, not independent accuracy.",
    }

    plane_takes: list[dict[str, Any]] = []
    for key in ("01_", "02_", "03_"):
        recording = _discover_recording(dataset_root, key)
        bag_path = recording / "原始BAG与内参" / "camera_1_rgb_depth.bag"
        refs, _ = bagio._ros1_topic_message_refs(bag_path, DEPTH_TOPIC)
        indices = np.unique(
            np.rint(
                np.linspace(0, len(refs) - 1, min(formal_sample_count, len(refs)))
            ).astype(np.int64)
        )
        frames, message_count = read_depth_frames(bag_path, indices.tolist())
        median = temporal_median_mm(frames)
        fit = fit_depth_pose(median, calibration, config)
        roi_variants: list[dict[str, Any]] = []
        for v_min in (380, 420):
            variant = fit_depth_pose(
                median,
                calibration,
                config,
                v_min=v_min,
            )
            roi_variants.append(
                {
                    "v_strictly_greater_than": v_min,
                    "relative_to_00": _pose_delta(variant, full_fit),
                }
            )
        plane_takes.append(
            {
                "take": recording.name,
                "depth_message_count": message_count,
                "sample_indices": indices,
                "sample_count": int(len(indices)),
                "table_plane": _plane_summary(fit.plane),
                "relative_to_00": _pose_delta(fit, full_fit),
                "roi_sensitivity_variants": roi_variants,
            }
        )
    cross_take_plane_pass = all(
        item["relative_to_00"]["plane_normal_angle_deg"] <= 0.5
        and abs(item["relative_to_00"]["plane_d_delta_mm"]) <= 5.0
        for item in plane_takes
    )
    plane_audit = {
        "method": (
            f"Uniform {formal_sample_count}-frame raw-depth temporal median per Take; "
            "same RANSAC and original ROI as 00."
        ),
        "thresholds": {"normal_angle_deg_max": 0.5, "absolute_plane_d_delta_mm_max": 5.0},
        "takes": plane_takes,
        "acceptance_pass": cross_take_plane_pass,
        "decision": "review" if not cross_take_plane_pass else "pass",
        "interpretation": (
            "The formal Takes agree with one another but differ from 00 and the result "
            "changes with table ROI. This is a depth-systematic/scene diagnostic, not "
            "sufficient evidence to estimate a per-Take pose correction."
        ),
    }

    rgb_audit = rgb_fixed_camera_audit(dataset_root, output_dir)
    d2c_result, d2c_candidate = d2c_audit(calibration)

    depth_rotation = delivered_depth_to_world[:3, :3]
    depth_se3 = {
        "depth_camera_to_world": delivered_depth_to_world,
        "world_to_depth_camera": np.asarray(
            calibration["transforms"]["world_to_depth_camera"], dtype=np.float64
        ),
        "depth_camera_origin_in_world_mm": delivered_depth_to_world[:3, 3],
        "rotation_determinant": float(np.linalg.det(depth_rotation)),
        "max_abs_rotation_orthogonality_error": float(
            np.max(np.abs(depth_rotation.T @ depth_rotation - np.eye(3)))
        ),
        "inverse_composition_max_abs_error": float(
            np.max(
                np.abs(
                    delivered_depth_to_world
                    @ np.asarray(
                        calibration["transforms"]["world_to_depth_camera"],
                        dtype=np.float64,
                    )
                    - np.eye(4)
                )
            )
        ),
    }
    depth_se3["is_valid_se3_at_1e-9"] = bool(
        abs(depth_se3["rotation_determinant"] - 1.0) <= 1e-9
        and depth_se3["max_abs_rotation_orthogonality_error"] <= 1e-9
        and depth_se3["inverse_composition_max_abs_error"] <= 1e-9
    )

    report: dict[str, Any] = {
        "schema": "gt_calib.depth_camera_pose_audit.v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "depth_se3_reproducible_fixed_view_supported_color_pose_review",
        "inputs": {
            "dataset_root": str(dataset_root),
            "calibration": str(calibration_path),
            "manual_marker_config": str(config_path),
            "metric_depth_source": "raw BAG mono16 millimetres",
            "depth_preview_used_for_geometry": False,
        },
        "camera_pose_inventory": {
            "available_pose": "one fixed depth-camera-to-MOCAP-world extrinsic",
            "per_frame_camera_pose_or_trajectory_available": False,
            "bag_pose_tf_odom_topics_available": False,
            "fixed_camera_assumption": True,
        },
        "depth_camera_pose": depth_se3,
        "reference_00": {
            "raw_depth_frame_count": reference_count,
            "delivered_transform_exact_reproduction": reproduction,
            "full_table_plane": _plane_summary(full_fit.plane),
            "temporal_repeatability": repeatability,
            "manual_marker_sensitivity": marker_click_sensitivity(
                full_fit,
                calibration,
                config,
                trials=monte_carlo_trials,
            ),
            "identifiability": {
                "table_plane": "roll, pitch, and translation along table normal",
                "four_marker_rays_plus_45mm_height": "yaw and in-plane x/y translation",
                "marker_holdout_count": 0,
                "axis_points_per_arm": 2,
            },
        },
        "cross_take_fixed_camera": {
            "rgb_background_registration": rgb_audit,
            "raw_depth_table_plane": plane_audit,
            "decision": (
                "The physical fixed-view assumption is strongly supported by RGB static "
                "background registration. Do not apply a table-only per-Take pose correction; "
                "the depth-plane shift is spatially/ROI sensitive and remains under review."
            ),
        },
        "depth_to_color": d2c_result,
        "recommended_use": {
            "depth_space": "Use the delivered rigid depth_camera_to_world transform.",
            "rgb_projection": (
                "Use the delivered D2C affine only as a review overlay baseline; do not call "
                "the resulting color transform a verified 6DoF camera pose."
            ),
            "absolute_gt": (
                "Not yet accepted. Independent known 3D/2D correspondences or a tracked "
                "camera rigid body are still required."
            ),
            "do_not_do": [
                "Do not infer millimetres from Depth.mp4.",
                "Do not silently replace the delivered D2C block with its polar rotation.",
                "Do not fit the camera pose to 01-03 hand depth and then report those Takes as holdout accuracy.",
            ],
        },
        "artifacts": {
            "audit_json": str((output_dir / "depth_camera_pose_audit.json").relative_to(PROJECT_ROOT)),
            "summary_image": str((output_dir / "depth_camera_pose_audit_summary.png").relative_to(PROJECT_ROOT)),
            "rgb_fixed_camera_image": rgb_audit["diagnostic_image"],
            "d2c_rigid_candidate": str((output_dir / "d2c_rigid_candidate.json").relative_to(PROJECT_ROOT)),
        },
    }

    audit_path = output_dir / "depth_camera_pose_audit.json"
    candidate_path = output_dir / "d2c_rigid_candidate.json"
    audit_path.write_text(
        json.dumps(_json_ready(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(_json_ready(d2c_candidate), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary_image(report, output_dir / "depth_camera_pose_audit_summary.png")
    return _json_ready(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the fixed raw-depth camera/MOCAP calibration."
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--formal-sample-count", type=int, default=61)
    parser.add_argument("--monte-carlo-trials", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.formal_sample_count < 9:
        raise ValueError("--formal-sample-count must be at least 9")
    if args.monte_carlo_trials < 100:
        raise ValueError("--monte-carlo-trials must be at least 100")
    report = run_audit(
        dataset_root=args.dataset_root,
        calibration_path=args.calibration,
        config_path=args.config,
        output_dir=args.output_dir,
        formal_sample_count=args.formal_sample_count,
        monte_carlo_trials=args.monte_carlo_trials,
    )
    repeatability = report["reference_00"]["temporal_repeatability"]
    rgb = report["cross_take_fixed_camera"]["rgb_background_registration"]
    print(f"Audit written to {args.output_dir.resolve()}")
    print(
        "00 repeatability max rotation/origin: "
        f"{repeatability['max_rotation_delta_deg']:.3f} deg / "
        f"{repeatability['max_origin_delta_mm']:.3f} mm"
    )
    print(f"RGB fixed-view registration passed: {rgb['all_takes_pass']}")
    print(f"D2C is a valid rotation: {report['depth_to_color']['is_valid_rotation_at_1e-6']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
