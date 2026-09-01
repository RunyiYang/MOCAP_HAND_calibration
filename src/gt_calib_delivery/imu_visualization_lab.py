from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

import gt_calib_viz as viz
import mocap_video_overlay as mocap_overlay

from .delivery import OLD_TAKES
from .imu_mocap_comparison import (
    DEFAULT_MANUAL_PROFILE,
    OLD_REGISTRATION_PROFILE,
    FixedSimilarity,
    _assert_tracked_bvh_registration,
    _contact_sheet,
    _distribution,
    _faststart,
    _fit_fixed_similarity,
    _probe_video,
    _sha256,
    _source_asset,
    _transcode_h264,
    write_checksums,
)
from .manual_profiles import FINAL_NINE_PROFILE_SCHEMA, load_final_nine_profile
from .new_capture import (
    ANATOMICAL_SIDES,
    BASE_MARKER_INDICES,
    DEFAULT_REAR_OFFSET_MM,
    GLOVE_CHAINS,
    MARKER_TRACKS,
    MARKER_TO_GLOVE_INDICES,
    RAW_CHAINS,
    TIP_MARKER_INDICES,
    load_new_capture,
    load_solved_pose,
    project_world,
    raw_mocap_nodes,
    virtual_wrist,
)


DELIVERY_SCHEMA = "gt_calib.imu_mocap_visualization_lab.v1"
METRICS_SCHEMA = "gt_calib.imu_mocap_visualization_method_metrics.v1"
MOTION_SCHEMA = "gt_calib.imu_mocap_visualization_motion.v1"
VALIDATION_SCHEMA = "gt_calib.imu_mocap_visualization_lab_validation.v1"
EXPECTED_TAKES = 4
EXPECTED_METHODS = 7
EXPECTED_VIDEOS = EXPECTED_TAKES * EXPECTED_METHODS
EXPECTED_TAKE_IDS = ("take005", "take006a", "take006b", "take007")
EXPECTED_SEGMENTS = (
    (
        "take005",
        "camera_glove_recording_20260831_161402/Take_005",
        0,
        1867,
    ),
    (
        "take006a",
        "camera_glove_recording_20260831_161610/Take_006",
        0,
        1309,
    ),
    (
        "take006b",
        "camera_glove_recording_20260831_161610/Take_006",
        1309,
        1310,
    ),
    (
        "take007",
        "camera_glove_recording_20260831_161912/Take_007",
        0,
        1981,
    ),
)
EXPECTED_CAMERA_INTRINSICS = {
    "take005": {
        "path": (
            "thor_new4_20260831_processed/"
            "camera_glove_recording_20260831_161402/"
            "rgbd_unpack/camera_1_intrinsics.json"
        ),
        "bytes": 2814,
        "sha256": "05090e1bd9d3f66a29010de6a43afddc99ac9ab07873761fb0c6b295874a642b",
    },
    "take006a": {
        "path": (
            "thor_new4_20260831_processed/"
            "camera_glove_recording_20260831_161610/"
            "rgbd_unpack/camera_1_intrinsics.json"
        ),
        "bytes": 2814,
        "sha256": "8d650632f8f0de8460a4bade8766329b6924d76a2e89481b149b833e8d502d1c",
    },
    "take006b": {
        "path": (
            "thor_new4_20260831_processed/"
            "camera_glove_recording_20260831_161610/"
            "rgbd_unpack/camera_1_intrinsics.json"
        ),
        "bytes": 2814,
        "sha256": "8d650632f8f0de8460a4bade8766329b6924d76a2e89481b149b833e8d502d1c",
    },
    "take007": {
        "path": (
            "thor_new4_20260831_processed/"
            "camera_glove_recording_20260831_161912/"
            "rgbd_unpack/camera_1_intrinsics.json"
        ),
        "bytes": 2814,
        "sha256": "7298ba3d82904c6d6b386b8fd58a3e2f773ead4f388702adce06161ef81287b8",
    },
}
MAX_CLOUDFLARE_ASSET_BYTES = 25 * 1024 * 1024

TAKE007_BASE_MARKER_INDICES = np.asarray((3, 5, 7, 9), dtype=np.int32)
TAKE007_BASE_GLOVE_INDICES = np.asarray((4, 8, 12, 16), dtype=np.int32)
CMM_CALIBRATION_FRACTION = 0.20

# The user-supplied Take_005/006/007 marker-ID table and placement photographs
# define the #1..#11 topology for all three CMM captures. Take_007 left marker
# #1 is the one exception: its 11781 -> 12503 alias is confirmed by trajectory
# continuity across the two-frame re-identification gap. The fitted palm
# residuals are recorded in every metrics file.
CMM_MARKER_TRACKS = {
    "Take_005": {
        "left": (
            ("11781",), ("11777",), ("11773",), ("11780",), ("11772",),
            ("11779",), ("11609",), ("11782",), ("11787",), ("11786",),
            ("11785",),
        ),
        "right": (
            ("11769",), ("11622",), ("11619",), ("11614",), ("11616",),
            ("11678",), ("11625",), ("11685",), ("11627",), ("11630",),
            ("11629",),
        ),
    },
    "Take_006": {
        "left": (("11781",),) + tuple(MARKER_TRACKS["left"][1:]),
        "right": (
            ("12058",), ("12069",), ("11619",), ("11614",), ("11616",),
            ("11678",), ("11625",), ("11685",), ("11627",), ("11630",),
            ("11629",),
        ),
    },
    "Take_007": MARKER_TRACKS,
}

CMM_CAPTURE_SPECS = (
    ("camera_glove_recording_20260831_161402", "Take_005"),
    ("camera_glove_recording_20260831_161610", "Take_006"),
    ("camera_glove_recording_20260831_161912", "Take_007"),
)


def _expected_dataset_scope() -> dict[str, Any]:
    return {
        "root": "thor_new4_20260831_processed",
        "policy": "new dataset only",
        "published_segments": [
            {
                "take_id": take_id,
                "source": source,
                "source_first": source_first,
                "frame_count": frame_count,
            }
            for take_id, source, source_first, frame_count in EXPECTED_SEGMENTS
        ],
        "take006_split": {
            "reason": (
                "publish four action-review segments from three complete "
                "synchronized action recordings"
            ),
            "windows_are_disjoint": True,
            "source_output_frames": 2619,
            "split_output_index": 1309,
        },
    }


def _expected_motion_layers() -> list[dict[str, Any]]:
    layers = (
        ("mocap-left", "mocap", "left", 0, 11, RAW_CHAINS),
        ("mocap-right", "mocap", "right", 11, 11, RAW_CHAINS),
        ("imu-left", "result", "left", 22, 20, GLOVE_CHAINS),
        ("imu-right", "result", "right", 42, 20, GLOVE_CHAINS),
    )
    return [
        {
            "id": layer_id,
            "kind": kind,
            "side": side,
            "joint_offset": offset,
            "joint_count": count,
            "chains": [list(chain) for chain in chains],
        }
        for layer_id, kind, side, offset, count, chains in layers
    ]

PARENTS = np.asarray(
    (-1, 0, 1, 2, 0, 4, 5, 6, 0, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18),
    dtype=np.int32,
)
CHILDREN = np.arange(1, 20, dtype=np.int32)
PARENT_FOR_CHILD = PARENTS[CHILDREN]
OLD_PRIMARY_INDICES = np.asarray(tuple(range(4, 20)), dtype=np.int32)

MOCAP_COLORS = {"left": (255, 220, 45), "right": (245, 70, 230)}
IMU_COLORS = {"left": (75, 245, 105), "right": (45, 210, 255)}
HALO_BGR = (5, 10, 18)


@dataclass(frozen=True)
class MethodSpec:
    id: str
    short_label: str
    label: str
    supervision: str
    description: str
    boundary: str


METHODS = (
    MethodSpec(
        "s2_continuous",
        "S2 continuous",
        "IMU-articulation S2 continuous interpolation",
        "imu_only",
        "True-timestamp spherical bone-direction interpolation across every solver bracket.",
        "No per-frame MOCAP finger target; IMU articulation is shown after MOCAP-derived spatial calibration.",
    ),
    MethodSpec(
        "s2_gaussian",
        "S2 Gaussian",
        "IMU-articulation zero-phase S2 Gaussian",
        "imu_only",
        "Symmetric 9-tap spherical smoothing (sigma 1.8 frames) with fixed median bone lengths.",
        "No per-frame MOCAP finger target; offline smoother after MOCAP-derived spatial calibration.",
    ),
    MethodSpec(
        "kalman_rts",
        "Kalman RTS",
        "IMU-articulation constant-velocity Kalman RTS",
        "imu_only",
        "Forward Kalman filtering plus backward Rauch-Tung-Striebel smoothing and kinematic projection.",
        "No MOCAP finger joint is a per-frame observation; spatial calibration is MOCAP-derived.",
    ),
    MethodSpec(
        "posterior_75",
        "75% posterior",
        "75% MOCAP-guided manifold posterior",
        "mocap_conditioned",
        "MOCAP bone direction plus 25% of the smoothed IMU tangent-space residual.",
        "Uses synchronized same-frame MOCAP and future frames; visualization only.",
    ),
    MethodSpec(
        "posterior_98",
        "98% posterior",
        "98% MOCAP-guided manifold posterior",
        "mocap_conditioned",
        "MOCAP bone direction plus 2% of the smoothed IMU tangent-space residual.",
        "Recommended visual posterior; not an independent IMU accuracy estimate.",
    ),
    MethodSpec(
        "rbf_self_fit",
        "RBF self-fit",
        "Same-sequence RBF neural IMU-to-MOCAP fit",
        "same_sequence_teacher_fit",
        "Full-sequence RBF network maps IMU pose and velocity to the same sequence's MOCAP target.",
        "No holdout by explicit request; demonstrates memorization/visual fit, not generalization.",
    ),
    MethodSpec(
        "guided_ik",
        "Anatomical IK",
        "MOCAP-guided anatomical constant-curvature IK upper bound",
        "mocap_conditioned_oracle",
        (
            "CMM endpoint-constrained fixed-phalanx IK with one continuous "
            "bend plane and bounded PIP/DIP/thumb flexion."
        ),
        (
            "Recommended visual oracle/upper bound; CMM base/tip surface "
            "markers are not anatomical joint centers or IMU-only output."
        ),
    ),
)
METHOD_BY_ID = {method.id: method for method in METHODS}
EXPECTED_METHOD_IDS = tuple(method.id for method in METHODS)


@dataclass
class LabTake:
    order: int
    id: str
    label: str
    source: str
    source_video: Path
    source_first: int
    frame_count: int
    fps: float
    source_width: int
    source_height: int
    projection_kind: str
    calibration: Any
    mocap_draw: dict[str, np.ndarray]
    mocap_chains: Sequence[Sequence[int]]
    raw_imu: dict[str, np.ndarray]
    target_pose: dict[str, np.ndarray]
    strict_valid: dict[str, np.ndarray]
    marker_targets: dict[str, np.ndarray] | None
    source_assets: dict[str, Any]
    preparation: dict[str, Any]


def _gaussian_kernel(sigma: float = 1.8, radius: int = 4) -> np.ndarray:
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * np.square(offsets / float(sigma)))
    return weights / np.sum(weights)


def _smooth(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    radius = len(kernel) // 2
    if len(data) <= radius:
        raise ValueError("Sequence is too short for symmetric smoothing")
    padded = np.pad(
        data,
        ((radius, radius),) + ((0, 0),) * (data.ndim - 1),
        mode="reflect",
    )
    output = np.zeros_like(data)
    for offset, weight in enumerate(np.asarray(kernel, dtype=np.float64)):
        output += float(weight) * padded[offset : offset + len(data)]
    return output


def _normalize(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(vectors, dtype=np.float64)
    lengths = np.linalg.norm(values, axis=-1)
    directions = np.divide(
        values,
        lengths[..., None],
        out=np.zeros_like(values),
        where=lengths[..., None] > 1e-9,
    )
    return directions, lengths


def _bone_vectors(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(pose, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (20, 3):
        raise ValueError("A hand pose must have shape [frame, 20, xyz]")
    vectors = points[:, CHILDREN] - points[:, PARENT_FOR_CHILD]
    directions, lengths = _normalize(vectors)
    if np.any(lengths <= 1e-6) or np.any(~np.isfinite(points)):
        raise ValueError("Hand pose contains a non-finite or zero-length bone")
    return vectors, directions, lengths


def _reconstruct_pose(
    root: np.ndarray,
    directions: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    root_values = np.asarray(root, dtype=np.float64)
    unit = np.asarray(directions, dtype=np.float64)
    bone_lengths = np.asarray(lengths, dtype=np.float64)
    if unit.shape != (len(root_values), 19, 3):
        raise ValueError("Directions must have shape [frame, 19, xyz]")
    if bone_lengths.shape not in ((19,), (len(root_values), 19)):
        raise ValueError("Bone lengths must have shape [19] or [frame, 19]")
    if bone_lengths.ndim == 1:
        bone_lengths = np.broadcast_to(bone_lengths, (len(root_values), 19))
    output = np.empty((len(root_values), 20, 3), dtype=np.float64)
    output[:, 0] = root_values
    for bone_index, child in enumerate(CHILDREN):
        output[:, child] = (
            output[:, PARENTS[child]]
            + unit[:, bone_index] * bone_lengths[:, bone_index, None]
        )
    return output


def _slerp_unit(first: np.ndarray, second: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    amount = np.asarray(alpha, dtype=np.float64)
    dot = np.clip(np.sum(left * right, axis=-1), -1.0, 1.0)
    theta = np.arccos(dot)
    sine = np.sin(theta)
    same = dot > 1.0 - 1e-8
    antipodal = dot < -1.0 + 1e-8
    safe = np.where(same | antipodal, 1.0, sine)
    a = amount
    while a.ndim < theta.ndim:
        a = a[..., None]
    weights_left = np.sin((1.0 - a) * theta) / safe
    weights_right = np.sin(a * theta) / safe
    result = weights_left[..., None] * left + weights_right[..., None] * right
    linear_result = (1.0 - a[..., None]) * left + a[..., None] * right
    basis = np.eye(3, dtype=np.float64)[np.argmin(np.abs(left), axis=-1)]
    perpendicular, _ = _normalize(np.cross(left, basis))
    antipodal_result = (
        np.cos(np.pi * a)[..., None] * left
        + np.sin(np.pi * a)[..., None] * perpendicular
    )
    result = np.where(same[..., None], linear_result, result)
    result = np.where(antipodal[..., None], antipodal_result, result)
    normalized, lengths = _normalize(result)
    if np.any(lengths <= 1e-9):
        raise ValueError("Spherical interpolation produced a zero direction")
    return normalized


def manifold_resample_pose(
    source_times_s: np.ndarray,
    source_pose: np.ndarray,
    target_times_s: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Resample a 20-joint pose with S2 bone interpolation on every bracket."""

    source_times = np.asarray(source_times_s, dtype=np.float64)
    target_times = np.asarray(target_times_s, dtype=np.float64)
    pose = np.asarray(source_pose, dtype=np.float64)
    if len(source_times) != len(pose) or np.any(np.diff(source_times) <= 0.0):
        raise ValueError("Source pose timestamps are malformed")
    if target_times[0] < source_times[0] or target_times[-1] > source_times[-1]:
        raise ValueError("Target interval extends beyond solver support")
    _, directions, lengths = _bone_vectors(pose)
    insertion = np.searchsorted(source_times, target_times, side="left")
    high = np.clip(insertion, 0, len(source_times) - 1)
    low = np.clip(high - 1, 0, len(source_times) - 1)
    exact = source_times[high] == target_times
    low[exact] = high[exact]
    span = source_times[high] - source_times[low]
    alpha = np.divide(
        target_times - source_times[low],
        span,
        out=np.zeros_like(target_times),
        where=span > 0.0,
    )
    sampled_directions = _slerp_unit(directions[low], directions[high], alpha)
    sampled_lengths = (
        lengths[low] * (1.0 - alpha[:, None])
        + lengths[high] * alpha[:, None]
    )
    root = pose[low, 0] * (1.0 - alpha[:, None]) + pose[high, 0] * alpha[:, None]
    sampled = _reconstruct_pose(root, sampled_directions, sampled_lengths)
    nearest_age = np.minimum(
        np.abs(target_times - source_times[low]),
        np.abs(source_times[high] - target_times),
    )
    return sampled, {
        "source_rows": int(len(source_times)),
        "source_rate_hz_median": float(1.0 / np.median(np.diff(source_times))),
        "maximum_source_bracket_ms": float(np.max(span) * 1000.0),
        "nearest_sample_age_median_ms": float(np.median(nearest_age) * 1000.0),
        "nearest_sample_age_p95_ms": float(np.percentile(nearest_age, 95) * 1000.0),
        "finite_output_frames": int(np.count_nonzero(np.all(np.isfinite(sampled), axis=(1, 2)))),
    }


def smooth_pose_s2(pose: np.ndarray, *, sigma: float = 1.8, radius: int = 4) -> np.ndarray:
    points = np.asarray(pose, dtype=np.float64)
    _, directions, lengths = _bone_vectors(points)
    smooth_directions, norms = _normalize(_smooth(directions, _gaussian_kernel(sigma, radius)))
    if np.any(norms <= 1e-9):
        raise ValueError("S2 smoothing produced a zero direction")
    fixed_lengths = np.median(lengths, axis=0)
    return _reconstruct_pose(points[:, 0], smooth_directions, fixed_lengths)


def _rts_smooth_coordinates(
    measurements: np.ndarray,
    *,
    fps: float,
    measurement_sigma_mm: float = 4.0,
    acceleration_sigma_mm_s2: float = 900.0,
) -> np.ndarray:
    values = np.asarray(measurements, dtype=np.float64)
    count, dimension = values.shape
    dt = 1.0 / float(fps)
    transition = np.asarray(((1.0, dt), (0.0, 1.0)), dtype=np.float64)
    process_basis = np.asarray((0.5 * dt * dt, dt), dtype=np.float64)
    process = np.outer(process_basis, process_basis) * acceleration_sigma_mm_s2**2
    observation_variance = measurement_sigma_mm**2

    filtered = np.empty((count, dimension, 2), dtype=np.float64)
    predicted = np.empty_like(filtered)
    filtered_covariance = np.empty((count, 2, 2), dtype=np.float64)
    predicted_covariance = np.empty_like(filtered_covariance)
    filtered[0, :, 0] = values[0]
    filtered[0, :, 1] = 0.0
    filtered_covariance[0] = np.diag((observation_variance, 1e4))
    predicted[0] = filtered[0]
    predicted_covariance[0] = filtered_covariance[0]
    identity = np.eye(2, dtype=np.float64)
    for frame in range(1, count):
        predicted[frame] = filtered[frame - 1] @ transition.T
        predicted_covariance[frame] = (
            transition @ filtered_covariance[frame - 1] @ transition.T + process
        )
        covariance = predicted_covariance[frame]
        gain = covariance[:, 0] / (covariance[0, 0] + observation_variance)
        residual = values[frame] - predicted[frame, :, 0]
        filtered[frame] = predicted[frame] + residual[:, None] * gain[None, :]
        filtered_covariance[frame] = (identity - np.outer(gain, (1.0, 0.0))) @ covariance

    smoothed = filtered.copy()
    for frame in range(count - 2, -1, -1):
        smoother_gain = (
            filtered_covariance[frame]
            @ transition.T
            @ np.linalg.inv(predicted_covariance[frame + 1])
        )
        smoothed[frame] += (
            smoothed[frame + 1] - predicted[frame + 1]
        ) @ smoother_gain.T
    return smoothed[:, :, 0]


def kalman_rts_pose(pose: np.ndarray, *, fps: float) -> np.ndarray:
    points = np.asarray(pose, dtype=np.float64)
    relative = points - points[:, 0, None, :]
    smoothed = _rts_smooth_coordinates(relative[:, 1:].reshape(len(points), -1), fps=fps)
    smoothed_points = np.concatenate(
        (np.zeros((len(points), 1, 3), dtype=np.float64), smoothed.reshape(len(points), 19, 3)),
        axis=1,
    )
    _, directions, _ = _bone_vectors(smoothed_points)
    _, _, raw_lengths = _bone_vectors(points)
    return _reconstruct_pose(points[:, 0], directions, np.median(raw_lengths, axis=0))


def _s2_log(base: np.ndarray, endpoint: np.ndarray) -> np.ndarray:
    first = np.asarray(base, dtype=np.float64)
    second = np.asarray(endpoint, dtype=np.float64)
    dot = np.clip(np.sum(first * second, axis=-1), -1.0, 1.0)
    theta = np.arccos(dot)
    tangent = second - dot[..., None] * first
    tangent_direction, tangent_length = _normalize(tangent)
    antipodal = dot < -1.0 + 1e-8
    basis = np.eye(3, dtype=np.float64)[np.argmin(np.abs(first), axis=-1)]
    perpendicular, _ = _normalize(np.cross(first, basis))
    tangent_direction = np.where(
        antipodal[..., None],
        perpendicular,
        tangent_direction,
    )
    tangent_length = np.where(antipodal, 1.0, tangent_length)
    return tangent_direction * np.where(tangent_length > 1e-9, theta, 0.0)[..., None]


def _s2_exp(base: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    first = np.asarray(base, dtype=np.float64)
    delta = np.asarray(tangent, dtype=np.float64)
    direction, theta = _normalize(delta)
    result = np.cos(theta)[..., None] * first + np.sin(theta)[..., None] * direction
    result = np.where((theta <= 1e-9)[..., None], first, result)
    normalized, _ = _normalize(result)
    return normalized


def mocap_guided_posterior(
    raw_pose: np.ndarray,
    target_pose: np.ndarray,
    *,
    imu_residual_gain: float,
    sigma: float = 1.8,
    radius: int = 4,
) -> np.ndarray:
    raw = np.asarray(raw_pose, dtype=np.float64)
    target = np.asarray(target_pose, dtype=np.float64)
    _, raw_directions, _ = _bone_vectors(raw)
    _, target_directions, target_lengths = _bone_vectors(target)
    residual = _s2_log(target_directions, raw_directions)
    residual = _smooth(residual, _gaussian_kernel(sigma, radius))
    # Smoothing combines samples whose tangent spaces differ. Reproject the
    # result into the current target direction's tangent plane before exp().
    residual -= (
        np.sum(residual * target_directions, axis=-1, keepdims=True)
        * target_directions
    )
    residual_direction, residual_angle = _normalize(residual)
    clipped_angle = np.minimum(residual_angle, np.deg2rad(60.0))
    clipped = residual_direction * clipped_angle[..., None]
    fused_directions = _s2_exp(
        target_directions,
        float(imu_residual_gain) * clipped,
    )
    return _reconstruct_pose(target[:, 0], fused_directions, target_lengths)


def _rbf_features(pose: np.ndarray, *, fps: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    relative = np.asarray(pose, dtype=np.float64)[:, 1:] - pose[:, 0, None, :]
    position = relative.reshape(len(relative), -1)
    velocity = np.gradient(position, 1.0 / float(fps), axis=0)
    raw = np.concatenate((position, velocity), axis=1)
    mean = np.mean(raw, axis=0)
    scale = np.std(raw, axis=0)
    active = scale > 1e-9
    standardized = (raw[:, active] - mean[active]) / scale[active]
    standardized /= math.sqrt(max(1, standardized.shape[1]))
    return standardized, {"mean": mean, "scale": scale, "active": active}


def rbf_same_sequence_fit(
    raw_pose: np.ndarray,
    target_pose: np.ndarray,
    *,
    fps: float,
    ridge: float = 1e-8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit and evaluate a full-sequence RBF neural mapper with no holdout."""

    raw = np.asarray(raw_pose, dtype=np.float64)
    target = np.asarray(target_pose, dtype=np.float64)
    features, feature_stats = _rbf_features(raw, fps=fps)
    squared_norm = np.sum(np.square(features), axis=1)
    distance_squared = np.maximum(
        squared_norm[:, None] + squared_norm[None, :] - 2.0 * features @ features.T,
        0.0,
    )
    nearest_squared = np.partition(distance_squared, 1, axis=1)[:, 1]
    sigma = float(np.median(np.sqrt(nearest_squared[nearest_squared > 1e-12])))
    if not np.isfinite(sigma) or sigma <= 1e-6:
        raise ValueError("RBF self-fit could not estimate a positive kernel width")
    kernel = np.exp(-distance_squared / (2.0 * sigma * sigma))
    target_relative = target[:, 1:] - target[:, 0, None, :]
    outputs = target_relative.reshape(len(target), -1)
    coefficients = np.linalg.solve(
        kernel + float(ridge) * np.eye(len(kernel), dtype=np.float64),
        outputs,
    )
    predicted_relative = (kernel @ coefficients).reshape(len(target), 19, 3)
    predicted = np.concatenate(
        (target[:, 0, None, :], target[:, 0, None, :] + predicted_relative),
        axis=1,
    )
    training_error = np.linalg.norm(predicted - target, axis=-1)
    return predicted, {
        "network": "Gaussian RBF neural network / kernel ridge output layer",
        "input": "root-relative IMU 19x3 position plus finite-difference velocity; no time index",
        "training_frames": int(len(raw)),
        "rbf_centers": int(len(raw)),
        "active_input_dimensions": int(np.count_nonzero(feature_stats["active"])),
        "sigma": sigma,
        "ridge": float(ridge),
        "test_frames": 0,
        "training_and_render_sequence_are_identical": True,
        "training_epe_mm": _distribution(training_error[:, 1:]),
    }


def _unit_perpendicular(axis: np.ndarray, guide: np.ndarray, fallback: np.ndarray | None) -> np.ndarray:
    direction = np.asarray(axis, dtype=np.float64)
    candidate = np.asarray(guide, dtype=np.float64)
    candidate = candidate - np.dot(candidate, direction) * direction
    norm = float(np.linalg.norm(candidate))
    if norm <= 1e-8 and fallback is not None:
        candidate = np.asarray(fallback, dtype=np.float64)
        candidate = candidate - np.dot(candidate, direction) * direction
        norm = float(np.linalg.norm(candidate))
    if norm <= 1e-8:
        basis = np.eye(3)[int(np.argmin(np.abs(direction)))]
        candidate = basis - np.dot(basis, direction) * direction
        norm = float(np.linalg.norm(candidate))
    candidate = candidate / norm
    if fallback is not None and float(np.dot(candidate, fallback)) < 0.0:
        candidate = -candidate
    return candidate


def _unit_bend_normal(
    vector: np.ndarray,
    endpoint_axis: np.ndarray,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """Return a unit bend normal transported onto the endpoint-axis tangent plane."""

    axis = np.asarray(endpoint_axis, dtype=np.float64)
    candidate = np.asarray(vector, dtype=np.float64)
    candidate = candidate - np.dot(candidate, axis) * axis
    norm = float(np.linalg.norm(candidate))
    if norm <= 1e-8 and fallback is not None:
        candidate = np.asarray(fallback, dtype=np.float64)
        candidate = candidate - np.dot(candidate, axis) * axis
        norm = float(np.linalg.norm(candidate))
    if norm <= 1e-8:
        basis = np.eye(3)[int(np.argmin(np.abs(axis)))]
        candidate = basis - np.dot(basis, axis) * axis
        norm = float(np.linalg.norm(candidate))
    return candidate / norm


def _previous_bend_normal(
    previous_normals: Sequence[np.ndarray | None],
    endpoint_axis: np.ndarray,
) -> np.ndarray | None:
    """Transport and average prior joint normals without allowing antipodal cancellation."""

    transported: list[np.ndarray] = []
    for value in previous_normals:
        if value is None:
            continue
        candidate = np.asarray(value, dtype=np.float64)
        candidate = candidate - np.dot(candidate, endpoint_axis) * endpoint_axis
        norm = float(np.linalg.norm(candidate))
        if norm <= 1e-8:
            continue
        candidate /= norm
        if transported and float(np.dot(candidate, transported[0])) < 0.0:
            candidate = -candidate
        transported.append(candidate)
    if not transported:
        return None
    combined = np.sum(transported, axis=0)
    return _unit_bend_normal(combined, endpoint_axis, transported[0])


def _chain_bend_normals(points: np.ndarray, fallback: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Measure the PIP and DIP bend normals, using the plane normal at singularities."""

    segments = np.diff(np.asarray(points, dtype=np.float64), axis=0)
    first = np.cross(segments[0], segments[1])
    second = np.cross(segments[1], segments[2])
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm > 1e-8:
        first /= first_norm
    if second_norm > 1e-8:
        second /= second_norm
    if first_norm <= 1e-8 and second_norm > 1e-8:
        first = second.copy()
    elif second_norm <= 1e-8 and first_norm > 1e-8:
        second = first.copy()
    elif first_norm <= 1e-8 and second_norm <= 1e-8:
        first = np.asarray(fallback, dtype=np.float64).copy()
        second = first.copy()
    return first, second


def _sphere_circle_point(
    first: np.ndarray,
    second: np.ndarray,
    first_radius: float,
    second_radius: float,
    guide: np.ndarray,
    previous_normal: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    origin = np.asarray(first, dtype=np.float64)
    endpoint = np.asarray(second, dtype=np.float64)
    delta = endpoint - origin
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-8:
        raise ValueError("IK sphere centers coincide")
    maximum = float(first_radius + second_radius)
    minimum = float(abs(first_radius - second_radius))
    tolerance = 1e-6
    if distance > maximum + tolerance or distance < minimum - tolerance:
        raise ValueError(
            "IK endpoint is infeasible for the declared phalanx lengths: "
            f"distance={distance:.9f}, feasible=[{minimum:.9f}, {maximum:.9f}]"
        )
    # Keep both requested endpoints exact. Clipping a center distance here would
    # move the effective tip while the caller still emitted the original tip.
    # Only the cosine terms below are clipped for floating-point round-off.
    axis = delta / distance
    along = (
        first_radius * first_radius
        - second_radius * second_radius
        + distance * distance
    ) / (2.0 * distance)
    height = math.sqrt(max(first_radius * first_radius - along * along, 0.0))
    center = origin + along * axis
    previous_bend = _previous_bend_normal((previous_normal,), axis)
    previous_offset = (
        None if previous_bend is None else np.cross(axis, previous_bend)
    )
    offset = _unit_perpendicular(
        axis,
        np.asarray(guide) - center,
        previous_offset,
    )
    middle = center + height * offset
    # For a point on the positive circle offset, cross(base->middle,
    # middle->tip) points along cross(offset, axis).
    bend = _unit_bend_normal(np.cross(offset, axis), axis, previous_bend)
    if previous_bend is not None and float(np.dot(bend, previous_bend)) < 0.0:
        # The offset helper already preserves this hemisphere; keep this guard
        # explicit because branch continuity is an IK invariant.
        offset = -offset
        middle = center + height * offset
        bend = -bend
    return middle, bend


def _guide_bend_normal(
    guide: np.ndarray,
    endpoint_axis: np.ndarray,
    previous: np.ndarray | None,
) -> np.ndarray:
    """Estimate a stable common flexion plane from an unconstrained guide chain."""

    points = np.asarray(guide, dtype=np.float64)
    segments = np.diff(points, axis=0)
    turns = (np.cross(segments[0], segments[1]), np.cross(segments[1], segments[2]))
    usable = [turn for turn in turns if float(np.linalg.norm(turn)) > 1e-8]
    if usable:
        # An S-shaped guide has antipodal local turns. Use the stronger local
        # observation instead of averaging the plane normal to zero.
        strongest = max(usable, key=lambda value: float(np.linalg.norm(value)))
        guide_normal = _unit_bend_normal(strongest, endpoint_axis, previous)
    else:
        chord_residuals = []
        base = points[0]
        for point in points[1:-1]:
            delta = point - base
            chord_residuals.append(
                delta - np.dot(delta, endpoint_axis) * endpoint_axis
            )
        residual = max(chord_residuals, key=lambda value: float(np.linalg.norm(value)))
        guide_normal = _unit_bend_normal(
            np.cross(residual, endpoint_axis),
            endpoint_axis,
            previous,
        )
    if previous is None:
        return guide_normal
    if float(np.dot(guide_normal, previous)) < 0.0:
        guide_normal = -guide_normal
    # Low-pass the flexion plane itself. The hard hemisphere check in the
    # candidate selector below handles branch identity; this blend suppresses
    # frame-to-frame plane wobble without freezing legitimate articulation.
    return _unit_bend_normal(0.8 * previous + 0.2 * guide_normal, endpoint_axis, previous)


IK_DIP_TO_PIP_RATIO = 0.65
IK_MAX_PIP_FLEXION_DEG = 110.0
IK_MAX_DIP_FLEXION_DEG = 75.0
IK_MAX_THUMB_FLEXION_DEG = 115.0
IK_ACTIVE_BEND_THRESHOLD_DEG = 5.0
IK_ANATOMICAL_LIMITS = {
    "pip_max_deg": IK_MAX_PIP_FLEXION_DEG,
    "dip_max_deg": IK_MAX_DIP_FLEXION_DEG,
    "thumb_max_deg": IK_MAX_THUMB_FLEXION_DEG,
    "endpoint_p95_max_mm": 0.7,
    "bend_plane_temporal_p95_max_deg": 8.0,
    "bend_plane_temporal_max_deg": 45.0,
}
IK_ANATOMICAL_ACCEPTANCE = {
    "no_opposite_active_bends": True,
    "pip_within_limit": True,
    "dip_within_limit": True,
    "thumb_within_limit": True,
    "fixed_bone_lengths": True,
    "endpoint_p95_within_0_7_mm": True,
    "bend_plane_temporal_p95_within_8_deg": True,
    "bend_plane_temporal_max_within_45_deg": True,
}
IK_ANATOMICAL_DISTRIBUTIONS = (
    "pip_flexion_deg",
    "dip_flexion_deg",
    "thumb_flexion_deg",
    "fixed_bone_length_drift_mm",
    "smoothed_cmm_base_tip_endpoint_epe_mm",
    "bend_plane_temporal_delta_deg",
)


def _validate_anatomical_ik_metrics_contract(
    metrics_payload: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Fail closed unless metrics retain the complete anatomical-IK receipt."""

    preparation = metrics_payload.get("take_preparation")
    guided = preparation.get("cmm_guided_ik") if isinstance(preparation, Mapping) else None
    if not isinstance(guided, Mapping) or set(guided) != set(ANATOMICAL_SIDES):
        raise ValueError(f"anatomical IK contract missing left/right diagnostics for {label}")

    distribution_keys = {"count", "mean", "median", "p95", "max", "rmse"}
    for side in ANATOMICAL_SIDES:
        side_payload = guided.get(side)
        validation = (
            side_payload.get("anatomical_validation")
            if isinstance(side_payload, Mapping)
            else None
        )
        if not isinstance(validation, Mapping):
            raise ValueError(f"anatomical IK contract missing for {label}/{side}")
        required_keys = {
            "solver",
            "dip_to_pip_flexion_ratio",
            "active_bend_threshold_deg",
            "opposite_bend_count",
            "active_bend_pair_count",
            "opposite_bend_fraction_active_gt_5deg",
            *IK_ANATOMICAL_DISTRIBUTIONS,
            "limits",
            "acceptance",
        }
        if set(validation) != required_keys:
            raise ValueError(f"anatomical IK contract fields mismatch for {label}/{side}")
        ratio = validation.get("dip_to_pip_flexion_ratio")
        threshold = validation.get("active_bend_threshold_deg")
        active_count = validation.get("active_bend_pair_count")
        opposite_fraction = validation.get("opposite_bend_fraction_active_gt_5deg")
        if (
            validation.get("solver") != "constant-curvature exact endpoint IK"
            or isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or float(ratio) != IK_DIP_TO_PIP_RATIO
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or float(threshold) != IK_ACTIVE_BEND_THRESHOLD_DEG
            or type(validation.get("opposite_bend_count")) is not int
            or validation.get("opposite_bend_count") != 0
            or type(active_count) is not int
            or active_count <= 0
            or isinstance(opposite_fraction, bool)
            or not isinstance(opposite_fraction, (int, float))
            or float(opposite_fraction) != 0.0
            or validation.get("limits") != IK_ANATOMICAL_LIMITS
            or validation.get("acceptance") != IK_ANATOMICAL_ACCEPTANCE
        ):
            raise ValueError(f"anatomical IK acceptance contract failed for {label}/{side}")

        distributions: dict[str, Mapping[str, Any]] = {}
        for name in IK_ANATOMICAL_DISTRIBUTIONS:
            values = validation.get(name)
            if (
                not isinstance(values, Mapping)
                or set(values) != distribution_keys
                or type(values.get("count")) is not int
                or values.get("count") <= 0
                or any(
                    isinstance(values.get(field), bool)
                    or not isinstance(values.get(field), (int, float))
                    or not np.isfinite(values.get(field))
                    or values.get(field) < 0.0
                    for field in ("mean", "median", "p95", "max", "rmse")
                )
            ):
                raise ValueError(
                    f"anatomical IK distribution contract failed for {label}/{side}/{name}"
                )
            distributions[name] = values
        if (
            distributions["pip_flexion_deg"]["max"] > IK_MAX_PIP_FLEXION_DEG + 1e-7
            or distributions["dip_flexion_deg"]["max"] > IK_MAX_DIP_FLEXION_DEG + 1e-7
            or distributions["thumb_flexion_deg"]["max"]
            > IK_MAX_THUMB_FLEXION_DEG + 1e-7
            or distributions["fixed_bone_length_drift_mm"]["max"] > 1e-7
            or distributions["smoothed_cmm_base_tip_endpoint_epe_mm"]["p95"]
            > IK_ANATOMICAL_LIMITS["endpoint_p95_max_mm"]
            or distributions["bend_plane_temporal_delta_deg"]["p95"]
            > IK_ANATOMICAL_LIMITS["bend_plane_temporal_p95_max_deg"]
            or distributions["bend_plane_temporal_delta_deg"]["max"]
            > IK_ANATOMICAL_LIMITS["bend_plane_temporal_max_deg"]
        ):
            raise ValueError(f"anatomical IK measured limits failed for {label}/{side}")


def _constant_curvature_candidates(
    base: np.ndarray,
    tip: np.ndarray,
    lengths: tuple[float, float, float],
    plane_normal: np.ndarray,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], float]:
    """Construct exact 3-link solutions with q_DIP = 0.65 q_PIP."""

    origin = np.asarray(base, dtype=np.float64)
    endpoint = np.asarray(tip, dtype=np.float64)
    chord = endpoint - origin
    distance = float(np.linalg.norm(chord))
    axis = chord / distance
    plane_normal = _unit_bend_normal(plane_normal, axis)
    plane_offset = np.cross(plane_normal, axis)
    plane_offset /= np.linalg.norm(plane_offset)
    first_length, second_length, third_length = lengths

    def reach(pip_flexion: float) -> tuple[float, float, float]:
        dip_flexion = IK_DIP_TO_PIP_RATIO * pip_flexion
        x = (
            first_length
            + second_length * math.cos(pip_flexion)
            + third_length * math.cos(pip_flexion + dip_flexion)
        )
        y = (
            second_length * math.sin(pip_flexion)
            + third_length * math.sin(pip_flexion + dip_flexion)
        )
        return math.hypot(x, y), x, y

    maximum_flexion = min(
        math.radians(IK_MAX_PIP_FLEXION_DEG),
        math.radians(IK_MAX_DIP_FLEXION_DEG) / IK_DIP_TO_PIP_RATIO,
    )
    straight_reach = float(sum(lengths))
    curled_reach, _, _ = reach(maximum_flexion)
    tolerance = 1e-6
    if distance > straight_reach + tolerance or distance < curled_reach - tolerance:
        raise ValueError(
            "IK endpoint is infeasible for the constant-curvature anatomical limits: "
            f"distance={distance:.9f}, "
            f"feasible=[{curled_reach:.9f}, {straight_reach:.9f}]"
        )
    if distance >= straight_reach - tolerance:
        pip_flexion = 0.0
    else:
        lower = 0.0
        upper = maximum_flexion
        # Reach is monotonic on the audited [0, 110 degree] interval. Bisection
        # gives a deterministic endpoint solution independent of the guide.
        for _ in range(64):
            middle = 0.5 * (lower + upper)
            middle_reach, _, _ = reach(middle)
            if middle_reach > distance:
                lower = middle
            else:
                upper = middle
        pip_flexion = 0.5 * (lower + upper)
    _, endpoint_x, endpoint_y = reach(pip_flexion)
    candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for side in (-1.0, 1.0):
        # Rotate the whole local constant-curvature chain so its resultant lies
        # exactly on the measured base-tip chord. The two side values are the
        # only anatomical mirror branches left to select temporally.
        first_heading = -math.atan2(side * endpoint_y, endpoint_x)
        second_heading = first_heading + side * pip_flexion
        third_heading = (
            second_heading + side * IK_DIP_TO_PIP_RATIO * pip_flexion
        )

        def direction(angle: float) -> np.ndarray:
            return math.cos(angle) * axis + math.sin(angle) * plane_offset

        pip = origin + first_length * direction(first_heading)
        dip = pip + second_length * direction(second_heading)
        solved_tip = dip + third_length * direction(third_heading)
        if float(np.linalg.norm(solved_tip - endpoint)) > 1e-7:
            raise ValueError("Constant-curvature IK failed to preserve the endpoint")
        solved = np.asarray((origin, pip, dip, endpoint))
        branch_normal = side * plane_normal
        first_bend, second_bend = _chain_bend_normals(solved, branch_normal)
        if float(np.dot(first_bend, second_bend)) < 1.0 - 1e-7:
            raise ValueError("Constant-curvature IK produced inconsistent bend normals")
        candidates.append((solved, first_bend, second_bend))
    return candidates, math.degrees(pip_flexion)


def _ik_chain(
    base: np.ndarray,
    tip: np.ndarray,
    guide: np.ndarray,
    lengths: Sequence[float],
    previous_normals: list[np.ndarray | None],
) -> tuple[np.ndarray, list[np.ndarray]]:
    segment_lengths = tuple(float(value) for value in lengths)
    if any(not np.isfinite(value) or value <= 0.0 for value in segment_lengths):
        raise ValueError("IK phalanx lengths must be finite and positive")
    if len(segment_lengths) == 2:
        if len(previous_normals) != 1:
            raise ValueError("Two-segment IK requires one previous bend normal")
        middle, normal = _sphere_circle_point(
            base,
            tip,
            segment_lengths[0],
            segment_lengths[1],
            guide[1],
            previous_normals[0],
        )
        solved = np.asarray((base, middle, tip))
        directions = np.diff(solved, axis=0)
        cosine = float(
            np.clip(
                np.dot(directions[0], directions[1])
                / (segment_lengths[0] * segment_lengths[1]),
                -1.0,
                1.0,
            )
        )
        thumb_flexion = math.degrees(math.acos(cosine))
        if thumb_flexion > IK_MAX_THUMB_FLEXION_DEG + 1e-7:
            raise ValueError(
                "Thumb IK exceeds the anatomical flexion limit: "
                f"{thumb_flexion:.6f} > {IK_MAX_THUMB_FLEXION_DEG:.6f} degrees"
            )
        return solved, [normal]
    if len(segment_lengths) != 3:
        raise ValueError("Only two- and three-segment finger IK is supported")
    if len(previous_normals) != 2:
        raise ValueError("Three-segment IK requires two previous bend normals")
    origin = np.asarray(base, dtype=np.float64)
    endpoint = np.asarray(tip, dtype=np.float64)
    base_tip = endpoint - origin
    base_tip_distance = float(np.linalg.norm(base_tip))
    if base_tip_distance <= 1e-8:
        raise ValueError("IK endpoints coincide")
    maximum_reach = float(sum(segment_lengths))
    endpoint_axis = base_tip / base_tip_distance
    previous_bend = _previous_bend_normal(previous_normals, endpoint_axis)
    plane_normal = _guide_bend_normal(guide, endpoint_axis, previous_bend)
    candidates, _ = _constant_curvature_candidates(
        origin,
        endpoint,
        segment_lengths,
        plane_normal,
    )
    if previous_bend is not None:
        continuous = [
            candidate
            for candidate in candidates
            if float(np.dot(candidate[1] + candidate[2], previous_bend)) >= -1e-7
        ]
        if not continuous:
            raise ValueError("IK could not preserve the previous anatomical branch")
        candidates = continuous
    if not candidates:
        raise ValueError("IK could not find an anatomical same-bend branch")
    guide_points = np.asarray(guide, dtype=np.float64)

    def candidate_cost(candidate: tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
        solved, first_bend, second_bend = candidate
        guide_cost = float(
            np.sum(np.square(solved[1] - guide_points[1]))
            + np.sum(np.square(solved[2] - guide_points[2]))
        )
        if previous_bend is None:
            return guide_cost
        continuity = 2.0 - float(np.dot(first_bend, previous_bend)) - float(
            np.dot(second_bend, previous_bend)
        )
        return guide_cost + maximum_reach * maximum_reach * continuity

    solved, pip_normal, dip_normal = min(candidates, key=candidate_cost)
    return solved, [pip_normal, dip_normal]


def _joint_flexion_degrees(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Vectorized unsigned angle between consecutive phalanx directions."""

    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    denominator = np.linalg.norm(first_values, axis=-1) * np.linalg.norm(
        second_values, axis=-1
    )
    cosine = np.sum(first_values * second_values, axis=-1) / np.maximum(
        denominator, 1e-12
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def cmm_guided_fixed_bone_ik(
    raw_pose: np.ndarray,
    roots_world_mm: np.ndarray,
    markers_world_mm: np.ndarray,
    *,
    side: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(raw_pose, dtype=np.float64)
    roots = np.asarray(roots_world_mm, dtype=np.float64)
    markers = np.asarray(markers_world_mm, dtype=np.float64)
    endpoint_kernel = np.asarray((1.0, 2.0, 1.0), dtype=np.float64) / 4.0
    pose_kernel = np.asarray((1.0, 4.0, 6.0, 4.0, 1.0), dtype=np.float64) / 16.0
    smooth_roots = _smooth(roots, endpoint_kernel)
    smooth_markers = _smooth(markers, endpoint_kernel)
    guide = _smooth(raw, np.asarray((1, 6, 15, 20, 15, 6, 1), dtype=np.float64) / 64.0)

    chains = ((0, 1, 2, 3), (0, 4, 5, 6, 7), (0, 8, 9, 10, 11), (0, 12, 13, 14, 15), (0, 16, 17, 18, 19))
    phalanx_lengths: list[tuple[float, ...]] = []
    endpoint_spans: list[float] = []
    for finger_index, chain in enumerate(chains):
        chain_indices = np.asarray(chain[1:], dtype=np.int32)
        raw_segments = np.linalg.norm(
            np.diff(raw[:, chain_indices], axis=1),
            axis=-1,
        )
        lengths = np.median(raw_segments, axis=0)
        base_marker = int(BASE_MARKER_INDICES[finger_index])
        tip_marker = int(TIP_MARKER_INDICES[finger_index])
        maximum_span = float(
            np.max(
                np.linalg.norm(
                    smooth_markers[:, tip_marker] - smooth_markers[:, base_marker],
                    axis=-1,
                )
            )
        )
        # Preserve the solver's median segment proportions. If the surface
        # marker span is longer, uniformly scale the chain with a 0.5 mm
        # feasibility margin so no endpoint is silently moved by the IK.
        total = float(np.sum(lengths))
        if total <= maximum_span + 0.5:
            lengths *= (maximum_span + 0.5) / total
        phalanx_lengths.append(tuple(float(value) for value in lengths))
        endpoint_spans.append(maximum_span)
    output = np.empty_like(raw)

    def project(guidance: np.ndarray) -> np.ndarray:
        projected = np.empty_like(guidance)
        projected[:, 0] = smooth_roots
        for finger_index, chain in enumerate(chains):
            base_marker = int(BASE_MARKER_INDICES[finger_index])
            tip_marker = int(TIP_MARKER_INDICES[finger_index])
            base = smooth_markers[:, base_marker]
            tip = smooth_markers[:, tip_marker]
            chain_indices = tuple(chain[1:])
            normals: list[np.ndarray | None] = [None] * (
                len(phalanx_lengths[finger_index]) - 1
            )
            for frame in range(len(raw)):
                local_guide = np.concatenate(
                    (base[frame, None, :], guidance[frame, chain_indices[1:]]),
                    axis=0,
                )
                local_guide[-1] = tip[frame]
                solved, solved_normals = _ik_chain(
                    base[frame],
                    tip[frame],
                    local_guide,
                    phalanx_lengths[finger_index],
                    normals,
                )
                normals = solved_normals
                projected[frame, chain_indices] = solved
        return projected

    output = project(guide)
    for _ in range(4):
        smoothed = _smooth(output, pose_kernel)
        smoothed[:, 0] = smooth_roots
        output = project(smoothed)
    thumb_angles: list[np.ndarray] = []
    pip_angles: list[np.ndarray] = []
    dip_angles: list[np.ndarray] = []
    bone_length_drift: list[np.ndarray] = []
    endpoint_error: list[np.ndarray] = []
    plane_temporal_delta: list[np.ndarray] = []
    opposite_bends = 0
    active_bend_pairs = 0
    active_threshold_deg = IK_ACTIVE_BEND_THRESHOLD_DEG
    for finger_index, chain in enumerate(chains):
        chain_indices = np.asarray(chain[1:], dtype=np.int32)
        points = output[:, chain_indices]
        segments = np.diff(points, axis=1)
        expected_lengths = np.asarray(phalanx_lengths[finger_index])[None, :]
        bone_length_drift.append(
            np.abs(np.linalg.norm(segments, axis=-1) - expected_lengths)
        )
        base_marker = int(BASE_MARKER_INDICES[finger_index])
        tip_marker = int(TIP_MARKER_INDICES[finger_index])
        endpoint_error.extend(
            (
                np.linalg.norm(points[:, 0] - smooth_markers[:, base_marker], axis=-1),
                np.linalg.norm(points[:, -1] - smooth_markers[:, tip_marker], axis=-1),
            )
        )
        if finger_index == 0:
            angles = _joint_flexion_degrees(segments[:, 0], segments[:, 1])
            thumb_angles.append(angles)
            plane_vectors = np.cross(segments[:, 0], segments[:, 1])
            plane_active = angles > active_threshold_deg
        else:
            first_angles = _joint_flexion_degrees(segments[:, 0], segments[:, 1])
            second_angles = _joint_flexion_degrees(segments[:, 1], segments[:, 2])
            pip_angles.append(first_angles)
            dip_angles.append(second_angles)
            first_planes = np.cross(segments[:, 0], segments[:, 1])
            second_planes = np.cross(segments[:, 1], segments[:, 2])
            first_norm = np.linalg.norm(first_planes, axis=-1)
            second_norm = np.linalg.norm(second_planes, axis=-1)
            active = (
                (first_angles > active_threshold_deg)
                & (second_angles > active_threshold_deg)
                & (first_norm > 1e-8)
                & (second_norm > 1e-8)
            )
            bend_dot = np.sum(first_planes * second_planes, axis=-1) / np.maximum(
                first_norm * second_norm, 1e-12
            )
            active_bend_pairs += int(np.count_nonzero(active))
            opposite_bends += int(np.count_nonzero(active & (bend_dot < 0.0)))
            plane_vectors = first_planes + second_planes
            plane_active = active
        plane_norm = np.linalg.norm(plane_vectors, axis=-1)
        normalized_planes = plane_vectors / np.maximum(plane_norm[:, None], 1e-12)
        temporal_active = (
            plane_active[1:]
            & plane_active[:-1]
            & (plane_norm[1:] > 1e-8)
            & (plane_norm[:-1] > 1e-8)
        )
        temporal_cosine = np.sum(
            normalized_planes[1:] * normalized_planes[:-1], axis=-1
        )
        plane_temporal_delta.append(
            np.degrees(
                np.arccos(np.clip(temporal_cosine[temporal_active], -1.0, 1.0))
            )
        )
    marker_error = np.linalg.norm(
        output[:, MARKER_TO_GLOVE_INDICES] - markers[:, :10],
        axis=-1,
    )
    thumb_values = np.concatenate(thumb_angles)
    pip_values = np.concatenate(pip_angles)
    dip_values = np.concatenate(dip_angles)
    bone_drift_values = np.concatenate(bone_length_drift, axis=1)
    endpoint_values = np.concatenate(endpoint_error)
    temporal_nonempty = [values for values in plane_temporal_delta if values.size]
    temporal_values = (
        np.concatenate(temporal_nonempty)
        if temporal_nonempty
        else np.zeros(1, dtype=np.float64)
    )
    opposite_fraction = (
        float(opposite_bends / active_bend_pairs) if active_bend_pairs else 0.0
    )
    anatomical_validation = {
        "solver": "constant-curvature exact endpoint IK",
        "dip_to_pip_flexion_ratio": IK_DIP_TO_PIP_RATIO,
        "active_bend_threshold_deg": active_threshold_deg,
        "opposite_bend_count": opposite_bends,
        "active_bend_pair_count": active_bend_pairs,
        "opposite_bend_fraction_active_gt_5deg": opposite_fraction,
        "pip_flexion_deg": _distribution(pip_values),
        "dip_flexion_deg": _distribution(dip_values),
        "thumb_flexion_deg": _distribution(thumb_values),
        "fixed_bone_length_drift_mm": _distribution(bone_drift_values),
        "smoothed_cmm_base_tip_endpoint_epe_mm": _distribution(endpoint_values),
        "bend_plane_temporal_delta_deg": _distribution(temporal_values),
        "limits": dict(IK_ANATOMICAL_LIMITS),
        "acceptance": {
            "no_opposite_active_bends": opposite_bends == 0,
            "pip_within_limit": float(np.max(pip_values))
            <= IK_MAX_PIP_FLEXION_DEG + 1e-7,
            "dip_within_limit": float(np.max(dip_values))
            <= IK_MAX_DIP_FLEXION_DEG + 1e-7,
            "thumb_within_limit": float(np.max(thumb_values))
            <= IK_MAX_THUMB_FLEXION_DEG + 1e-7,
            "fixed_bone_lengths": float(np.max(bone_drift_values)) <= 1e-7,
            "endpoint_p95_within_0_7_mm": float(np.percentile(endpoint_values, 95))
            <= 0.7,
            "bend_plane_temporal_p95_within_8_deg": float(
                np.percentile(temporal_values, 95)
            )
            <= 8.0,
            "bend_plane_temporal_max_within_45_deg": float(
                np.max(temporal_values)
            )
            <= 45.0,
        },
    }
    failed_acceptance = [
        name
        for name, accepted in anatomical_validation["acceptance"].items()
        if accepted is not True
    ]
    if failed_acceptance:
        raise ValueError(
            f"Anatomical IK acceptance failed for {side}: {failed_acceptance}"
        )
    return output, {
        "endpoint_smoothing_kernel": endpoint_kernel.tolist(),
        "guide_smoothing_kernel": [1 / 64, 6 / 64, 15 / 64, 20 / 64, 15 / 64, 6 / 64, 1 / 64],
        "alternating_smoothing_projection_rounds": 4,
        "fixed_phalanx_lengths_mm": [list(values) for values in phalanx_lengths],
        "phalanx_length_provenance": {
            "source": "per-segment median of the calibrated IMU solver pose",
            "endpoint_feasibility": (
                "uniformly scale only when needed so sum(lengths) is "
                "max smoothed CMM base-tip span + 0.5 mm"
            ),
            "maximum_smoothed_cmm_base_tip_span_mm": endpoint_spans,
        },
        "anatomical_validation": anatomical_validation,
        "marker_epe_to_raw_cmm_mm": _distribution(marker_error),
    }


def _old_target_pose(mocap_pose: np.ndarray) -> np.ndarray:
    """Map BVH 21J to the glove's 20J anatomy, dropping thumb CMC."""

    mocap = np.asarray(mocap_pose, dtype=np.float64)
    target = np.empty((len(mocap), 20, 3), dtype=np.float64)
    target[:, 0] = mocap[:, 0]
    # Glove thumb MCP/PIP/tip map to BVH thumb MCP/IP/tip. BVH CMC is kept
    # visible in the hollow reference but has no 20J solver counterpart.
    target[:, 1:4] = mocap[:, 2:5]
    target[:, 4:20] = mocap[:, 5:21]
    return target


def _prepare_old_take(
    project_root: Path,
    take: Any,
    *,
    translation_xyz_mm: np.ndarray,
    registrations: Mapping[str, viz.SimilarityRegistration],
    registration_source: str,
) -> LabTake:
    dataset = project_root / "同步整理_20260829_三段"
    segment = viz._discover_segment(dataset, take.segment_key)
    calibration_path = viz._default_calibration(dataset)
    prepared = viz.prepare_take(
        segment,
        calibration_path,
        registrations=registrations,
        registration_source_segment=registration_source,
        max_interpolation_gap_ms=25.0,
        pose_mode=viz.POSE_MODE_GLOVE_WRIST,
        operator_world_translation_xyz_mm=translation_xyz_mm,
    )
    mocap = mocap_overlay.prepare_mocap_video(
        segment,
        calibration_path,
        max_interpolation_gap_ms=25.0,
        mocap_position_source=mocap_overlay.MOCAP_POSITION_SOURCE_SKELETON_BVH,
        mocap_world_translation_mm=translation_xyz_mm,
    )
    if not np.allclose(prepared.rgb_device_s, mocap.rgb_device_s, atol=0.0, rtol=0.0):
        raise ValueError("Old-take preparations disagree on RGB timestamps")
    first = int(take.source_first)
    stop = first + int(take.frame_count)
    if not np.all(mocap.mocap_valid[first:stop]):
        raise ValueError(f"BVH MOCAP is incomplete for {segment.name}")

    raw_imu: dict[str, np.ndarray] = {}
    target_pose: dict[str, np.ndarray] = {}
    strict_valid: dict[str, np.ndarray] = {}
    resampling: dict[str, Any] = {}
    glove_paths: dict[str, Path] = {}
    for side_index, side in enumerate(("left", "right")):
        glove_path = (
            segment
            / "手套解算"
            / "solved"
            / "primary"
            / f"{side}_hand_keypoints.csv"
        )
        glove_paths[side] = glove_path
        sequence = viz.load_glove_keypoints(glove_path)
        sampled, diagnostics = manifold_resample_pose(
            sequence.times_s,
            sequence.world_mm,
            prepared.monotonic_s[first:stop],
        )
        raw_imu[side] = viz.apply_registration(
            sampled,
            mocap.mocap_mm[first:stop, side_index],
            registrations[side],
            mocap_wrist_rotations=None,
        )
        target_pose[side] = _old_target_pose(mocap.mocap_mm[first:stop, side_index])
        strict_valid[side] = (
            prepared.glove_valid[side][first:stop]
            & mocap.mocap_valid[first:stop]
        )
        resampling[side] = diagnostics
        if diagnostics["finite_output_frames"] != take.frame_count:
            raise ValueError(f"Continuous solver output is incomplete for {segment.name} {side}")

    return LabTake(
        order=int(take.number),
        id=f"take{take.number:02d}",
        label=f"Take {take.number:02d}",
        source=segment.name,
        source_video=prepared.video.path,
        source_first=first,
        frame_count=int(take.frame_count),
        fps=float(prepared.video.fps),
        source_width=int(prepared.video.width),
        source_height=int(prepared.video.height),
        projection_kind="old",
        calibration=mocap.calibration,
        mocap_draw={
            side: mocap.mocap_mm[first:stop, side_index]
            for side_index, side in enumerate(("left", "right"))
        },
        mocap_chains=viz.MOCAP_CHAINS,
        raw_imu=raw_imu,
        target_pose=target_pose,
        strict_valid=strict_valid,
        marker_targets=None,
        source_assets={
            "rgb_video": _source_asset(project_root, prepared.video.path),
            "camera_world_calibration": _source_asset(project_root, calibration_path),
            "left_skeleton_bvh": _source_asset(project_root, mocap.mocap_position_paths["left"]),
            "right_skeleton_bvh": _source_asset(project_root, mocap.mocap_position_paths["right"]),
            "left_glove_solver_keypoints": _source_asset(project_root, glove_paths["left"]),
            "right_glove_solver_keypoints": _source_asset(project_root, glove_paths["right"]),
        },
        preparation={
            "solver_resampling": resampling,
            "thumb_topology": "glove MCP/PIP/tip -> BVH MCP/IP/tip; BVH CMC omitted",
            "primary_scientific_joints": [viz.GLOVE_NAMES[index] for index in OLD_PRIMARY_INDICES],
            "original_strict_frame_counts": {
                side: int(np.count_nonzero(strict_valid[side])) for side in ("left", "right")
            },
            "manual_world_translation_xyz_mm": translation_xyz_mm.tolist(),
        },
    )


def _prepare_cmm_capture(
    project_root: Path,
    *,
    recording: str,
    take: str,
    order: int,
    take_id: str,
    label: str,
    world_translation_xyz_mm: np.ndarray,
) -> LabTake:
    capture = load_new_capture(
        project_root / "thor_new4_20260831_processed",
        project_root / "movementcap_20260831_worldcalib_tabletop_final.tar.gz",
        recording=recording,
        take=take,
        marker_tracks=CMM_MARKER_TRACKS[take],
    )
    _, strict_both, strict_diagnostics = load_solved_pose(capture)
    count = len(capture.clock.output_indices)
    calibration_stop = int(round(count * CMM_CALIBRATION_FRACTION))
    calibration_window = np.arange(count) < calibration_stop
    world_translation = np.asarray(world_translation_xyz_mm, dtype=np.float64)
    if world_translation.shape != (3,) or not np.all(np.isfinite(world_translation)):
        raise ValueError("CMM display translation must contain three finite millimetres")

    shifted_markers = np.asarray(capture.marker_world_mm, dtype=np.float64).copy()
    roots = np.empty((count, 2, 3), dtype=np.float64)
    raw_imu: dict[str, np.ndarray] = {}
    target_pose: dict[str, np.ndarray] = {}
    strict_valid: dict[str, np.ndarray] = {}
    marker_targets: dict[str, np.ndarray] = {}
    fixed_registration: dict[str, Any] = {}
    resampling: dict[str, Any] = {}
    ik_diagnostics: dict[str, Any] = {}
    glove_paths: dict[str, Path] = {}

    for side_index, side in enumerate(ANATOMICAL_SIDES):
        offset = world_translation
        shifted_markers[:, side_index] += offset
        roots[:, side_index] = virtual_wrist(
            capture.marker_world_mm[:, side_index],
            rear_offset_mm=DEFAULT_REAR_OFFSET_MM,
        ) + offset
        marker_targets[side] = shifted_markers[:, side_index, :10]

        glove_path = (
            capture.root
            / "glove_processing"
            / "solved"
            / "primary"
            / f"{side}_hand_keypoints.csv"
        )
        glove_paths[side] = glove_path
        sequence = viz.load_glove_keypoints(glove_path)
        sampled, diagnostics = manifold_resample_pose(
            sequence.times_s,
            sequence.world_mm,
            capture.clock.target_monotonic_s,
        )
        resampling[side] = diagnostics
        source_vectors = (
            sampled[:, TAKE007_BASE_GLOVE_INDICES]
            - sampled[:, 0, None, :]
        )
        target_vectors = (
            capture.marker_world_mm[:, side_index, TAKE007_BASE_MARKER_INDICES]
            - virtual_wrist(
                capture.marker_world_mm[:, side_index],
                rear_offset_mm=DEFAULT_REAR_OFFSET_MM,
            )[:, None, :]
        )
        fit_mask = strict_both & calibration_window
        transform: FixedSimilarity = _fit_fixed_similarity(
            source_vectors[fit_mask], target_vectors[fit_mask]
        )
        raw_imu[side] = (
            (sampled - sampled[:, 0, None, :])
            @ transform.rotation.T
            * transform.scale
            + roots[:, side_index, None, :]
        )
        target_pose[side], ik_diagnostics[side] = cmm_guided_fixed_bone_ik(
            raw_imu[side],
            roots[:, side_index],
            shifted_markers[:, side_index],
            side=side,
        )
        strict_valid[side] = strict_both.copy()
        fixed_registration[side] = {
            "scale": transform.scale,
            "rotation_mocap_from_solver": transform.rotation.tolist(),
            "input_frames": transform.input_frame_count,
            "retained_frames": transform.retained_frame_count,
            "base_residual_median_mm": transform.residual_median_mm,
            "base_residual_p95_mm": transform.residual_p95_mm,
        }
        if not np.all(np.isfinite(target_pose[side])):
            raise ValueError(f"{take} IK is incomplete for {side}")

    mocap_nodes = raw_mocap_nodes(
        capture.marker_world_mm,
        rear_offset_mm=DEFAULT_REAR_OFFSET_MM,
    )
    for side_index, side in enumerate(ANATOMICAL_SIDES):
        mocap_nodes[:, side_index] += world_translation

    probe = cv2.VideoCapture(str(capture.rgb_path))
    source_width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(probe.get(cv2.CAP_PROP_FPS))
    probe.release()
    return LabTake(
        order=order,
        id=take_id,
        label=label,
        source=f"{capture.recording}/{capture.take}",
        source_video=capture.rgb_path,
        source_first=0,
        frame_count=count,
        fps=fps,
        source_width=source_width,
        source_height=source_height,
        projection_kind="take007",
        calibration=capture.camera,
        mocap_draw={
            side: mocap_nodes[:, side_index]
            for side_index, side in enumerate(ANATOMICAL_SIDES)
        },
        mocap_chains=RAW_CHAINS,
        raw_imu=raw_imu,
        target_pose=target_pose,
        strict_valid=strict_valid,
        marker_targets=marker_targets,
        source_assets={
            "rgb_video": _source_asset(project_root, capture.rgb_path),
            "camera_intrinsics": _source_asset(
                project_root, capture.camera.intrinsics_path
            ),
            "cmm_markers": _source_asset(project_root, capture.cmm_path),
            "camera_alignment": _source_asset(project_root, capture.alignment_path),
            "glove_validity_summary": _source_asset(project_root, capture.frame_summary_path),
            "left_glove_solver_keypoints": _source_asset(project_root, glove_paths["left"]),
            "right_glove_solver_keypoints": _source_asset(project_root, glove_paths["right"]),
        },
        preparation={
            "solver_resampling": resampling,
            "fixed_first_20_percent_registration": fixed_registration,
            "cmm_guided_ik": ik_diagnostics,
            "original_strict_validity": strict_diagnostics,
            "marker_track_semantics": {
                "source": (
                    "user-supplied Take_005/006/007 #1..#11 marker-ID table and "
                    "placement photographs; only Take_007 left #1 11781->12503 "
                    "alias is trajectory-continuity inferred"
                ),
                "tracks": {
                    side: [list(track) for track in CMM_MARKER_TRACKS[take][side]]
                    for side in ANATOMICAL_SIDES
                },
            },
            "manual_world_translation": {
                "global_world_xyz_mm": world_translation.tolist(),
                "left_world_xyz_mm": [0.0, 0.0, 0.0],
                "right_world_xyz_mm": [0.0, 0.0, 0.0],
                "rear_offset_mm": DEFAULT_REAR_OFFSET_MM,
            },
        },
    )


def _slice_lab_take(
    take: LabTake,
    *,
    start: int,
    stop: int,
    order: int,
    take_id: str,
    label: str,
) -> LabTake:
    if not (0 <= start < stop <= take.frame_count):
        raise ValueError(f"Invalid CMM take slice [{start}, {stop})")
    sliced = replace(
        take,
        order=order,
        id=take_id,
        label=label,
        source_first=take.source_first + start,
        frame_count=stop - start,
        mocap_draw={
            side: values[start:stop].copy()
            for side, values in take.mocap_draw.items()
        },
        raw_imu={
            side: values[start:stop].copy()
            for side, values in take.raw_imu.items()
        },
        target_pose={
            side: values[start:stop].copy()
            for side, values in take.target_pose.items()
        },
        strict_valid={
            side: values[start:stop].copy()
            for side, values in take.strict_valid.items()
        },
        marker_targets={
            side: values[start:stop].copy()
            for side, values in (take.marker_targets or {}).items()
        },
        preparation={
            **take.preparation,
            "published_window": {
                "source_output_start_inclusive": start,
                "source_output_stop_exclusive": stop,
                "source_output_frames": stop - start,
                "split_is_disjoint": True,
            },
            "window_strict_frame_counts": {
                side: int(np.count_nonzero(take.strict_valid[side][start:stop]))
                for side in ANATOMICAL_SIDES
            },
        },
    )
    if sliced.marker_targets is None or len(sliced.marker_targets) != 2:
        raise ValueError("CMM take slice lost marker targets")
    return sliced


def build_method_poses(
    take: LabTake,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
    poses: dict[str, dict[str, np.ndarray]] = {method.id: {} for method in METHODS}
    diagnostics: dict[str, dict[str, Any]] = {method.id: {} for method in METHODS}
    for side in ANATOMICAL_SIDES:
        raw = take.raw_imu[side]
        target = take.target_pose[side]
        poses["s2_continuous"][side] = raw.copy()
        poses["s2_gaussian"][side] = smooth_pose_s2(raw)
        poses["kalman_rts"][side] = kalman_rts_pose(raw, fps=take.fps)
        poses["posterior_75"][side] = mocap_guided_posterior(
            raw, target, imu_residual_gain=0.25
        )
        poses["posterior_98"][side] = mocap_guided_posterior(
            raw, target, imu_residual_gain=0.02
        )
        poses["rbf_self_fit"][side], diagnostics["rbf_self_fit"][side] = (
            rbf_same_sequence_fit(raw, target, fps=take.fps)
        )
        poses["guided_ik"][side] = target.copy()
        for method in METHODS:
            finite = np.all(np.isfinite(poses[method.id][side]), axis=(1, 2))
            diagnostics[method.id].setdefault(side, {})
            diagnostics[method.id][side]["finite_frames"] = int(np.count_nonzero(finite))
            diagnostics[method.id][side]["total_frames"] = int(take.frame_count)
            if not np.all(finite):
                raise ValueError(f"{take.id}/{method.id}/{side} is not continuous")
    return poses, diagnostics


def _high_frequency_residual(pose: np.ndarray) -> np.ndarray:
    points = np.asarray(pose, dtype=np.float64)
    relative = points - points[:, 0, None, :]
    kernel = np.asarray((1.0, 4.0, 6.0, 4.0, 1.0), dtype=np.float64) / 16.0
    return np.linalg.norm(relative - _smooth(relative, kernel), axis=-1)


def _jerk_magnitude(pose: np.ndarray, *, fps: float) -> np.ndarray:
    points = np.asarray(pose, dtype=np.float64)
    relative = points - points[:, 0, None, :]
    return np.linalg.norm(np.diff(relative, n=3, axis=0) * float(fps) ** 3, axis=-1)


def _bone_length_diagnostics(pose: np.ndarray) -> dict[str, Any]:
    _, _, lengths = _bone_vectors(pose)
    means = np.mean(lengths, axis=0)
    coefficient = np.divide(
        np.std(lengths, axis=0),
        means,
        out=np.zeros_like(means),
        where=means > 1e-9,
    )
    return {
        "maximum_coefficient_of_variation": float(np.max(coefficient)),
        "median_coefficient_of_variation": float(np.median(coefficient)),
        "median_lengths_mm": np.median(lengths, axis=0).tolist(),
    }


def _method_metrics(
    take: LabTake,
    method: MethodSpec,
    method_poses: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, float | int]]:
    sides: dict[str, Any] = {}
    pooled_primary: list[np.ndarray] = []
    pooled_jitter: list[np.ndarray] = []
    for side in ANATOMICAL_SIDES:
        pose = np.asarray(method_poses[side], dtype=np.float64)
        target = take.target_pose[side]
        relative = pose - pose[:, 0, None, :]
        target_relative = target - target[:, 0, None, :]
        full_target_error = np.linalg.norm(relative[:, 1:] - target_relative[:, 1:], axis=-1)
        if take.marker_targets is None:
            primary_error = np.linalg.norm(
                relative[:, OLD_PRIMARY_INDICES]
                - target_relative[:, OLD_PRIMARY_INDICES],
                axis=-1,
            )
            primary_label = "root-normalized 16 non-thumb BVH joints"
            tip_error = primary_error[:, (3, 7, 11, 15)]
        else:
            primary_error = np.linalg.norm(
                pose[:, MARKER_TO_GLOVE_INDICES] - take.marker_targets[side],
                axis=-1,
            )
            primary_label = "ten photographed CMM surface-marker correspondences"
            tip_error = primary_error[:, TIP_MARKER_INDICES]
        strict = take.strict_valid[side]
        raw_relative = take.raw_imu[side] - take.raw_imu[side][:, 0, None, :]
        raw_deviation = np.linalg.norm(relative[:, 1:] - raw_relative[:, 1:], axis=-1)
        jitter = _high_frequency_residual(pose)[:, 1:]
        jerk = _jerk_magnitude(pose, fps=take.fps)[:, 1:]
        pooled_primary.append(primary_error.reshape(-1))
        pooled_jitter.append(jitter.reshape(-1))
        sides[side] = {
            "finite_frames": int(np.count_nonzero(np.all(np.isfinite(pose), axis=(1, 2)))),
            "total_frames": int(take.frame_count),
            "primary_reference": primary_label,
            "primary_reference_epe_mm_all_frames": _distribution(primary_error),
            "primary_reference_epe_mm_original_strict_subset": _distribution(primary_error[strict]),
            "primary_fingertip_epe_mm_all_frames": _distribution(tip_error),
            "full_19_joint_pseudo_target_epe_mm": _distribution(full_target_error),
            "deviation_from_continuous_imu_mm": _distribution(raw_deviation),
            "root_relative_high_frequency_residual_mm": _distribution(jitter),
            "root_relative_jerk_mm_s3": _distribution(jerk),
            "bone_lengths": _bone_length_diagnostics(pose),
            "algorithm_diagnostics": diagnostics.get(side, {}),
        }
    primary_distribution = _distribution(np.concatenate(pooled_primary))
    jitter_distribution = _distribution(np.concatenate(pooled_jitter))
    payload = {
        "schema": METRICS_SCHEMA,
        "status": "complete",
        "take_id": take.id,
        "take_label": take.label,
        "source": take.source,
        "source_first": take.source_first,
        "method_id": method.id,
        "method_label": method.label,
        "supervision": method.supervision,
        "method_description": method.description,
        "interpretation_boundary": method.boundary,
        "rendered_frames": take.frame_count,
        "display_contract": {
            "pose_drawn_on_every_frame": True,
            "validity_gate_hides_pose": False,
            "stale_color_change": False,
            "connector_or_error_lines": False,
            "per_frame_error_text": False,
        },
        "pooled_sides": {
            "primary_reference_epe_mm": primary_distribution,
            "root_relative_high_frequency_residual_mm": jitter_distribution,
        },
        "sides": sides,
        "take_preparation": take.preparation,
        "source_assets": take.source_assets,
        "claim_boundary": (
            "MOCAP-conditioned and same-sequence teacher-fit methods optimize this recording's "
            "visual adherence and are not independent IMU accuracy or generalization evidence."
        ),
    }
    return payload, {
        "reference_median_mm": float(primary_distribution["median"]),
        "reference_p95_mm": float(primary_distribution["p95"]),
        "jitter_p95_mm": float(jitter_distribution["p95"]),
        "finite_frames": int(take.frame_count),
    }


def _project_take(points: np.ndarray, take: LabTake) -> tuple[np.ndarray, np.ndarray]:
    if take.projection_kind == "old":
        return viz.project_world_to_rgb(points, take.calibration)
    if take.projection_kind == "take007":
        return project_world(points, take.calibration)
    raise ValueError(f"Unknown projection kind: {take.projection_kind}")


def _draw_header(
    image: np.ndarray,
    take: LabTake,
    method: MethodSpec,
) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 88), (3, 9, 17), -1)
    cv2.addWeighted(overlay, 0.74, image, 0.26, 0.0, image)
    def clean_text(
        text: str,
        origin: tuple[int, int],
        *,
        scale: float,
        color: tuple[int, int, int],
    ) -> None:
        # The header already has a dark backing plate. A single AA pass stays
        # crisp after H.264 encoding; the generic four-pixel outline can look
        # like repeated trailing letters at 540p.
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

    clean_text(
        f"{take.label} | {method.label}",
        (15, 26),
        scale=0.56,
        color=(245, 248, 252),
    )
    clean_text(
        "MOCAP hollow | IMU result solid | 100% frames | no connector lines",
        (15, 51),
        scale=0.45,
        color=(180, 232, 242),
    )
    boundary = {
        "imu_only": "IMU articulation / MOCAP-derived spatial calibration / no per-frame finger target",
        "mocap_conditioned": "same-frame MOCAP-conditioned offline visualization",
        "same_sequence_teacher_fit": "same-sequence neural teacher-fit / NO HOLDOUT",
        "mocap_conditioned_oracle": "MOCAP-guided IK visual oracle / NOT IMU ACCURACY",
    }[method.supervision]
    clean_text(
        boundary,
        (15, 75),
        scale=0.43,
        color=(110, 245, 175) if method.supervision == "imu_only" else (90, 190, 255),
    )


def _draw_layered_hand(
    image: np.ndarray,
    mocap_pixels: np.ndarray,
    imu_pixels: np.ndarray,
    mocap_chains: Sequence[Sequence[int]],
    *,
    side: str,
) -> None:
    # A dark halo separates the two layers even when the optimized pose lands
    # exactly on MOCAP. No joint-to-joint residual connector is ever drawn.
    viz.draw_hand_skeleton(
        image,
        mocap_pixels,
        mocap_chains,
        HALO_BGR,
        thickness=8,
        radius=8,
        hollow=False,
    )
    viz.draw_hand_skeleton(
        image,
        mocap_pixels,
        mocap_chains,
        MOCAP_COLORS[side],
        thickness=5,
        radius=6,
        hollow=True,
    )
    viz.draw_hand_skeleton(
        image,
        imu_pixels,
        GLOVE_CHAINS,
        HALO_BGR,
        thickness=5,
        radius=5,
        hollow=False,
    )
    viz.draw_hand_skeleton(
        image,
        imu_pixels,
        GLOVE_CHAINS,
        IMU_COLORS[side],
        thickness=2,
        radius=3,
        hollow=False,
    )


def _write_motion_asset(
    path: Path,
    take: LabTake,
    method: MethodSpec,
    method_poses: Mapping[str, np.ndarray],
    *,
    reference_extent_mm: float,
) -> dict[str, Any]:
    mocap_count = int(take.mocap_draw["left"].shape[1])
    imu_count = 20
    layers = (
        ("mocap-left", "mocap", "left", take.mocap_draw["left"], take.mocap_chains),
        ("mocap-right", "mocap", "right", take.mocap_draw["right"], take.mocap_chains),
        ("imu-left", "result", "left", method_poses["left"], GLOVE_CHAINS),
        ("imu-right", "result", "right", method_poses["right"], GLOVE_CHAINS),
    )
    arrays = [np.asarray(layer[3], dtype=np.float64) for layer in layers]
    world = np.concatenate(arrays, axis=1)
    if not np.all(np.isfinite(world)):
        raise ValueError(f"3D motion contains non-finite values: {take.id}/{method.id}")
    quantum_mm = 0.1
    origin = np.floor(np.min(world, axis=(0, 1)) / quantum_mm) * quantum_mm
    quantized = np.rint((world - origin[None, None, :]) / quantum_mm).astype(np.int32)
    reconstructed = origin[None, None, :] + quantized.astype(np.float64) * quantum_mm
    maximum_quantization_error = float(np.max(np.abs(reconstructed - world)))

    reference_extent = float(reference_extent_mm)
    if not np.isfinite(reference_extent) or reference_extent <= 0.0:
        raise ValueError("3D reference extent must be finite and positive")
    layout: list[dict[str, Any]] = []
    offset = 0
    for layer_id, kind, side, array, chains in layers:
        count = int(array.shape[1])
        layout.append(
            {
                "id": layer_id,
                "kind": kind,
                "side": side,
                "joint_offset": offset,
                "joint_count": count,
                "chains": [list(chain) for chain in chains],
            }
        )
        offset += count
    payload = {
        "schema": MOTION_SCHEMA,
        "take_id": take.id,
        "method_id": method.id,
        "frame_count": int(take.frame_count),
        "fps": float(take.fps),
        "units": "mm",
        "coordinate_frame": "per-take MOCAP world after the reviewed display translation",
        "encoding": {
            "kind": "frame-major-flat-int32-json",
            "components": "XYZ",
            "quantum_mm": quantum_mm,
            "origin_mm": origin.tolist(),
            "joint_count": int(world.shape[1]),
            "values_per_frame": int(world.shape[1] * 3),
        },
        "layers": layout,
        "view": {
            "focus": "per-frame midpoint of MOCAP left/right wrists",
            "reference_extent_mm": reference_extent,
            "fixed_across_methods_for_take": True,
        },
        "frames": quantized.reshape(len(world), -1).tolist(),
        "validation": {
            "finite_frames": int(len(world)),
            "maximum_quantization_error_mm": maximum_quantization_error,
            "status": "pass",
        },
        "claim_boundary": (
            "Interactive orthographic 3D review of the same rendered arrays; "
            "MOCAP-conditioned layers remain visualization, not independent IMU accuracy."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "joint_count": int(world.shape[1]),
        "reference_extent_mm": reference_extent,
        "maximum_quantization_error_mm": maximum_quantization_error,
    }


def _render_take_methods(
    take: LabTake,
    staging: Path,
    poses: Mapping[str, Mapping[str, np.ndarray]],
    diagnostics: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output_width = 960
    output_height = int(round(take.source_height * output_width / take.source_width))
    output_height += output_height % 2
    if (output_width, output_height) != (960, 540):
        raise ValueError(f"Unexpected source aspect ratio for {take.id}")
    scale = output_width / take.source_width

    common_center = 0.5 * (
        np.asarray(take.mocap_draw["left"])[:, 0]
        + np.asarray(take.mocap_draw["right"])[:, 0]
    )
    extent_samples = [
        np.linalg.norm(
            np.asarray(take.mocap_draw[side]) - common_center[:, None, :],
            axis=-1,
        ).reshape(-1)
        for side in ANATOMICAL_SIDES
    ]
    for method in METHODS:
        for side in ANATOMICAL_SIDES:
            extent_samples.append(
                np.linalg.norm(
                    np.asarray(poses[method.id][side]) - common_center[:, None, :],
                    axis=-1,
                ).reshape(-1)
            )
    common_reference_extent = max(
        120.0,
        float(np.percentile(np.concatenate(extent_samples), 99.5) * 1.15),
    )

    projected_mocap: dict[str, np.ndarray] = {}
    projected_imu: dict[str, dict[str, np.ndarray]] = {
        method.id: {} for method in METHODS
    }
    for side in ANATOMICAL_SIDES:
        pixels, positive = _project_take(take.mocap_draw[side], take)
        pixels[~positive] = np.nan
        projected_mocap[side] = pixels * scale
        for method in METHODS:
            pixels, positive = _project_take(poses[method.id][side], take)
            pixels[~positive] = np.nan
            projected_imu[method.id][side] = pixels * scale

    entries: list[dict[str, Any]] = []
    writers: dict[str, cv2.VideoWriter] = {}
    intermediate_paths: dict[str, Path] = {}
    output_paths: dict[str, Path] = {}
    poster_paths: dict[str, Path] = {}
    snapshots: dict[str, list[np.ndarray]] = {method.id: [] for method in METHODS}
    snapshot_indices = set(
        np.rint(np.linspace(0, take.frame_count - 1, 6)).astype(int).tolist()
    )
    for method_index, method in enumerate(METHODS, start=1):
        order = (take.order - 1) * EXPECTED_METHODS + method_index
        stem = f"{order:02d}_{take.id}_{method.id}"
        output = staging / "videos" / f"{stem}.mp4"
        intermediate = staging / "videos" / f"{stem}.mp4v.mp4"
        writer = cv2.VideoWriter(
            str(intermediate),
            cv2.VideoWriter_fourcc(*"mp4v"),
            take.fps,
            (output_width, output_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create {intermediate}")
        writers[method.id] = writer
        intermediate_paths[method.id] = intermediate
        output_paths[method.id] = output
        poster_paths[method.id] = staging / "posters" / f"{stem}.jpg"

        metrics_payload, summary = _method_metrics(
            take,
            method,
            poses[method.id],
            diagnostics[method.id],
        )
        metrics_relative = Path("metrics") / f"{stem}.json"
        metrics_path = staging / metrics_relative
        metrics_path.write_text(
            json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        motion_relative = Path("motions") / f"{stem}.json"
        motion_info = _write_motion_asset(
            staging / motion_relative,
            take,
            method,
            poses[method.id],
            reference_extent_mm=common_reference_extent,
        )
        entries.append(
            {
                "order": order,
                "id": f"{take.id}-{method.id}",
                "take_id": take.id,
                "take_label": take.label,
                "method_id": method.id,
                "method_label": method.label,
                "supervision": method.supervision,
                "source": take.source,
                "source_first": take.source_first,
                "filename": output.name,
                "poster": poster_paths[method.id].relative_to(staging).as_posix(),
                "metrics": metrics_relative.as_posix(),
                "motion": motion_relative.as_posix(),
                "motion_bytes": motion_info["bytes"],
                "motion_sha256": motion_info["sha256"],
                "expected_frames": take.frame_count,
                "summary": summary,
            }
        )

    source_video = cv2.VideoCapture(str(take.source_video))
    try:
        for source_index in range(take.source_first + take.frame_count):
            ok, frame = source_video.read()
            if not ok:
                raise RuntimeError(f"RGB decode stopped at source frame {source_index}")
            if source_index < take.source_first:
                continue
            output_index = source_index - take.source_first
            base = cv2.resize(
                frame,
                (output_width, output_height),
                interpolation=cv2.INTER_AREA,
            )
            for method in METHODS:
                image = base.copy()
                for side in ANATOMICAL_SIDES:
                    _draw_layered_hand(
                        image,
                        projected_mocap[side][output_index],
                        projected_imu[method.id][side][output_index],
                        take.mocap_chains,
                        side=side,
                    )
                _draw_header(image, take, method)
                writers[method.id].write(image)
                if output_index in snapshot_indices:
                    snapshots[method.id].append(image.copy())
    finally:
        source_video.release()
        for writer in writers.values():
            writer.release()

    for method in METHODS:
        _contact_sheet(snapshots[method.id], poster_paths[method.id])
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _transcode_h264,
                intermediate_paths[method.id],
                output_paths[method.id],
            )
            for method in METHODS
        ]
        for future in futures:
            future.result()
    return entries


def _method_rankings(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rankings: list[dict[str, Any]] = []
    for method in METHODS:
        selected = [entry for entry in entries if entry["method_id"] == method.id]
        median = float(np.mean([entry["summary"]["reference_median_mm"] for entry in selected]))
        p95 = float(np.mean([entry["summary"]["reference_p95_mm"] for entry in selected]))
        jitter = float(np.mean([entry["summary"]["jitter_p95_mm"] for entry in selected]))
        rankings.append(
            {
                "method_id": method.id,
                "method_label": method.label,
                "supervision": method.supervision,
                "mean_take_reference_median_mm": median,
                "mean_take_reference_p95_mm": p95,
                "mean_take_jitter_p95_mm": jitter,
                "visual_score_lower_is_better": median + 0.25 * p95 + 0.25 * jitter,
            }
        )
    rankings.sort(key=lambda item: item["visual_score_lower_is_better"])
    for rank, item in enumerate(rankings, start=1):
        item["rank"] = rank
    return rankings


def _write_readme(destination: Path) -> None:
    (destination / "README.md").write_text(
        """# IMU -> MOCAP visualization lab

This independent review folder contains 28 videos: four action-review segments
from `thor_new4_20260831_processed`, multiplied by seven offline visualization
methods. Every method draws a pose on every RGB frame. There is no validity-gate
disappearance, stale-color flash, joint-error connector, or per-frame error text.

The four published segments are Take_005, the disjoint first and second halves
of Take_006, and Take_007. They come from the three recordings that contain the
complete RGB + solved 20-joint hand + CMM marker stack. The fourth recording,
`camera_glove_recording_20260831_155410`, has 98 RGB/depth calibration frames but
zero glove packets, no solved hand pose, and no CMM action target; it is recorded
in `manifest.json` as calibration evidence and is not mislabeled as a hand demo.

## Seven methods

1. `s2_continuous`: IMU solver bones resampled at RGB time by S2 SLERP.
2. `s2_gaussian`: IMU-articulation symmetric S2 Gaussian smoothing.
3. `kalman_rts`: IMU-articulation forward Kalman plus backward RTS smoothing.
4. `posterior_75`: MOCAP direction plus 25% smoothed IMU residual.
5. `posterior_98`: MOCAP direction plus 2% smoothed IMU residual.
6. `rbf_self_fit`: full-sequence IMU-pose/velocity to MOCAP RBF neural fit.
   All rendered frames are training frames and there is deliberately no test
   split, following the operator request.
7. `guided_ik`: CMM endpoint-constrained fixed-phalanx constant-curvature IK,
   with one temporally continuous bend plane and bounded joint flexion. This is
   the recommended visual upper bound.

## Interpretation boundary

The last four methods consume synchronized MOCAP targets. The RBF model is
trained and rendered on the same sequence. Those outputs demonstrate how well
the recording can be made to look after teacher conditioning; they are not an
independent IMU accuracy estimate, a held-out evaluation, or evidence of
generalization. All four segments use CMM glove-surface reflectors, not
anatomical joint centers. The Take_005/006/007 marker IDs come from the supplied
#1..#11 ID table and placement photographs. Only the Take_007 left #1
`11781 -> 12503` alias is inferred from trajectory continuity; the exact IDs and
palm-fit residuals are retained in every metrics file.

Open `index.html` through the local review server. `manifest.json`, all 28
method metrics, the 28 synchronized 3D motion assets, `validation.json`, and
`SHA256SUMS.txt` provide the reproducible audit trail. The interactive viewer
uses the selected MP4 as its 30 FPS clock and supports orbit, pan, zoom, timeline
seek, and view reset without an external runtime or CDN. The earlier unmodified comparison remains in
`../imu_mocap_comparison_delivery/` and is not overwritten.
""",
        encoding="utf-8",
    )


def _write_webpage_2d_legacy(
    destination: Path,
    entries: Sequence[Mapping[str, Any]],
    rankings: Sequence[Mapping[str, Any]],
) -> None:
    method_options = "".join(
        f'<option value="{method.id}"{" selected" if method.id == "guided_ik" else ""}>{method.short_label}</option>'
        for method in METHODS
    )
    take_options = "".join(
        f'<option value="{take_id}"{" selected" if take_id == "take007" else ""}>{label}</option>'
        for take_id, label in (
            ("take005", "Take_005 · CMM"),
            ("take006a", "Take_006 · A · CMM"),
            ("take006b", "Take_006 · B · CMM"),
            ("take007", "Take_007 · CMM"),
        )
    )
    ranking_rows = "".join(
        f"<tr><td>{item['rank']}</td><td>{item['method_label']}</td>"
        f"<td><span class=\"badge {item['supervision']}\">{item['supervision']}</span></td>"
        f"<td>{item['mean_take_reference_median_mm']:.3f}</td>"
        f"<td>{item['mean_take_reference_p95_mm']:.3f}</td>"
        f"<td>{item['mean_take_jitter_p95_mm']:.3f}</td></tr>"
        for item in rankings
    )
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#06121e"><title>IMU -> MOCAP Visual Lab</title><link rel="stylesheet" href="styles.css"></head>
    <body><header class="top"><a href="/">← Dataset 首页</a><nav><a href="README.md">数据说明</a><a href="manifest.json">manifest</a></nav></header>
<main><section class="hero"><p>4 TAKES × 7 METHODS × 100% CONTINUOUS</p><h1>IMU → MOCAP<br><em>可视化方案实验室</em></h1>
<div class="notice"><b>默认推荐：Anatomical IK。</b> 它固定骨长、限制关节角并锁定连续弯曲平面；新版没有闪烁、stale 变色或误差连线。空心粗骨架是 MOCAP，实心细骨架是当前方法结果。</div></section>
<section class="lab"><div class="controls"><label>视频<select id="take">{take_options}</select></label><label>方案<select id="method">{method_options}</select></label></div>
<div class="stage"><video id="player" controls playsinline preload="metadata"></video><div class="meta"><p id="kind"></p><h2 id="title"></h2><div class="numbers"><div><b id="median"></b><span>reference median</span></div><div><b id="p95"></b><span>reference P95</span></div><div><b id="jitter"></b><span>jitter P95</span></div></div><nav><a id="download" download>下载当前 MP4</a><a id="metrics">查看 metrics</a></nav></div></div></section>
<section class="explain"><article><h2>三种 IMU-only</h2><p>S2 continuous、零相位 S2 Gaussian 和 Kalman RTS 不读取 MOCAP 手指关节，只使用既有的 MOCAP 腕部平移条件。</p></article><article><h2>四种“尽量贴 GT”</h2><p>75%/98% posterior、同段 RBF neural self-fit 和 guided IK 都读取了本段 MOCAP。它们是展示优化，不是独立 IMU accuracy。</p></article></section>
<section class="ranking"><h2>同一标尺下的可视化排序</h2><p>四段 primary reference 的平均值；旧三段用 16 个无歧义非拇指 BVH 关节，Take_007 用 10 个 CMM surface-marker 对应点。</p><div class="table"><table><thead><tr><th>#</th><th>method</th><th>supervision</th><th>median mm</th><th>P95 mm</th><th>jitter P95 mm</th></tr></thead><tbody>{ranking_rows}</tbody></table></div></section>
<section class="boundary"><h2>必须一起看的边界</h2><p>RBF 使用本段全部帧训练并在同一段上渲染，测试集为 0；MOCAP-conditioned 方法也直接读取 GT。页面显示的是“这段数据最多能被优化到多贴”，不能用来宣称 IMU 在新动作上的泛化精度。原始未平滑定量比较仍保留在独立页面。</p><a href="README.md">完整 README</a></section></main>
<footer>GT_CALIB · offline visualization lab · no flicker · no connectors</footer>
<script src="app.js" defer></script></body></html>"""
    styles = """:root{color-scheme:dark;--bg:#06121e;--panel:#0d2132;--line:#254158;--ink:#edf5fb;--muted:#91a9ba;--cyan:#56e0f1;--green:#5cf0a8;--amber:#ffc15a}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 82% 0,#123a50 0,transparent 35rem),var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,"PingFang SC",sans-serif}.top{position:sticky;top:0;z-index:5;display:flex;justify-content:space-between;padding:17px max(22px,calc((100vw - 1240px)/2));border-bottom:1px solid var(--line);background:#06121ee8;backdrop-filter:blur(14px)}.top nav{display:flex;gap:20px}a{color:var(--cyan);font-weight:800;text-decoration:none}.hero,.lab,.explain,.ranking,.boundary{width:min(1240px,calc(100% - 40px));margin:auto}.hero{padding:75px 0 48px}.hero>p{color:var(--cyan);font-size:.75rem;font-weight:950;letter-spacing:.18em}.hero h1{margin:12px 0 28px;font-size:clamp(3rem,7vw,6.2rem);line-height:.94;letter-spacing:-.06em}.hero em{color:var(--green);font-style:normal}.notice,.boundary{border:1px solid var(--line);border-radius:17px;background:#0a1b2abf;padding:22px 25px;color:var(--muted)}.notice b{color:var(--ink)}.controls{display:flex;gap:16px;margin-bottom:15px}.controls label{display:flex;align-items:center;gap:10px;color:var(--muted);font-size:.8rem;font-weight:800}select{min-width:180px;border:1px solid var(--line);border-radius:10px;background:var(--panel);color:var(--ink);padding:11px}.stage{display:grid;grid-template-columns:minmax(0,2fr) minmax(280px,1fr);overflow:hidden;border:1px solid var(--line);border-radius:19px;background:var(--panel)}video{display:block;width:100%;aspect-ratio:16/9;background:#000}.meta{padding:27px}.meta>p{color:var(--cyan);font-size:.72rem;font-weight:900;text-transform:uppercase}.meta h2{min-height:3.2em}.numbers{display:grid;gap:1px;background:var(--line);border:1px solid var(--line);border-radius:10px;overflow:hidden}.numbers div{padding:13px;background:#091927}.numbers b,.numbers span{display:block}.numbers span{color:var(--muted);font-size:.7rem}.meta nav{display:flex;gap:17px;margin-top:22px}.explain{display:grid;grid-template-columns:1fr 1fr;gap:20px;padding:42px 0}.explain article,.ranking{border:1px solid var(--line);border-radius:17px;background:#0a1b2a;padding:22px}.explain h2,.ranking h2{margin-top:0}.explain p,.ranking p,.boundary p{color:var(--muted);line-height:1.65}.table{overflow:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{padding:12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}.badge{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:4px 8px;font-size:.67rem}.imu_only{color:var(--green)}.mocap_conditioned,.mocap_conditioned_oracle,.same_sequence_teacher_fit{color:var(--amber)}.boundary{margin-top:38px;margin-bottom:70px}footer{padding:28px;border-top:1px solid var(--line);color:var(--muted);text-align:center;font-size:.76rem}@media(max-width:850px){.stage{grid-template-columns:1fr}.controls{align-items:stretch;flex-direction:column}.controls label{justify-content:space-between}.explain{grid-template-columns:1fr}.top nav{gap:10px;font-size:.75rem}}"""
    (destination / "index.html").write_text(html, encoding="utf-8")
    (destination / "styles.css").write_text(styles, encoding="utf-8")
    app = """'use strict';
let rows = [];
const take = document.querySelector('#take');
const method = document.querySelector('#method');
const player = document.querySelector('#player');
function update() {
  const row = rows.find((item) => item.take_id === take.value && item.method_id === method.value);
  if (!row) return;
  player.poster = row.poster;
  player.src = `videos/${row.filename}`;
  document.querySelector('#kind').textContent = row.supervision;
  document.querySelector('#title').textContent = `${row.take_label} · ${row.method_label}`;
  document.querySelector('#median').textContent = `${row.summary.reference_median_mm.toFixed(3)} mm`;
  document.querySelector('#p95').textContent = `${row.summary.reference_p95_mm.toFixed(3)} mm`;
  document.querySelector('#jitter').textContent = `${row.summary.jitter_p95_mm.toFixed(3)} mm`;
  document.querySelector('#download').href = `videos/${row.filename}`;
  document.querySelector('#metrics').href = row.metrics;
}
take.addEventListener('change', update);
method.addEventListener('change', update);
fetch('manifest.json')
  .then((response) => {
    if (!response.ok) throw new Error(`manifest HTTP ${response.status}`);
    return response.json();
  })
  .then((manifest) => {
    rows = manifest.videos;
    update();
  })
  .catch((error) => {
    document.querySelector('#title').textContent = `Failed to load manifest: ${error}`;
  });
"""
    (destination / "app.js").write_text(app, encoding="utf-8")


def _write_webpage(
    destination: Path,
    entries: Sequence[Mapping[str, Any]],
    rankings: Sequence[Mapping[str, Any]],
) -> None:
    method_options = "".join(
        f'<option value="{method.id}"{" selected" if method.id == "guided_ik" else ""}>{method.short_label}</option>'
        for method in METHODS
    )
    take_options = "".join(
        f'<option value="{take_id}"{" selected" if take_id == "take007" else ""}>{label}</option>'
        for take_id, label in (
            ("take005", "Take_005 · CMM"),
            ("take006a", "Take_006 · A · CMM"),
            ("take006b", "Take_006 · B · CMM"),
            ("take007", "Take_007 · CMM"),
        )
    )
    ranking_rows = "".join(
        f"<tr><td>{item['rank']}</td><td>{item['method_label']}</td>"
        f"<td><span class=\"badge {item['supervision']}\">{item['supervision']}</span></td>"
        f"<td>{item['mean_take_reference_median_mm']:.3f}</td>"
        f"<td>{item['mean_take_reference_p95_mm']:.3f}</td>"
        f"<td>{item['mean_take_jitter_p95_mm']:.3f}</td></tr>"
        for item in rankings
    )
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#06121e"><meta name="robots" content="noindex,noarchive">
<title>IMU → MOCAP Visual Lab</title><link rel="stylesheet" href="styles.css"><script src="app.js" defer></script></head>
    <body><header class="top"><a href="/">← Dataset 首页</a><nav><a href="README.md">数据说明</a><a href="manifest.json">manifest</a></nav></header>
<main><section class="hero"><p>4 TAKES × 7 METHODS × VIDEO-SYNCED 3D</p><h1>IMU → MOCAP<br><em>可视化方案实验室</em></h1>
    <div class="notice"><b>本页只使用 thor_new4_20260831_processed。</b> 默认推荐 Anatomical IK：固定骨长、PIP≤110°、DIP≤75°、thumb≤115°，并锁定连续弯曲平面；没有闪烁、stale 变色或误差连线。视频与交互 3D 共用同一 30 FPS 时间轴。</div></section>
<section class="lab"><div class="controls"><label>视频<select id="take">{take_options}</select></label><label>方案<select id="method">{method_options}</select></label><span id="load-state" role="status">正在读取 manifest…</span></div>
<div class="review-grid"><article class="media-card"><header><div><p>RGB · MASTER CLOCK</p><h2>视频叠加</h2></div><span>30 FPS</span></header><video id="player" controls playsinline preload="metadata"></video></article>
<article class="viewer-card"><header><div><p>INTERACTIVE ORTHOGRAPHIC 3D</p><h2>MOCAP + 当前方法</h2></div><div class="view-tools" role="group" aria-label="3D interaction mode"><button id="orbit-mode" class="active" type="button">旋转</button><button id="pan-mode" type="button">平移</button><button id="reset-view" type="button">重置</button></div></header><div class="canvas-wrap"><canvas id="motion-canvas" aria-label="交互三维双手骨架。选择旋转或平移后拖动，滚轮缩放，双击复位。"></canvas><div class="hud"><span id="frame-readout">FRAME —</span><span id="view-readout">3D —</span></div><div class="hint">拖动旋转/平移 · Shift/右键临时平移 · 滚轮缩放 · 双击复位</div></div><div class="timeline-row"><input id="timeline" type="range" min="0" max="1" value="0" step="1" disabled><span id="time-readout">00:00.000</span></div><div class="legend"><span><i class="mocap-left"></i>MOCAP L</span><span><i class="mocap-right"></i>MOCAP R</span><span><i class="imu-left"></i>RESULT L</span><span><i class="imu-right"></i>RESULT R</span></div></article></div>
<div class="meta"><div><p id="kind"></p><h2 id="title"></h2></div><div class="numbers"><div><b id="median"></b><span>reference median</span></div><div><b id="p95"></b><span>reference P95</span></div><div><b id="jitter"></b><span>jitter P95</span></div></div><nav><a id="download" download>下载当前 MP4</a><a id="motion-download" download>下载 3D motion</a><a id="metrics">查看 metrics</a></nav></div></section>
<section class="explain"><article><h2>三种 IMU articulation</h2><p>S² continuous、零相位 S² Gaussian 和 Kalman RTS 不读取逐帧 MOCAP 手指目标；但空间配准与腕部位置来自 MOCAP，因此不称为纯 IMU。</p></article><article><h2>四种“尽量贴 GT”</h2><p>75%/98% posterior、同段 RBF neural self-fit 和 guided IK 都读取本段 MOCAP。它们是展示优化，不是独立 IMU accuracy。</p></article></section>
    <section class="ranking"><h2>同一标尺下的可视化排序</h2><p>四段都使用同一新数据管线，并以左右手各 10 个 CMM surface-marker 对应点计分。Take_006 A/B 是同一录制的两个不重叠窗口；监督强度不同，低误差不等于公平 IMU benchmark。</p><div class="table"><table><thead><tr><th>#</th><th>method</th><th>supervision</th><th>median mm</th><th>P95 mm</th><th>jitter P95 mm</th></tr></thead><tbody>{ranking_rows}</tbody></table></div></section>
    <section class="boundary"><h2>必须一起看的边界</h2><p>Take_005、Take_006 和 Take_007 是三段完整 RGB + 20J hand solver + CMM 动作录制；Take_006 被无重叠地切成 A/B，组成四段审阅视频。155410 没有 glove/CMM 动作数据，只作为 calibration evidence。RBF 使用本段全部帧训练并在同一段上渲染，测试集为 0；MOCAP-conditioned 方法也直接读取 GT。</p><a href="README.md">完整 README</a></section></main>
<footer>GT_CALIB · video-synchronized 3D · no CDN · no flicker · no connectors</footer></body></html>"""
    styles = """:root{color-scheme:dark;--bg:#06121e;--panel:#0d2132;--line:#254158;--ink:#edf5fb;--muted:#91a9ba;--cyan:#56e0f1;--green:#5cf0a8;--amber:#ffc15a;--magenta:#f06be6}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 82% 0,#123a50 0,transparent 35rem),var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,"PingFang SC",sans-serif}.top{position:sticky;top:0;z-index:5;display:flex;justify-content:space-between;padding:17px max(22px,calc((100vw - 1380px)/2));border-bottom:1px solid var(--line);background:#06121ee8;backdrop-filter:blur(14px)}.top nav{display:flex;gap:20px}a{color:var(--cyan);font-weight:800;text-decoration:none}.hero,.lab,.explain,.ranking,.boundary{width:min(1380px,calc(100% - 40px));margin:auto}.hero{padding:70px 0 44px}.hero>p,.media-card header p,.viewer-card header p{color:var(--cyan);font-size:.72rem;font-weight:950;letter-spacing:.16em}.hero h1{margin:12px 0 28px;font-size:clamp(3rem,7vw,6.2rem);line-height:.94;letter-spacing:-.06em}.hero em{color:var(--green);font-style:normal}.notice,.boundary{border:1px solid var(--line);border-radius:17px;background:#0a1b2abf;padding:22px 25px;color:var(--muted)}.notice b{color:var(--ink)}.controls{display:flex;gap:16px;align-items:center;margin-bottom:15px}.controls label{display:flex;align-items:center;gap:10px;color:var(--muted);font-size:.8rem;font-weight:800}.controls>span{margin-left:auto;color:var(--muted);font-size:.76rem}select,button{border:1px solid var(--line);border-radius:10px;background:var(--panel);color:var(--ink);padding:10px 13px;font:inherit}select{min-width:180px}.view-tools{display:flex;gap:5px}.view-tools button{padding:7px 9px;font-size:.7rem}.view-tools button.active{border-color:var(--cyan);background:#12384a;color:var(--cyan)}.review-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.media-card,.viewer-card,.meta,.explain article,.ranking{overflow:hidden;border:1px solid var(--line);border-radius:18px;background:var(--panel)}.media-card header,.viewer-card header{display:flex;align-items:center;justify-content:space-between;height:78px;padding:15px 20px}.media-card header p,.viewer-card header p{margin:0}.media-card header h2,.viewer-card header h2{margin:4px 0 0;font-size:1.05rem}.media-card header>span{color:var(--muted);font-size:.75rem}.media-card video{display:block;width:100%;aspect-ratio:16/9;background:#000}.canvas-wrap{position:relative;aspect-ratio:16/9;background:radial-gradient(circle at 50% 40%,#112b3e,#050c14 72%);overflow:hidden}.canvas-wrap canvas{display:block;width:100%;height:100%;touch-action:none;cursor:grab}.canvas-wrap canvas[data-dragging=true]{cursor:grabbing}.hud{position:absolute;top:12px;left:12px;right:12px;display:flex;justify-content:space-between;pointer-events:none;color:#c7d8e5;font:700 .67rem ui-monospace,monospace}.hint{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);white-space:nowrap;color:#8ba4b8;font-size:.66rem;pointer-events:none}.timeline-row{display:flex;gap:12px;align-items:center;padding:12px 16px;border-top:1px solid var(--line)}.timeline-row input{width:100%;accent-color:var(--cyan)}.timeline-row span{min-width:72px;color:var(--muted);font:700 .7rem ui-monospace,monospace}.legend{display:flex;gap:14px;flex-wrap:wrap;padding:0 16px 13px;color:var(--muted);font-size:.67rem}.legend span{display:flex;gap:6px;align-items:center}.legend i{width:9px;height:9px;border-radius:50%}.mocap-left{background:var(--cyan)}.mocap-right{background:var(--magenta)}.imu-left{background:var(--green)}.imu-right{background:var(--amber)}.meta{display:grid;grid-template-columns:1.2fr 1fr auto;gap:25px;align-items:center;margin-top:18px;padding:22px}.meta p{margin:0;color:var(--cyan);font-size:.7rem;font-weight:900;text-transform:uppercase}.meta h2{margin:5px 0 0;font-size:1.1rem}.numbers{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:10px;overflow:hidden}.numbers div{padding:11px;background:#091927}.numbers b,.numbers span{display:block}.numbers span{color:var(--muted);font-size:.66rem}.meta nav{display:flex;flex-direction:column;gap:7px;font-size:.76rem}.explain{display:grid;grid-template-columns:1fr 1fr;gap:20px;padding:42px 0}.explain article,.ranking{padding:22px}.explain h2,.ranking h2{margin-top:0}.explain p,.ranking p,.boundary p{color:var(--muted);line-height:1.65}.table{overflow:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{padding:12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}.badge{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:4px 8px;font-size:.67rem}.imu_only{color:var(--green)}.mocap_conditioned,.mocap_conditioned_oracle,.same_sequence_teacher_fit{color:var(--amber)}.boundary{margin-top:38px;margin-bottom:70px}footer{padding:28px;border-top:1px solid var(--line);color:var(--muted);text-align:center;font-size:.76rem}@media(max-width:1000px){.review-grid{grid-template-columns:1fr}.meta{grid-template-columns:1fr}.meta nav{flex-direction:row}.controls{align-items:stretch;flex-direction:column}.controls label{justify-content:space-between}.controls>span{margin-left:0}.explain{grid-template-columns:1fr}}@media(max-width:600px){.numbers{grid-template-columns:1fr}.top nav{gap:10px;font-size:.7rem}.hint{font-size:.55rem}}"""
    app = r"""'use strict';
const COLORS = Object.freeze({
  'mocap-left': '#42deef', 'mocap-right': '#f06be6',
  'imu-left': '#67efa5', 'imu-right': '#ffc15a',
  outline: 'rgba(2,7,13,.88)', grid: 'rgba(118,165,198,.11)', gridStrong: 'rgba(118,185,215,.22)'
});
const DEFAULT_VIEW = Object.freeze({yaw:-0.48,pitch:-0.56,zoom:1,panX:0,panY:0});
const MOTION_LAYERS = Object.freeze([{"id":"mocap-left","kind":"mocap","side":"left","joint_offset":0,"joint_count":11,"chains":[[0,2,1],[0,4,3],[0,6,5],[0,8,7],[0,10,9]]},{"id":"mocap-right","kind":"mocap","side":"right","joint_offset":11,"joint_count":11,"chains":[[0,2,1],[0,4,3],[0,6,5],[0,8,7],[0,10,9]]},{"id":"imu-left","kind":"result","side":"left","joint_offset":22,"joint_count":20,"chains":[[0,1,2,3],[0,4,5,6,7],[0,8,9,10,11],[0,12,13,14,15],[0,16,17,18,19]]},{"id":"imu-right","kind":"result","side":"right","joint_offset":42,"joint_count":20,"chains":[[0,1,2,3],[0,4,5,6,7],[0,8,9,10,11],[0,12,13,14,15],[0,16,17,18,19]]}]);
const state = {rows:[],row:null,motion:null,generation:0,frame:0,buffer:null,view:{...DEFAULT_VIEW},tool:'orbit',drag:null,abort:null};
const dom = {take:document.querySelector('#take'),method:document.querySelector('#method'),player:document.querySelector('#player'),canvas:document.querySelector('#motion-canvas'),timeline:document.querySelector('#timeline'),orbit:document.querySelector('#orbit-mode'),pan:document.querySelector('#pan-mode'),reset:document.querySelector('#reset-view'),load:document.querySelector('#load-state'),frame:document.querySelector('#frame-readout'),view:document.querySelector('#view-readout'),time:document.querySelector('#time-readout')};
const ctx = dom.canvas.getContext('2d',{alpha:false});
const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
function hex(buffer){return Array.from(new Uint8Array(buffer),b=>b.toString(16).padStart(2,'0')).join('');}
async function fetchJson(url,sha,signal){const response=await fetch(url,{cache:'no-store',signal});if(!response.ok)throw new Error(`${url} HTTP ${response.status}`);const bytes=await response.arrayBuffer();if(sha&&globalThis.crypto?.subtle){const digest=await crypto.subtle.digest('SHA-256',bytes);if(hex(digest)!==sha)throw new Error(`${url} SHA mismatch`);}return JSON.parse(new TextDecoder().decode(bytes));}
function validateMotion(m,row){if(m?.schema!=='gt_calib.imu_mocap_visualization_motion.v1'||m.take_id!==row.take_id||m.method_id!==row.method_id||m.frame_count!==row.frame_count||m.fps!==30||m.units!=='mm'||!Array.isArray(m.layers)||JSON.stringify(m.layers)!==JSON.stringify(MOTION_LAYERS)||!Array.isArray(m.frames)||m.frames.length!==row.frame_count)throw new Error('3D motion contract mismatch');const e=m.encoding;if(e?.kind!=='frame-major-flat-int32-json'||e.components!=='XYZ'||!Number.isFinite(e.quantum_mm)||e.quantum_mm<=0||!Array.isArray(e.origin_mm)||e.origin_mm.length!==3||e.joint_count!==62||e.values_per_frame!==186||!m.frames.every(frame=>Array.isArray(frame)&&frame.length===186&&frame.every(Number.isInteger)))throw new Error('3D encoding mismatch');}
function selectedRow(){return state.rows.find(r=>r.take_id===dom.take.value&&r.method_id===dom.method.value);}
async function update(){const row=selectedRow();if(!row)return;const previous=state.row,desiredFrame=previous?.take_id===row.take_id?state.frame:0,resume=previous!==null&&!dom.player.paused&&!dom.player.ended,generation=++state.generation;if(state.abort)state.abort.abort();state.abort=new AbortController();state.row=row;state.motion=null;state.buffer=null;dom.load.textContent='正在校验 3D motion…';dom.timeline.disabled=true;dom.player.pause();clearCanvas();dom.player.poster=row.poster;dom.player.src=`videos/${row.filename}`;document.querySelector('#kind').textContent=row.supervision==='imu_only'?'IMU articulation · MOCAP-calibrated':row.supervision;document.querySelector('#title').textContent=`${row.take_label} · ${row.method_label}`;document.querySelector('#median').textContent=`${row.summary.reference_median_mm.toFixed(3)} mm`;document.querySelector('#p95').textContent=`${row.summary.reference_p95_mm.toFixed(3)} mm`;document.querySelector('#jitter').textContent=`${row.summary.jitter_p95_mm.toFixed(3)} mm`;document.querySelector('#download').href=`videos/${row.filename}`;document.querySelector('#motion-download').href=row.motion;document.querySelector('#metrics').href=row.metrics;try{const motion=await fetchJson(row.motion,row.motion_sha256,state.abort.signal);if(generation!==state.generation)return;validateMotion(motion,row);state.motion=motion;state.buffer=new Float32Array(motion.encoding.joint_count*3);state.frame=clamp(desiredFrame,0,row.frame_count-1);dom.timeline.max=String(row.frame_count-1);dom.timeline.value=String(state.frame);dom.timeline.disabled=false;if(!previous||previous.take_id!==row.take_id)state.view={...DEFAULT_VIEW};drawFrame(state.frame);const seek=()=>{if(generation!==state.generation)return;dom.player.currentTime=state.frame/state.motion.fps;if(resume)dom.player.play().catch(()=>{});};if(dom.player.readyState>=1)seek();else dom.player.addEventListener('loadedmetadata',seek,{once:true});dom.load.textContent=`3D ready · ${motion.encoding.joint_count} joints · ${row.frame_count} frames`;armVideoClock(generation);}catch(error){if(generation!==state.generation)return;console.error(error);clearCanvas();dom.load.textContent=`3D 加载失败：${error.message}`;}}
function decodeFrame(index){const m=state.motion;const frame=m.frames[index];const q=m.encoding.quantum_mm,o=m.encoding.origin_mm,b=state.buffer;for(let i=0;i<b.length;i++)b[i]=o[i%3]+frame[i]*q;return b;}
function ensureResolution(){const rect=dom.canvas.getBoundingClientRect(),dpr=Math.min(2,devicePixelRatio||1),w=Math.max(1,Math.round(rect.width*dpr)),h=Math.max(1,Math.round(rect.height*dpr));if(dom.canvas.width!==w||dom.canvas.height!==h){dom.canvas.width=w;dom.canvas.height=h;}return {w,h,dpr};}
function clearCanvas(){const {w,h}=ensureResolution();ctx.fillStyle='#050c14';ctx.fillRect(0,0,w,h);dom.frame.textContent='FRAME —';dom.view.textContent='3D —';}
function layerById(id){return state.motion.layers.find(layer=>layer.id===id);}
function joint(buffer,index){const i=index*3;return [buffer[i],buffer[i+1],buffer[i+2]];}
function drawFrame(requested){if(!state.motion)return;state.frame=clamp(Math.round(requested),0,state.motion.frame_count-1);decodeFrame(state.frame);dom.timeline.value=String(state.frame);dom.frame.textContent=`FRAME ${state.frame} / ${state.motion.frame_count-1}`;dom.time.textContent=formatTime(state.frame/state.motion.fps);drawScene();}
function formatTime(value){const minutes=Math.floor(value/60),seconds=value-minutes*60;return `${String(minutes).padStart(2,'0')}:${seconds.toFixed(3).padStart(6,'0')}`;}
function drawScene(){if(!state.motion)return;const {w,h}=ensureResolution(),b=state.buffer,left=layerById('mocap-left'),right=layerById('mocap-right'),a=joint(b,left.joint_offset),c=joint(b,right.joint_offset),center=[(a[0]+c[0])/2,(a[1]+c[1])/2,(a[2]+c[2])/2],extent=state.motion.view.reference_extent_mm,scale=Math.min(w,h)/(2*extent)*state.view.zoom;ctx.fillStyle='#050c14';ctx.fillRect(0,0,w,h);drawGrid(w,h);const cy=Math.cos(state.view.yaw),sy=Math.sin(state.view.yaw),cp=Math.cos(state.view.pitch),sp=Math.sin(state.view.pitch),projected=[];for(let i=0;i<b.length;i+=3){const x=b[i]-center[0],y=b[i+1]-center[1],z=b[i+2]-center[2],yx=cy*x+sy*z,yz=-sy*x+cy*z,py=cp*y-sp*yz,pz=sp*y+cp*yz;projected.push({x:w/2+yx*scale+state.view.panX,y:h/2-py*scale+state.view.panY,z:pz});}const bones=[];for(const layer of state.motion.layers){for(const chain of layer.chains){for(let i=0;i<chain.length-1;i++){const u=layer.joint_offset+chain[i],v=layer.joint_offset+chain[i+1];bones.push({u,v,id:layer.id,kind:layer.kind,depth:(projected[u].z+projected[v].z)/2});}}}bones.sort((x,y)=>x.depth-y.depth);ctx.lineCap='round';ctx.lineJoin='round';for(const bone of bones){const u=projected[bone.u],v=projected[bone.v];ctx.beginPath();ctx.moveTo(u.x,u.y);ctx.lineTo(v.x,v.y);ctx.lineWidth=bone.kind==='mocap'?7:6;ctx.strokeStyle=COLORS.outline;ctx.stroke();ctx.beginPath();ctx.moveTo(u.x,u.y);ctx.lineTo(v.x,v.y);ctx.lineWidth=bone.kind==='mocap'?2.2:3.2;ctx.strokeStyle=COLORS[bone.id];ctx.setLineDash(bone.kind==='mocap'?[6,3]:[]);ctx.stroke();ctx.setLineDash([]);}const nodes=[];for(const layer of state.motion.layers){for(let i=0;i<layer.joint_count;i++){const index=layer.joint_offset+i;nodes.push({index,id:layer.id,kind:layer.kind,depth:projected[index].z});}}nodes.sort((x,y)=>x.depth-y.depth);for(const node of nodes){const p=projected[node.index];ctx.beginPath();ctx.arc(p.x,p.y,node.kind==='mocap'?4:3.2,0,Math.PI*2);ctx.lineWidth=2;ctx.strokeStyle=COLORS[node.id];if(node.kind==='mocap'){ctx.fillStyle='#07111b';ctx.fill();ctx.stroke();}else{ctx.fillStyle=COLORS[node.id];ctx.fill();}}drawAxes(w,h,cy,sy,cp,sp);dom.view.textContent=`yaw ${(state.view.yaw*180/Math.PI).toFixed(0)}° · pitch ${(state.view.pitch*180/Math.PI).toFixed(0)}° · ${state.view.zoom.toFixed(2)}×`;}
function drawGrid(w,h){ctx.strokeStyle=COLORS.grid;ctx.lineWidth=1;const step=Math.max(28,Math.round(Math.min(w,h)/12));for(let x=w/2%step;x<w;x+=step){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}for(let y=h/2%step;y<h;y+=step){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}ctx.strokeStyle=COLORS.gridStrong;ctx.beginPath();ctx.moveTo(w/2,0);ctx.lineTo(w/2,h);ctx.moveTo(0,h/2);ctx.lineTo(w,h/2);ctx.stroke();}
function drawAxes(w,h,cy,sy,cp,sp){const ox=w-42,oy=h-39,len=22,axes=[['X',[1,0,0],'#ff7e89'],['Y',[0,1,0],'#7de7ad'],['Z',[0,0,1],'#79a9ff']];ctx.font='700 9px ui-monospace,monospace';for(const [label,[x,y,z],color] of axes){const yx=cy*x+sy*z,yz=-sy*x+cy*z,py=cp*y-sp*yz;ctx.beginPath();ctx.moveTo(ox,oy);ctx.lineTo(ox+yx*len,oy-py*len);ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.stroke();ctx.fillStyle=color;ctx.fillText(label,ox+yx*len+3,oy-py*len+3);}}
function resetView(){state.view={...DEFAULT_VIEW};drawScene();}
function setTool(tool){state.tool=tool;dom.orbit.classList.toggle('active',tool==='orbit');dom.pan.classList.toggle('active',tool==='pan');dom.orbit.setAttribute('aria-pressed',String(tool==='orbit'));dom.pan.setAttribute('aria-pressed',String(tool==='pan'));}
function syncVideo(){if(!state.motion||!Number.isFinite(dom.player.currentTime))return;drawFrame(Math.round(dom.player.currentTime*state.motion.fps));}
function armVideoClock(generation){if(!('requestVideoFrameCallback' in dom.player))return;dom.player.requestVideoFrameCallback((_now,metadata)=>{if(generation!==state.generation)return;drawFrame(Math.round(metadata.mediaTime*state.motion.fps));armVideoClock(generation);});}
dom.take.addEventListener('change',update);dom.method.addEventListener('change',update);dom.timeline.addEventListener('input',()=>{if(!state.motion)return;drawFrame(Number(dom.timeline.value));dom.player.currentTime=state.frame/state.motion.fps;});for(const event of ['loadeddata','seeked','timeupdate'])dom.player.addEventListener(event,syncVideo);dom.orbit.addEventListener('click',()=>setTool('orbit'));dom.pan.addEventListener('click',()=>setTool('pan'));dom.reset.addEventListener('click',resetView);dom.canvas.addEventListener('pointerdown',event=>{if(!state.motion)return;event.preventDefault();dom.canvas.setPointerCapture(event.pointerId);dom.canvas.dataset.dragging='true';state.drag={id:event.pointerId,x:event.clientX,y:event.clientY,mode:event.shiftKey||event.button!==0?'pan':state.tool};});dom.canvas.addEventListener('pointermove',event=>{if(!state.drag||state.drag.id!==event.pointerId)return;const dx=event.clientX-state.drag.x,dy=event.clientY-state.drag.y;state.drag.x=event.clientX;state.drag.y=event.clientY;if(state.drag.mode==='pan'){state.view.panX+=dx*(devicePixelRatio||1);state.view.panY+=dy*(devicePixelRatio||1);}else{state.view.yaw+=dx*.008;state.view.pitch=clamp(state.view.pitch+dy*.008,-1.48,1.48);}drawScene();});function endDrag(event){if(state.drag?.id===event.pointerId){state.drag=null;delete dom.canvas.dataset.dragging;}}dom.canvas.addEventListener('pointerup',endDrag);dom.canvas.addEventListener('pointercancel',endDrag);dom.canvas.addEventListener('contextmenu',event=>event.preventDefault());dom.canvas.addEventListener('wheel',event=>{event.preventDefault();state.view.zoom=clamp(state.view.zoom*Math.exp(-event.deltaY*.001),.25,6);drawScene();},{passive:false});dom.canvas.addEventListener('dblclick',resetView);new ResizeObserver(()=>{if(state.motion)drawScene();else clearCanvas();}).observe(dom.canvas);setTool('orbit');clearCanvas();fetch('manifest.json').then(response=>{if(!response.ok)throw new Error(`manifest HTTP ${response.status}`);return response.json();}).then(manifest=>{state.rows=manifest.videos;update();}).catch(error=>{console.error(error);dom.load.textContent=`加载失败：${error.message}`;});
"""
    (destination / "index.html").write_text(html, encoding="utf-8")
    (destination / "styles.css").write_text(styles, encoding="utf-8")
    (destination / "app.js").write_text(app, encoding="utf-8")


def _safe_relative(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"{label} must be a safe relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} escapes the delivery")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"{label} is not a regular delivery file")
    return path


def validate_visualization_lab(
    destination: Path,
    *,
    full_decode: bool = False,
) -> dict[str, Any]:
    root_input = Path(destination).expanduser()
    failures: list[str] = []
    if root_input.is_symlink():
        return {
            "schema": VALIDATION_SCHEMA,
            "status": "fail",
            "video_count": 0,
            "full_decode": full_decode,
            "failures": ["visualization lab root must not be a symlink"],
        }
    root = root_input.resolve()
    try:
        manifest = json.loads(
            _safe_relative(root, "manifest.json", label="manifest").read_text(encoding="utf-8")
        )
    except Exception as error:
        return {
            "schema": VALIDATION_SCHEMA,
            "status": "fail",
            "video_count": 0,
            "full_decode": full_decode,
            "failures": [str(error)],
        }
    videos = manifest.get("videos")
    if (
        manifest.get("schema") != DELIVERY_SCHEMA
        or manifest.get("status") != "pass"
        or manifest.get("take_count") != EXPECTED_TAKES
        or manifest.get("method_count") != EXPECTED_METHODS
        or manifest.get("video_count") != EXPECTED_VIDEOS
        or not isinstance(videos, list)
        or len(videos) != EXPECTED_VIDEOS
    ):
        failures.append("manifest must be a clean 4-take x 7-method pass")
        videos = videos if isinstance(videos, list) else []
    expected_scope = _expected_dataset_scope()
    if manifest.get("dataset_scope") != expected_scope:
        failures.append("dataset_scope must lock the canonical new-data source windows")
    expected_segments = {
        take_id: {
            "source": source,
            "source_first": source_first,
            "frame_count": frame_count,
        }
        for take_id, source, source_first, frame_count in EXPECTED_SEGMENTS
    }
    expected_display = {
        "pose_drawn_on_every_frame": True,
        "validity_gate_hides_pose": False,
        "stale_color_change": False,
        "connector_or_error_lines": False,
        "per_frame_error_text": False,
    }
    if manifest.get("display_policy") != expected_display:
        failures.append("no-flicker/no-connector display policy mismatch")
    manifest_methods = manifest.get("methods")
    if (
        not isinstance(manifest_methods, list)
        or [item.get("id") for item in manifest_methods if isinstance(item, Mapping)]
        != list(EXPECTED_METHOD_IDS)
        or len(manifest_methods) != EXPECTED_METHODS
    ):
        failures.append("manifest method identities/order do not match the canonical seven")

    orders: list[int] = []
    pairs: set[tuple[str, str]] = set()
    ordered_pairs: list[tuple[str, str]] = []
    reference_extents_by_take: dict[str, float] = {}
    video_receipts: list[dict[str, Any]] = []
    expected_assets = {
        "README.md", "index.html", "styles.css", "app.js", "manifest.json", "validation.json", "SHA256SUMS.txt"
    }
    for index, entry in enumerate(videos):
        try:
            if not isinstance(entry, Mapping):
                raise ValueError(f"videos[{index}] is not an object")
            if type(entry.get("order")) is not int:
                raise ValueError(f"videos[{index}].order must be an integer")
            orders.append(entry["order"])
            pair = (str(entry["take_id"]), str(entry["method_id"]))
            if pair in pairs:
                raise ValueError(f"duplicate take/method pair: {pair}")
            pairs.add(pair)
            ordered_pairs.append(pair)
            expected_stem = f"{index + 1:02d}_{pair[0]}_{pair[1]}"
            segment = expected_segments.get(pair[0])
            if (
                entry.get("id") != f"{pair[0]}-{pair[1]}"
                or entry.get("filename") != f"{expected_stem}.mp4"
                or entry.get("poster") != f"posters/{expected_stem}.jpg"
                or entry.get("metrics") != f"metrics/{expected_stem}.json"
                or entry.get("motion") != f"motions/{expected_stem}.json"
                or segment is None
                or entry.get("source") != segment["source"]
                or entry.get("source_first") != segment["source_first"]
                or entry.get("frame_count") != segment["frame_count"]
            ):
                raise ValueError(f"canonical asset identity mismatch for videos[{index}]")
            video_relative = f"videos/{entry['filename']}"
            video = _safe_relative(root, video_relative, label=f"video[{index}]")
            poster = _safe_relative(root, entry["poster"], label=f"poster[{index}]")
            metrics = _safe_relative(root, entry["metrics"], label=f"metrics[{index}]")
            motion = _safe_relative(root, entry["motion"], label=f"motion[{index}]")
            expected_assets.update(
                (video_relative, entry["poster"], entry["metrics"], entry["motion"])
            )
            for path, key in (
                (video, "sha256"),
                (poster, "poster_sha256"),
                (metrics, "metrics_sha256"),
                (motion, "motion_sha256"),
            ):
                if _sha256(path) != entry.get(key):
                    raise ValueError(f"{key} mismatch for {entry.get('id')}")
                if path.stat().st_size > MAX_CLOUDFLARE_ASSET_BYTES:
                    raise ValueError(f"Cloudflare asset limit exceeded: {path.name}")
            if motion.stat().st_size != entry.get("motion_bytes"):
                raise ValueError(f"motion_bytes mismatch for {entry.get('id')}")
            probe = _probe_video(video)
            for key in ("codec", "pixel_format", "width", "height", "frame_count", "bytes"):
                if probe[key] != entry.get(key):
                    raise ValueError(f"{key} mismatch for {entry.get('id')}")
            if (
                probe["codec"] != "h264"
                or probe["pixel_format"] != "yuv420p"
                or probe["width"] != 960
                or probe["height"] != 540
                or not np.isclose(probe["fps"], 30.0, atol=1e-9, rtol=0.0)
                or not entry.get("faststart")
                or not _faststart(video)
            ):
                raise ValueError(f"browser media contract failed for {entry.get('id')}")
            if full_decode:
                completed = subprocess.run(
                    ["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:v:0", "-f", "null", "-"],
                    capture_output=True,
                    text=True,
                )
                if completed.returncode != 0:
                    raise ValueError(f"full decode failed: {video.name}")
            video_receipts.append(
                {
                    "id": entry.get("id"),
                    "sha256": entry.get("sha256"),
                    "frame_count": probe["frame_count"],
                    "codec": probe["codec"],
                    "pixel_format": probe["pixel_format"],
                    "width": probe["width"],
                    "height": probe["height"],
                    "fps": probe["fps"],
                    "decoded": True,
                }
            )
            metrics_payload = json.loads(metrics.read_text(encoding="utf-8"))
            if (
                metrics_payload.get("schema") != METRICS_SCHEMA
                or metrics_payload.get("status") != "complete"
                or metrics_payload.get("take_id") != entry.get("take_id")
                or metrics_payload.get("method_id") != entry.get("method_id")
                or metrics_payload.get("source") != segment["source"]
                or metrics_payload.get("source_first") != segment["source_first"]
                or metrics_payload.get("rendered_frames") != probe["frame_count"]
                or metrics_payload.get("display_contract") != expected_display
            ):
                raise ValueError(f"metrics contract mismatch for {entry.get('id')}")
            source_assets = metrics_payload.get("source_assets")
            if (
                not isinstance(source_assets, Mapping)
                or source_assets.get("camera_intrinsics")
                != EXPECTED_CAMERA_INTRINSICS[pair[0]]
            ):
                raise ValueError(
                    f"action camera intrinsics provenance mismatch for {entry.get('id')}"
                )
            _validate_anatomical_ik_metrics_contract(
                metrics_payload,
                label=str(entry.get("id")),
            )
            primary = metrics_payload.get("pooled_sides", {}).get(
                "primary_reference_epe_mm", {}
            )
            jitter = metrics_payload.get("pooled_sides", {}).get(
                "root_relative_high_frequency_residual_mm", {}
            )
            expected_summary = {
                "reference_median_mm": primary.get("median"),
                "reference_p95_mm": primary.get("p95"),
                "jitter_p95_mm": jitter.get("p95"),
                "finite_frames": probe["frame_count"],
            }
            if entry.get("summary") != expected_summary:
                raise ValueError(f"summary/metrics mismatch for {entry.get('id')}")
            for side in ANATOMICAL_SIDES:
                side_payload = metrics_payload.get("sides", {}).get(side, {})
                if side_payload.get("finite_frames") != probe["frame_count"]:
                    raise ValueError(f"continuity contract failed for {entry.get('id')} {side}")
            motion_payload = json.loads(motion.read_text(encoding="utf-8"))
            encoding = motion_payload.get("encoding", {})
            layers = motion_payload.get("layers")
            frames = motion_payload.get("frames")
            quantum = encoding.get("quantum_mm")
            origin = encoding.get("origin_mm")
            joint_count = encoding.get("joint_count")
            values_per_frame = encoding.get("values_per_frame")
            expected_layers = _expected_motion_layers()
            maximum_error = motion_payload.get("validation", {}).get(
                "maximum_quantization_error_mm"
            )
            reference_extent = motion_payload.get("view", {}).get(
                "reference_extent_mm"
            )
            if (
                motion_payload.get("schema") != MOTION_SCHEMA
                or motion_payload.get("take_id") != entry.get("take_id")
                or motion_payload.get("method_id") != entry.get("method_id")
                or motion_payload.get("frame_count") != probe["frame_count"]
                or motion_payload.get("fps") != 30.0
                or motion_payload.get("units") != "mm"
                or encoding.get("kind") != "frame-major-flat-int32-json"
                or encoding.get("components") != "XYZ"
                or isinstance(quantum, bool)
                or not isinstance(quantum, (int, float))
                or not np.isfinite(quantum)
                or quantum <= 0.0
                or not isinstance(origin, list)
                or len(origin) != 3
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                    for value in origin
                )
                or joint_count != 62
                or type(values_per_frame) is not int
                or values_per_frame != joint_count * 3
                or layers != expected_layers
                or not isinstance(frames, list)
                or len(frames) != probe["frame_count"]
                or not all(
                    isinstance(frame, list)
                    and len(frame) == values_per_frame
                    and all(type(value) is int and -(2**31) <= value < 2**31 for value in frame)
                    for frame in frames
                )
                or motion_payload.get("validation", {}).get("status") != "pass"
                or motion_payload.get("validation", {}).get("finite_frames")
                != probe["frame_count"]
                or isinstance(maximum_error, bool)
                or not isinstance(maximum_error, (int, float))
                or not np.isfinite(maximum_error)
                or maximum_error < 0.0
                or maximum_error > quantum / 2.0 + 1e-9
                or motion_payload.get("view", {}).get("fixed_across_methods_for_take") is not True
                or motion_payload.get("view", {}).get("focus")
                != "per-frame midpoint of MOCAP left/right wrists"
                or isinstance(reference_extent, bool)
                or not isinstance(reference_extent, (int, float))
                or not np.isfinite(reference_extent)
                or reference_extent <= 0.0
            ):
                raise ValueError(f"3D motion contract mismatch for {entry.get('id')}")
            previous_extent = reference_extents_by_take.setdefault(
                str(entry.get("take_id")),
                float(reference_extent),
            )
            if previous_extent != float(reference_extent):
                raise ValueError(
                    f"3D view extent changes across methods for {entry.get('take_id')}"
                )
        except Exception as error:
            failures.append(str(error))
    expected_ordered_pairs = [
        (take_id, method_id)
        for take_id in EXPECTED_TAKE_IDS
        for method_id in EXPECTED_METHOD_IDS
    ]
    if (
        orders != list(range(1, EXPECTED_VIDEOS + 1))
        or ordered_pairs != expected_ordered_pairs
        or pairs != set(expected_ordered_pairs)
    ):
        failures.append("video order/method matrix must be the canonical 4 x 7 cartesian product")

    actual_assets: set[str] = set()
    if root.is_dir():
        for path in root.rglob("*"):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                failures.append(f"symlinks are forbidden: {relative}")
            elif path.is_dir():
                continue
            elif path.is_file():
                actual_assets.add(relative)
            else:
                failures.append(f"non-regular delivery entry: {relative}")
    if actual_assets != expected_assets:
        failures.append("delivery file inventory does not match manifest")

    checksum_path = root / "SHA256SUMS.txt"
    if checksum_path.is_file() and not checksum_path.is_symlink():
        declared: dict[str, str] = {}
        for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), start=1):
            match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
            if match is None:
                failures.append(f"malformed checksum line {line_number}")
                continue
            digest, relative = match.groups()
            if relative in declared:
                failures.append(f"duplicate checksum entry: {relative}")
            declared[relative] = digest
        checksum_assets = actual_assets - {"SHA256SUMS.txt"}
        if set(declared) != checksum_assets:
            failures.append("checksum inventory mismatch")
        else:
            for relative, digest in declared.items():
                if _sha256(root / relative) != digest:
                    failures.append(f"checksum mismatch: {relative}")
    else:
        failures.append("SHA256SUMS.txt is missing")
    manifest_sha256 = _sha256(root / "manifest.json")
    full_decode_receipt = {
        "schema": VALIDATION_SCHEMA,
        "status": "pass",
        "video_count": len(videos),
        "full_decode": True,
        "manifest_sha256": manifest_sha256,
        "videos": video_receipts,
        "failures": [],
    }
    receipt_verified = False
    if not full_decode:
        try:
            stored_receipt = json.loads(
                _safe_relative(root, "validation.json", label="validation receipt").read_text(
                    encoding="utf-8"
                )
            )
            receipt_verified = stored_receipt == full_decode_receipt
            if not receipt_verified:
                failures.append(
                    "validation receipt is stale or not bound to this manifest/video set"
                )
        except Exception as error:
            failures.append(f"validation receipt failed: {error}")
    if full_decode:
        return {
            **full_decode_receipt,
            "status": "pass" if not failures else "fail",
            "failures": failures,
        }
    return {
        "schema": VALIDATION_SCHEMA,
        "status": "pass" if not failures else "fail",
        "video_count": len(videos),
        "full_decode": False,
        "manifest_sha256": manifest_sha256,
        "receipt_verified": receipt_verified,
        "failures": failures,
    }


def build_visualization_lab(
    project_root: Path,
    destination: Path,
    *,
    manual_profile: Path | None = None,
) -> Path:
    project_root = Path(project_root).resolve()
    destination = Path(destination).resolve()
    profile_path = (
        Path(manual_profile).resolve()
        if manual_profile is not None
        else project_root / DEFAULT_MANUAL_PROFILE
    )
    final_profile = load_final_nine_profile(profile_path)
    if final_profile.schema != FINAL_NINE_PROFILE_SCHEMA:
        raise ValueError(
            "IMU visualization lab requires a final-nine manual profile; "
            "the legacy Take_007-only profile cannot be applied to Take_005/006"
        )
    staging = destination.with_name(destination.name + ".staging")
    if destination.exists() or staging.exists():
        raise FileExistsError(f"Refusing to overwrite visualization lab or staging: {destination}")
    for name in ("videos", "posters", "metrics", "motions"):
        (staging / name).mkdir(parents=True, exist_ok=True)
    try:
        print("[visual-lab] preparing four new-dataset CMM segments", flush=True)
        world_translation = np.asarray(
            final_profile.global_world_xyz_mm,
            dtype=np.float64,
        )
        take005 = _prepare_cmm_capture(
            project_root,
            recording="camera_glove_recording_20260831_161402",
            take="Take_005",
            order=1,
            take_id="take005",
            label="Take_005",
            world_translation_xyz_mm=world_translation,
        )
        print(f"[visual-lab] prepared take005: {take005.frame_count} frames", flush=True)
        take006 = _prepare_cmm_capture(
            project_root,
            recording="camera_glove_recording_20260831_161610",
            take="Take_006",
            order=2,
            take_id="take006",
            label="Take_006",
            world_translation_xyz_mm=world_translation,
        )
        split = take006.frame_count // 2
        take006a = _slice_lab_take(
            take006,
            start=0,
            stop=split,
            order=2,
            take_id="take006a",
            label="Take_006 · A",
        )
        take006b = _slice_lab_take(
            take006,
            start=split,
            stop=take006.frame_count,
            order=3,
            take_id="take006b",
            label="Take_006 · B",
        )
        print(
            f"[visual-lab] prepared take006a/b: {take006a.frame_count} + "
            f"{take006b.frame_count} disjoint frames",
            flush=True,
        )
        take007 = _prepare_cmm_capture(
            project_root,
            recording="camera_glove_recording_20260831_161912",
            take="Take_007",
            order=4,
            take_id="take007",
            label="Take_007",
            world_translation_xyz_mm=world_translation,
        )
        print(f"[visual-lab] prepared take007: {take007.frame_count} frames", flush=True)
        takes = [take005, take006a, take006b, take007]
        actual_segments = tuple(
            (take.id, take.source, take.source_first, take.frame_count)
            for take in takes
        )
        if actual_segments != EXPECTED_SEGMENTS:
            raise ValueError("Prepared CMM source windows do not match the delivery contract")

        entries: list[dict[str, Any]] = []
        for take in takes:
            print(f"[visual-lab] fitting seven methods for {take.id}", flush=True)
            poses, diagnostics = build_method_poses(take)
            print(f"[visual-lab] rendering and encoding seven videos for {take.id}", flush=True)
            entries.extend(_render_take_methods(take, staging, poses, diagnostics))
            print(f"[visual-lab] completed {take.id}", flush=True)
        for entry in entries:
            video = staging / "videos" / entry["filename"]
            poster = staging / entry["poster"]
            metrics = staging / entry["metrics"]
            probe = _probe_video(video)
            expected_frames = int(entry.pop("expected_frames"))
            if probe["frame_count"] != expected_frames:
                raise ValueError(f"Rendered frame count mismatch: {video}")
            if probe["fps"] != 30.0:
                raise ValueError(f"Rendered FPS mismatch: {video}")
            entry.update(probe)
            entry["sha256"] = _sha256(video)
            entry["poster_sha256"] = _sha256(poster)
            entry["metrics_sha256"] = _sha256(metrics)
            entry["faststart"] = _faststart(video)
            if video.stat().st_size > MAX_CLOUDFLARE_ASSET_BYTES:
                raise ValueError(f"Cloudflare asset limit exceeded: {video}")

        rankings = _method_rankings(entries)
        manifest = {
            "schema": DELIVERY_SCHEMA,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "pass",
            "take_count": EXPECTED_TAKES,
            "method_count": EXPECTED_METHODS,
            "video_count": EXPECTED_VIDEOS,
            "recommended_method_id": "guided_ik",
            "methods": [
                {
                    "id": method.id,
                    "label": method.label,
                    "short_label": method.short_label,
                    "supervision": method.supervision,
                    "description": method.description,
                    "boundary": method.boundary,
                }
                for method in METHODS
            ],
            "rankings": rankings,
            "videos": entries,
            "display_policy": {
                "pose_drawn_on_every_frame": True,
                "validity_gate_hides_pose": False,
                "stale_color_change": False,
                "connector_or_error_lines": False,
                "per_frame_error_text": False,
            },
            "interactive_3d": {
                "enabled": True,
                "schema": MOTION_SCHEMA,
                "clock": "selected 30 FPS video mediaTime",
                "layers": ["MOCAP left/right", "selected result left/right"],
                "projection": "orthographic pseudo-3D with orbit, pan, zoom, and reset",
                "external_runtime_or_cdn": False,
            },
            "dataset_scope": _expected_dataset_scope(),
            "calibration_only_recording": {
                "recording": "camera_glove_recording_20260831_155410",
                "role": "camera/depth calibration evidence only; no glove packets, solved hand pose, or CMM action target",
                "rgb_preview": _source_asset(
                    project_root,
                    project_root / "thor_new4_20260831_processed/camera_glove_recording_20260831_155410/rgbd_unpack/RGB.mp4",
                ),
                "depth_preview": _source_asset(
                    project_root,
                    project_root / "thor_new4_20260831_processed/camera_glove_recording_20260831_155410/rgbd_unpack/Depth.mp4",
                ),
                "camera_intrinsics": _source_asset(
                    project_root,
                    project_root / "thor_new4_20260831_processed/camera_glove_recording_20260831_155410/rgbd_unpack/camera_1_intrinsics.json",
                ),
                "glove_processing_report": _source_asset(
                    project_root,
                    project_root / "thor_new4_20260831_processed/camera_glove_recording_20260831_155410/glove_processing/processing_report.json",
                ),
            },
            "camera_world_calibration_source": _source_asset(
                project_root,
                project_root / "movementcap_20260831_worldcalib_tabletop_final.tar.gz",
            ),
            "manual_profile_source": _source_asset(project_root, profile_path),
            "canonical_final_nine_unchanged": True,
            "raw_comparison_unchanged": True,
            "claim_boundary": (
                "MOCAP-conditioned and same-sequence self-fit methods are visual optimization "
                "results, not held-out or independent IMU accuracy."
            ),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_readme(staging)
        _write_webpage(staging, entries, rankings)
        placeholder = {
            "schema": VALIDATION_SCHEMA,
            "status": "pass",
            "video_count": EXPECTED_VIDEOS,
            "full_decode": True,
            "failures": [],
        }
        (staging / "validation.json").write_text(
            json.dumps(placeholder, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_checksums(staging)
        print("[visual-lab] full-decoding all 28 H.264 videos", flush=True)
        validation = validate_visualization_lab(staging, full_decode=True)
        if validation["status"] != "pass":
            raise ValueError(f"Visualization lab full validation failed: {validation['failures']}")
        (staging / "validation.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_checksums(staging)
        final_validation = validate_visualization_lab(staging, full_decode=False)
        if final_validation["status"] != "pass":
            raise ValueError(f"Visualization lab final validation failed: {final_validation['failures']}")
        staging.rename(destination)
        print(f"[visual-lab] delivery complete: {destination}", flush=True)
        return destination
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
