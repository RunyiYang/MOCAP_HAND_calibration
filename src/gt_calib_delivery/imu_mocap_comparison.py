from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

# The validated legacy renderers remain project-root scripts.  The UV console
# entry point starts with only ``src`` on sys.path, so expose this checkout's
# root before importing their shared data-contract helpers.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import gt_calib_viz as viz
import mocap_video_overlay as mocap_overlay

from .delivery import OLD_TAKES
from .manual_profiles import load_final_nine_profile
from .new_capture import (
    ANATOMICAL_SIDES,
    BASE_MARKER_INDICES,
    GLOVE_CHAINS,
    MARKER_TO_GLOVE_INDICES,
    RAW_CHAINS,
    TIP_MARKER_INDICES,
    _contact_sheet,
    _transcode_h264,
    load_manual_calibration,
    load_new_capture,
    load_solved_pose,
    project_world,
    raw_mocap_nodes,
    virtual_wrist,
)


DELIVERY_SCHEMA = "gt_calib.imu_solved_pose_mocap_comparison_delivery.v1"
METRICS_SCHEMA_OLD = "gt_calib.imu_solved_pose_vs_skeleton_bvh.v1"
METRICS_SCHEMA_TAKE007 = "gt_calib.imu_solved_pose_vs_cmm_markers.v1"
VALIDATION_SCHEMA = "gt_calib.imu_mocap_comparison_validation.v1"
EXPECTED_COMPARISONS = 4
EXPECTED_VIDEO_CONTRACTS = (
    (
        "take01-imu-solved-vs-bvh",
        "01_take01_imu_solved_vs_bvh_mocap.mp4",
        1747,
        METRICS_SCHEMA_OLD,
    ),
    (
        "take02-imu-solved-vs-bvh",
        "02_take02_imu_solved_vs_bvh_mocap.mp4",
        1805,
        METRICS_SCHEMA_OLD,
    ),
    (
        "take03-imu-solved-vs-bvh",
        "03_take03_imu_solved_vs_bvh_mocap.mp4",
        1799,
        METRICS_SCHEMA_OLD,
    ),
    (
        "take007-imu-solved-vs-cmm",
        "04_take007_imu_solved_vs_cmm_mocap.mp4",
        1981,
        METRICS_SCHEMA_TAKE007,
    ),
)
OLD_REGISTRATION_PROFILE = (
    "calibration_profiles/imu_solver_vs_bvh_take01_registration.v1.json"
)
DEFAULT_MANUAL_PROFILE = (
    "calibration_profiles/operator_y_minus_44_all_hand_overlays.v1.json"
)

MOCAP_BGR = (25, 205, 255)
IMU_FRESH_BGR = (75, 245, 90)
IMU_STALE_BGR = (55, 165, 255)
CONNECTOR_BGR = (255, 145, 45)
TAKE007_MOCAP_BGR = {"left": (255, 220, 45), "right": (245, 55, 230)}
TAKE007_IMU_BGR = {"left": (80, 245, 95), "right": (55, 225, 255)}

TAKE007_BASE_MARKER_INDICES = np.asarray((3, 5, 7, 9), dtype=np.int32)
TAKE007_BASE_GLOVE_INDICES = np.asarray((4, 8, 12, 16), dtype=np.int32)
TAKE007_CALIBRATION_FRACTION = 0.20

NON_THUMB_NAMES = tuple(viz.GLOVE_NAMES[index] for index in viz.GLOVE_COMMON_INDICES)


@dataclass(frozen=True)
class NearestSamples:
    values: np.ndarray
    indices: np.ndarray
    age_ms: np.ndarray


@dataclass(frozen=True)
class FixedSimilarity:
    scale: float
    rotation: np.ndarray
    input_frame_count: int
    retained_frame_count: int
    residual_median_mm: float
    residual_p95_mm: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _project_relative(project_root: Path, path: Path) -> str:
    return Path(path).resolve().relative_to(Path(project_root).resolve()).as_posix()


def _source_asset(project_root: Path, path: Path) -> dict[str, Any]:
    source = Path(path).resolve()
    return {
        "path": _project_relative(project_root, source),
        "bytes": source.stat().st_size,
        "sha256": _sha256(source),
    }


def nearest_solver_samples(
    source_times_s: np.ndarray,
    values: np.ndarray,
    target_times_s: np.ndarray,
) -> NearestSamples:
    """Select an observed solver row for every target without smoothing.

    This is deliberately a display sampler, not an interpolation or a validity
    test.  Scientific masks remain separate and can exclude stale selections.
    """

    source = np.asarray(source_times_s, dtype=np.float64)
    target = np.asarray(target_times_s, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("Solver and target timestamps must be one-dimensional")
    if len(source) < 1 or len(data) != len(source):
        raise ValueError("Solver values do not match the source timestamps")
    if np.any(~np.isfinite(source)) or np.any(np.diff(source) <= 0.0):
        raise ValueError("Solver timestamps must be finite and strictly increasing")
    if np.any(~np.isfinite(target)):
        raise ValueError("Target timestamps must be finite")
    if target[0] < source[0] or target[-1] > source[-1]:
        raise ValueError("Target video interval extends beyond solver support")

    insertion = np.searchsorted(source, target, side="left")
    upper = np.clip(insertion, 0, len(source) - 1)
    lower = np.clip(insertion - 1, 0, len(source) - 1)
    choose_lower = np.abs(target - source[lower]) <= np.abs(source[upper] - target)
    indices = np.where(choose_lower, lower, upper).astype(np.int64, copy=False)
    selected = data[indices]
    if np.any(~np.isfinite(selected)):
        raise ValueError("Nearest solver rows contain non-finite pose values")
    return NearestSamples(
        values=selected,
        indices=indices,
        age_ms=np.abs(target - source[indices]) * 1000.0,
    )


def fit_take01_bvh_registration(
    project_root: Path,
) -> tuple[str, dict[str, viz.SimilarityRegistration], dict[str, Any]]:
    """Fit the frozen old-take coordinate registration against BVH FK.

    This deliberately does not reuse :func:`viz.prepare_take`'s default palm
    fit because that legacy path targets ``Human.cma`` joint positions.  The
    comparison supplement uses the latest ``Skeleton_0/1.bvh`` FK positions as
    both its registration target and its evaluation reference.
    """

    project_root = Path(project_root).resolve()
    take = OLD_TAKES[0]
    dataset = project_root / "同步整理_20260829_三段"
    segment = viz._discover_segment(dataset, take.segment_key)
    calibration = viz._default_calibration(dataset)
    identity = {
        side: viz.SimilarityRegistration(
            scale=1.0,
            rotation=np.eye(3, dtype=np.float64),
            fit_frame_count=1,
            anchor_residual_median_mm=0.0,
            anchor_residual_p95_mm=0.0,
        )
        for side in ("left", "right")
    }
    # External identities prevent the legacy Human.cma palm positions from
    # participating in a hidden fit.  prepare_take is used here only for the
    # checked RGB->solver timing mask and common-interval contract.
    prepared = viz.prepare_take(
        segment,
        calibration,
        registrations=identity,
        registration_source_segment=segment.name,
        max_interpolation_gap_ms=25.0,
        pose_mode=viz.POSE_MODE_GLOVE_WRIST,
    )
    mocap = mocap_overlay.prepare_mocap_video(
        segment,
        calibration,
        max_interpolation_gap_ms=25.0,
        mocap_position_source=mocap_overlay.MOCAP_POSITION_SOURCE_SKELETON_BVH,
        mocap_world_translation_mm=np.zeros(3, dtype=np.float64),
    )
    if not np.allclose(prepared.rgb_device_s, mocap.rgb_device_s, atol=0.0, rtol=0.0):
        raise ValueError("Take01 BVH registration timestamp preparations disagree")

    first = int(take.source_first)
    stop = first + int(take.frame_count)
    selected = slice(first, stop)
    if not np.all(mocap.mocap_valid[selected]):
        raise ValueError("Take01 BVH registration interval is incomplete")

    registrations: dict[str, viz.SimilarityRegistration] = {}
    strict_counts: dict[str, int] = {}
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
        sampled = nearest_solver_samples(
            sequence.times_s,
            sequence.world_mm,
            prepared.monotonic_s[selected],
        )
        strict = prepared.glove_valid[side][selected] & mocap.mocap_valid[selected]
        strict_counts[side] = int(np.count_nonzero(strict))
        registrations[side] = viz.fit_fixed_palm_registration(
            sampled.values,
            mocap.mocap_mm[selected, side_index],
            strict,
            mocap_wrist_rotations=None,
        )

    details = {
        "fit_position_source": "Skeleton_0/1 BVH forward kinematics",
        "sampling": "nearest observed solver row; no pose interpolation",
        "fit_joint_names": ["index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"],
        "rendered_source_rgb_interval": {
            "first": first,
            "stop_exclusive": stop,
            "frame_count": int(take.frame_count),
        },
        "strict_25ms_candidate_frames": strict_counts,
        "source_assets": {
            "left_skeleton_bvh": _source_asset(
                project_root, mocap.mocap_position_paths["left"]
            ),
            "right_skeleton_bvh": _source_asset(
                project_root, mocap.mocap_position_paths["right"]
            ),
            "left_glove_solver_keypoints": _source_asset(
                project_root, glove_paths["left"]
            ),
            "right_glove_solver_keypoints": _source_asset(
                project_root, glove_paths["right"]
            ),
            "human_cma_timestamps_only": _source_asset(
                project_root,
                segment / "动捕" / "Take_000" / "Take_000_Human.cma",
            ),
            "camera_cmavatar_alignment": _source_asset(
                project_root,
                segment / "同步校验" / "camera_cmavatar_alignment.csv",
            ),
            "common_interval_sync_report": _source_asset(
                project_root,
                segment / "同步校验" / "common_interval_sync_report.json",
            ),
        },
    }
    return segment.name, registrations, details


def _assert_tracked_bvh_registration(
    project_root: Path,
    profile_path: Path,
) -> tuple[str, dict[str, viz.SimilarityRegistration], Mapping[str, Any]]:
    source, tracked, payload = viz.load_registration_profile(profile_path)
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Tracked BVH registration is missing provenance")
    if provenance.get("fit_position_source") != "Skeleton_0/1 BVH forward kinematics":
        raise ValueError("Tracked registration was not fitted against Skeleton BVH FK")

    fitted_source, fitted, details = fit_take01_bvh_registration(project_root)
    if source != fitted_source:
        raise ValueError("Tracked BVH registration source segment is stale")
    if provenance.get("source_assets") != details["source_assets"]:
        raise ValueError("Tracked BVH registration source asset hashes are stale")
    for side in ("left", "right"):
        actual = tracked[side]
        expected = fitted[side]
        if not np.isclose(actual.scale, expected.scale, atol=1e-12, rtol=0.0):
            raise ValueError(f"Tracked BVH registration scale is stale for {side}")
        if not np.allclose(actual.rotation, expected.rotation, atol=1e-12, rtol=0.0):
            raise ValueError(f"Tracked BVH registration rotation is stale for {side}")
        if actual.fit_frame_count != expected.fit_frame_count:
            raise ValueError(f"Tracked BVH registration frame count is stale for {side}")
        if not np.isclose(
            actual.anchor_residual_median_mm,
            expected.anchor_residual_median_mm,
            atol=1e-9,
            rtol=0.0,
        ) or not np.isclose(
            actual.anchor_residual_p95_mm,
            expected.anchor_residual_p95_mm,
            atol=1e-9,
            rtol=0.0,
        ):
            raise ValueError(f"Tracked BVH registration residuals are stale for {side}")
    return source, fitted, payload


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    selected = np.asarray(values, dtype=np.float64).reshape(-1)
    selected = selected[np.isfinite(selected)]
    if len(selected) == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
            "rmse": float("nan"),
        }
    return {
        "count": int(len(selected)),
        "mean": float(np.mean(selected)),
        "median": float(np.median(selected)),
        "p95": float(np.percentile(selected, 95)),
        "max": float(np.max(selected)),
        "rmse": float(np.sqrt(np.mean(np.square(selected)))),
    }


def _error_metrics(
    error_mm: np.ndarray,
    frame_mask: np.ndarray,
    *,
    joint_names: Sequence[str],
) -> dict[str, Any]:
    error = np.asarray(error_mm, dtype=np.float64)
    mask = np.asarray(frame_mask, dtype=bool)
    if error.ndim != 2 or error.shape[0] != len(mask):
        raise ValueError("Error matrix and frame mask disagree")
    if error.shape[1] != len(joint_names):
        raise ValueError("Joint names do not match the error matrix")
    selected = error[mask]
    return {
        "frame_count": int(np.count_nonzero(mask)),
        "pooled_joint_epe_mm": _distribution(selected),
        "frame_mpjpe_mm": _distribution(np.mean(selected, axis=1)),
        "per_joint_epe_mm": {
            name: _distribution(selected[:, index])
            for index, name in enumerate(joint_names)
        },
    }


def _fit_fixed_similarity(
    source_vectors: np.ndarray,
    target_vectors: np.ndarray,
) -> FixedSimilarity:
    source = np.asarray(source_vectors, dtype=np.float64)
    target = np.asarray(target_vectors, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError("Fixed-similarity inputs must have shape [frame, point, xyz]")
    finite = np.all(np.isfinite(source), axis=(1, 2)) & np.all(
        np.isfinite(target), axis=(1, 2)
    )
    if np.count_nonzero(finite) < 20:
        raise ValueError("Too few finite frames for a fixed coordinate registration")

    keep = finite.copy()

    def solve(use: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        x = source[use].reshape(-1, 3)
        y = target[use].reshape(-1, 3)
        left, singular, right_t = np.linalg.svd(x.T @ y)
        if singular[1] <= singular[0] * 1e-8:
            raise ValueError("Fixed coordinate registration is degenerate")
        rotation = right_t.T @ left.T
        if np.linalg.det(rotation) < 0.0:
            right_t[-1] *= -1.0
            rotation = right_t.T @ left.T
        mapped = x @ rotation.T
        denominator = float(np.sum(mapped * mapped))
        if denominator <= 1e-9:
            raise ValueError("Fixed coordinate registration has zero extent")
        scale = float(np.sum(mapped * y) / denominator)
        residual = np.linalg.norm(
            source @ rotation.T * scale - target,
            axis=-1,
        )
        return scale, rotation, residual

    scale, rotation, residual = solve(keep)
    frame_residual = np.median(residual, axis=1)
    median = float(np.median(frame_residual[keep]))
    mad = float(np.median(np.abs(frame_residual[keep] - median)))
    threshold = median + max(3.0 * 1.4826 * mad, 3.0)
    refined = keep & (frame_residual <= threshold)
    if np.count_nonzero(refined) >= 20:
        keep = refined
        scale, rotation, residual = solve(keep)
    retained = residual[keep]
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Fixed coordinate registration produced an invalid scale")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("Fixed coordinate registration rotation is not orthogonal")
    return FixedSimilarity(
        scale=scale,
        rotation=rotation,
        input_frame_count=int(np.count_nonzero(finite)),
        retained_frame_count=int(np.count_nonzero(keep)),
        residual_median_mm=float(np.median(retained)),
        residual_p95_mm=float(np.percentile(retained, 95)),
    )


def _draw_header(
    image: np.ndarray,
    lines: Sequence[tuple[str, tuple[int, int, int]]],
) -> None:
    height = 24 + 23 * len(lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0.0, image)
    for index, (line, color) in enumerate(lines):
        viz._draw_text(
            image,
            line,
            (15, 23 + index * 22),
            scale=0.46 if index else 0.53,
            color=color,
        )


def _draw_connector(
    image: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    color: tuple[int, int, int] = CONNECTOR_BGR,
) -> None:
    if not (np.all(np.isfinite(first)) and np.all(np.isfinite(second))):
        return
    rectangle = (0, 0, image.shape[1], image.shape[0])
    p1 = tuple(np.rint(first).astype(int))
    p2 = tuple(np.rint(second).astype(int))
    visible, clipped_first, clipped_second = cv2.clipLine(rectangle, p1, p2)
    if visible:
        cv2.line(image, clipped_first, clipped_second, color, 1, cv2.LINE_AA)


def _render_old_comparison(
    project_root: Path,
    staging: Path,
    take: Any,
    *,
    translation_xyz_mm: np.ndarray,
    registrations: Mapping[str, viz.SimilarityRegistration],
    registration_source: str,
    registration_profile: Path,
) -> dict[str, Any]:
    dataset = project_root / "同步整理_20260829_三段"
    segment = viz._discover_segment(dataset, take.segment_key)
    calibration = viz._default_calibration(dataset)
    prepared = viz.prepare_take(
        segment,
        calibration,
        registrations=registrations,
        registration_source_segment=registration_source,
        max_interpolation_gap_ms=25.0,
        pose_mode=viz.POSE_MODE_GLOVE_WRIST,
        operator_world_translation_xyz_mm=translation_xyz_mm,
    )
    mocap = mocap_overlay.prepare_mocap_video(
        segment,
        calibration,
        max_interpolation_gap_ms=25.0,
        mocap_position_source=mocap_overlay.MOCAP_POSITION_SOURCE_SKELETON_BVH,
        mocap_world_translation_mm=translation_xyz_mm,
    )
    if not np.allclose(prepared.rgb_device_s, mocap.rgb_device_s, atol=0.0, rtol=0.0):
        raise ValueError("Glove and Skeleton BVH preparations disagree on RGB timestamps")

    first = int(take.source_first)
    stop = first + int(take.frame_count)
    selected = slice(first, stop)
    if not np.all(mocap.mocap_valid[selected]):
        raise ValueError(f"Latest BVH reference is not complete for {segment.name}")

    world: dict[str, np.ndarray] = {}
    nearest: dict[str, NearestSamples] = {}
    strict: dict[str, np.ndarray] = {}
    errors: dict[str, np.ndarray] = {}
    frame_rows: list[dict[str, Any]] = []
    glove_paths: dict[str, Path] = {}
    for side_index, side in enumerate(("left", "right")):
        glove_path = (
            segment / "手套解算" / "solved" / "primary"
            / f"{side}_hand_keypoints.csv"
        )
        glove_paths[side] = glove_path
        sequence = viz.load_glove_keypoints(glove_path)
        sampled = nearest_solver_samples(
            sequence.times_s,
            sequence.world_mm,
            prepared.monotonic_s[selected],
        )
        nearest[side] = sampled
        # Keep the solver wrist orientation.  Only the absent global wrist
        # translation is supplied by the synchronized MOCAP reference.
        world[side] = viz.apply_registration(
            sampled.values,
            mocap.mocap_mm[selected, side_index],
            registrations[side],
            mocap_wrist_rotations=None,
        )
        strict[side] = (
            prepared.glove_valid[side][selected]
            & mocap.mocap_valid[selected]
        )
        glove_relative = world[side] - world[side][:, 0, None, :]
        mocap_relative = (
            mocap.mocap_mm[selected, side_index]
            - mocap.mocap_mm[selected, side_index, 0, None, :]
        )
        errors[side] = np.linalg.norm(
            glove_relative[:, viz.GLOVE_COMMON_INDICES]
            - mocap_relative[:, viz.MOCAP_COMMON_INDICES],
            axis=2,
        )

    filename = f"{take.number:02d}_take{take.number:02d}_imu_solved_vs_bvh_mocap.mp4"
    poster_name = f"{take.number:02d}_take{take.number:02d}_imu_solved_vs_bvh_mocap.jpg"
    metrics_name = f"{take.number:02d}_take{take.number:02d}_imu_solved_vs_bvh_mocap.json"
    frame_map_name = f"take{take.number:02d}_imu_vs_mocap.csv"
    output = staging / "videos" / filename
    poster = staging / "posters" / poster_name
    metrics_path = staging / "metrics" / metrics_name
    frame_map_path = staging / "frame_maps" / frame_map_name
    intermediate = output.with_name(output.stem + ".mp4v.mp4")

    output_width = 960
    output_height = int(round(prepared.video.height * output_width / prepared.video.width))
    output_height += output_height % 2
    writer = cv2.VideoWriter(
        str(intermediate),
        cv2.VideoWriter_fourcc(*"mp4v"),
        prepared.video.fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {intermediate}")
    source_video = cv2.VideoCapture(str(prepared.video.path))
    snapshots: list[np.ndarray] = []
    snapshot_indices = set(
        np.rint(np.linspace(0, take.frame_count - 1, 6)).astype(int).tolist()
    )
    scale = output_width / prepared.video.width
    try:
        for source_index in range(stop):
            ok, frame = source_video.read()
            if not ok:
                raise RuntimeError(f"RGB decode stopped at source frame {source_index}")
            if source_index < first:
                continue
            output_index = source_index - first
            frame = cv2.resize(
                frame,
                (output_width, output_height),
                interpolation=cv2.INTER_AREA,
            )
            frame_summaries: list[str] = []
            frame_ages: list[float] = []
            strict_frame = True
            for side_index, side in enumerate(("left", "right")):
                mocap_pixels, mocap_positive = viz.project_world_to_rgb(
                    mocap.mocap_mm[source_index, side_index],
                    mocap.calibration,
                )
                mocap_pixels[~mocap_positive] = np.nan
                mocap_pixels *= scale
                viz.draw_hand_skeleton(
                    frame,
                    mocap_pixels,
                    viz.MOCAP_CHAINS,
                    MOCAP_BGR,
                    thickness=4,
                    radius=4,
                    hollow=True,
                )

                imu_pixels, imu_positive = viz.project_world_to_rgb(
                    world[side][output_index],
                    mocap.calibration,
                )
                imu_pixels[~imu_positive] = np.nan
                imu_pixels *= scale
                is_strict = bool(strict[side][output_index])
                strict_frame &= is_strict
                imu_color = IMU_FRESH_BGR if is_strict else IMU_STALE_BGR
                viz.draw_hand_skeleton(
                    frame,
                    imu_pixels,
                    viz.GLOVE_CHAINS,
                    imu_color,
                    thickness=3,
                    radius=3,
                    hollow=False,
                )
                for glove_index, mocap_index in zip(
                    viz.GLOVE_COMMON_INDICES,
                    viz.MOCAP_COMMON_INDICES,
                    strict=True,
                ):
                    _draw_connector(
                        frame,
                        imu_pixels[glove_index],
                        mocap_pixels[mocap_index],
                    )
                wrist = mocap_pixels[0]
                if np.all(np.isfinite(wrist)):
                    x, y = np.rint(wrist).astype(int)
                    viz._draw_text(
                        frame,
                        side[0].upper(),
                        (int(x + 8), int(y - 8)),
                        scale=0.45,
                    )
                frame_error = errors[side][output_index]
                frame_summaries.append(
                    f"{side[0].upper()} 16J={np.mean(frame_error):.1f}mm "
                    f"tips={np.mean(frame_error[[3, 7, 11, 15]]):.1f}mm"
                )
                frame_ages.append(float(nearest[side].age_ms[output_index]))
                frame_rows.append(
                    {
                        "output_frame": output_index,
                        "source_rgb_frame": source_index,
                        "side": side,
                        "rgb_elapsed_s": float(
                            prepared.rgb_device_s[source_index]
                            - prepared.rgb_device_s[first]
                        ),
                        "solver_sample_index": int(nearest[side].indices[output_index]),
                        "solver_sample_age_ms": float(nearest[side].age_ms[output_index]),
                        "strict_timing_valid": int(is_strict),
                        "root_normalized_16_joint_mpjpe_mm": float(
                            np.mean(frame_error)
                        ),
                        "index_tip_epe_mm": float(frame_error[3]),
                        "middle_tip_epe_mm": float(frame_error[7]),
                        "ring_tip_epe_mm": float(frame_error[11]),
                        "pinky_tip_epe_mm": float(frame_error[15]),
                    }
                )

            stale_label = "STRICT" if strict_frame else "STALE DISPLAY / NOT SCORED"
            _draw_header(
                frame,
                (
                    (
                        f"Take {take.number:02d} | IMU-SOLVED 20J vs MOCAP BVH-FK 21J",
                        (245, 245, 245),
                    ),
                    (
                        "nearest observed solver row | no interpolation, no smoothing, always drawn",
                        IMU_FRESH_BGR if strict_frame else IMU_STALE_BGR,
                    ),
                    ("  ".join(frame_summaries), (225, 225, 225)),
                    (
                        f"sample age max={max(frame_ages):.2f}ms | {stale_label} | "
                        "MOCAP wrist translation only; thumb/absolute wrist excluded",
                        (210, 225, 240),
                    ),
                ),
            )
            writer.write(frame)
            if output_index in snapshot_indices:
                snapshots.append(frame.copy())
    finally:
        source_video.release()
        writer.release()

    _transcode_h264(intermediate, output)
    _contact_sheet(snapshots, poster)
    frame_map_path.parent.mkdir(parents=True, exist_ok=True)
    with frame_map_path.open("w", encoding="utf-8", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(frame_rows)

    side_metrics: dict[str, Any] = {}
    for side in ("left", "right"):
        selected_error = errors[side]
        selected_strict = strict[side]
        tips = selected_error[:, (3, 7, 11, 15)]
        intermediates = selected_error[:, (1, 2, 5, 6, 9, 10, 13, 14)]
        mcps = selected_error[:, (0, 4, 8, 12)]
        side_metrics[side] = {
            "solver_sample_age_ms_all_displayed_frames": _distribution(
                nearest[side].age_ms
            ),
            "display_all_frames_not_scientific_gate": _error_metrics(
                selected_error,
                np.ones(take.frame_count, dtype=bool),
                joint_names=NON_THUMB_NAMES,
            ),
            "strict_25ms_scientific_subset": {
                **_error_metrics(
                    selected_error,
                    selected_strict,
                    joint_names=NON_THUMB_NAMES,
                ),
                "non_thumb_mcp_epe_mm": _distribution(mcps[selected_strict]),
                "non_thumb_intermediate_epe_mm": _distribution(
                    intermediates[selected_strict]
                ),
                "non_thumb_fingertip_epe_mm": _distribution(tips[selected_strict]),
            },
        }

    metrics = {
        "schema": METRICS_SCHEMA_OLD,
        "status": "comparison_complete",
        "source_segment": segment.name,
        "rendered_source_rgb_interval": {
            "first": first,
            "stop_exclusive": stop,
            "frame_count": take.frame_count,
        },
        "visualization": {
            "imu_layer": "nearest observed glove solver world-keypoint row; always drawn",
            "pose_interpolation": False,
            "pose_smoothing": False,
            "validity_gate_hides_pose": False,
            "stale_frames_colored_amber": True,
            "mocap_layer": "Skeleton_0/1 BVH forward kinematics on Human.cma timestamps",
            "drawn_connectors": "16 non-thumb one-to-one joints",
        },
        "comparison_protocol": {
            "imu_semantics": "20-joint glove solver output derived from IMU sensors; not raw sensor quaternions",
            "registration": "one frozen Take01 palm-only rotation and uniform scale",
            "mocap_wrist_translation_copied_per_frame": True,
            "mocap_wrist_orientation_copied_per_frame": False,
            "thumb_excluded_reason": "glove has three thumb joints while BVH has four",
            "absolute_wrist_translation_evaluable": False,
            "absolute_6dof_evaluable": False,
            "scientific_metric_mask": "original strict 25 ms bracketing/finite/common-interval gate",
            "stale_display_frames_in_scientific_metrics": False,
        },
        "manual_display_translation_xyz_mm": [
            float(value) for value in translation_xyz_mm
        ],
        "sides": side_metrics,
        "source_assets": {
            "rgb_video": _source_asset(project_root, prepared.video.path),
            "left_skeleton_bvh": _source_asset(
                project_root, mocap.mocap_position_paths["left"]
            ),
            "right_skeleton_bvh": _source_asset(
                project_root, mocap.mocap_position_paths["right"]
            ),
            "left_glove_solver_keypoints": _source_asset(
                project_root, glove_paths["left"]
            ),
            "right_glove_solver_keypoints": _source_asset(
                project_root, glove_paths["right"]
            ),
            "camera_world_calibration": _source_asset(project_root, calibration),
            "frozen_registration_profile": _source_asset(
                project_root, registration_profile
            ),
        },
        "claim_boundary": (
            "Root-normalized joint-position difference against a MOCAP solved-skeleton "
            "reference. The glove has no global translation, so this is not independent "
            "absolute wrist or 6DoF accuracy."
        ),
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "order": take.number,
        "id": f"take{take.number:02d}-imu-solved-vs-bvh",
        "label": f"Take {take.number:02d} · IMU solved pose vs BVH MOCAP",
        "source": segment.name,
        "filename": filename,
        "poster": f"posters/{poster_name}",
        "metrics": f"metrics/{metrics_name}",
        "frame_map": f"frame_maps/{frame_map_name}",
        "expected_frames": take.frame_count,
        "summary": {
            side: {
                "strict_frames": side_metrics[side]["strict_25ms_scientific_subset"]["frame_count"],
                "joint_median_mm": side_metrics[side]["strict_25ms_scientific_subset"]["pooled_joint_epe_mm"]["median"],
                "joint_p95_mm": side_metrics[side]["strict_25ms_scientific_subset"]["pooled_joint_epe_mm"]["p95"],
                "tip_median_mm": side_metrics[side]["strict_25ms_scientific_subset"]["non_thumb_fingertip_epe_mm"]["median"],
            }
            for side in ("left", "right")
        },
    }


def _render_take007_comparison(
    project_root: Path,
    staging: Path,
    *,
    manual_profile: Path,
) -> dict[str, Any]:
    capture = load_new_capture(
        project_root / "thor_new4_20260831_processed",
        project_root / "movementcap_20260831_worldcalib_tabletop_final.tar.gz",
    )
    manual = load_manual_calibration(manual_profile, video_id="take007-solved")
    _, strict_valid, strict_diagnostics = load_solved_pose(capture)
    output_count = len(capture.clock.output_indices)
    calibration_stop = int(round(output_count * TAKE007_CALIBRATION_FRACTION))
    calibration_window = np.arange(output_count) < calibration_stop

    nearest: dict[str, NearestSamples] = {}
    fixed: dict[str, FixedSimilarity] = {}
    world: dict[str, np.ndarray] = {}
    errors: dict[str, np.ndarray] = {}
    glove_paths: dict[str, Path] = {}
    shifted_markers = np.asarray(capture.marker_world_mm, dtype=np.float64).copy()
    roots = np.empty((output_count, 2, 3), dtype=np.float64)
    for side_index, side in enumerate(ANATOMICAL_SIDES):
        side_offset = manual.global_world_xyz_mm + manual.side_world_xyz_mm[side]
        shifted_markers[:, side_index] += side_offset
        roots[:, side_index] = virtual_wrist(
            capture.marker_world_mm[:, side_index],
            rear_offset_mm=manual.rear_offset_mm,
        ) + side_offset

        glove_path = (
            capture.root / "glove_processing" / "solved" / "primary"
            / f"{side}_hand_keypoints.csv"
        )
        glove_paths[side] = glove_path
        sequence = viz.load_glove_keypoints(glove_path)
        sampled = nearest_solver_samples(
            sequence.times_s,
            sequence.world_mm,
            capture.clock.target_monotonic_s,
        )
        nearest[side] = sampled
        source_vectors = (
            sampled.values[:, TAKE007_BASE_GLOVE_INDICES]
            - sampled.values[:, 0, None, :]
        )
        target_vectors = (
            capture.marker_world_mm[:, side_index, TAKE007_BASE_MARKER_INDICES]
            - virtual_wrist(
                capture.marker_world_mm[:, side_index],
                rear_offset_mm=manual.rear_offset_mm,
            )[:, None, :]
        )
        fit_mask = strict_valid & calibration_window
        transform = _fit_fixed_similarity(
            source_vectors[fit_mask],
            target_vectors[fit_mask],
        )
        fixed[side] = transform
        world[side] = (
            (sampled.values - sampled.values[:, 0, None, :])
            @ transform.rotation.T
            * transform.scale
            + roots[:, side_index, None, :]
        )
        errors[side] = np.linalg.norm(
            world[side][:, MARKER_TO_GLOVE_INDICES]
            - shifted_markers[:, side_index, :10],
            axis=2,
        )

    filename = "04_take007_imu_solved_vs_cmm_mocap.mp4"
    poster_name = "04_take007_imu_solved_vs_cmm_mocap.jpg"
    metrics_name = "04_take007_imu_solved_vs_cmm_mocap.json"
    frame_map_name = "take007_imu_vs_mocap.csv"
    output = staging / "videos" / filename
    poster = staging / "posters" / poster_name
    metrics_path = staging / "metrics" / metrics_name
    frame_map_path = staging / "frame_maps" / frame_map_name
    intermediate = output.with_name(output.stem + ".mp4v.mp4")

    # Intrinsics payload variants do not consistently expose width/height;
    # use the verified MP4 properties as the render contract.
    probe = cv2.VideoCapture(str(capture.rgb_path))
    source_width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(probe.get(cv2.CAP_PROP_FPS))
    probe.release()
    output_width = 960
    output_height = int(round(source_height * output_width / source_width))
    output_height += output_height % 2
    writer = cv2.VideoWriter(
        str(intermediate),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {intermediate}")
    source_video = cv2.VideoCapture(str(capture.rgb_path))
    snapshots: list[np.ndarray] = []
    snapshot_indices = set(
        np.rint(np.linspace(0, output_count - 1, 6)).astype(int).tolist()
    )
    scale = output_width / source_width
    frame_rows: list[dict[str, Any]] = []
    mocap_nodes = raw_mocap_nodes(
        capture.marker_world_mm,
        rear_offset_mm=manual.rear_offset_mm,
    )
    for side_index, side in enumerate(ANATOMICAL_SIDES):
        mocap_nodes[:, side_index] += (
            manual.global_world_xyz_mm + manual.side_world_xyz_mm[side]
        )
    try:
        for frame_index in capture.clock.output_indices:
            ok, frame = source_video.read()
            if not ok:
                raise RuntimeError(f"RGB decode stopped at frame {frame_index}")
            frame = cv2.resize(
                frame,
                (output_width, output_height),
                interpolation=cv2.INTER_AREA,
            )
            summaries: list[str] = []
            ages: list[float] = []
            for side_index, side in enumerate(ANATOMICAL_SIDES):
                marker_pixels, marker_positive = project_world(
                    mocap_nodes[frame_index, side_index],
                    capture.camera,
                )
                marker_pixels[~marker_positive] = np.nan
                marker_pixels *= scale
                viz.draw_hand_skeleton(
                    frame,
                    marker_pixels,
                    RAW_CHAINS,
                    TAKE007_MOCAP_BGR[side],
                    thickness=4,
                    radius=5,
                    hollow=True,
                )
                imu_pixels, imu_positive = project_world(
                    world[side][frame_index],
                    capture.camera,
                )
                imu_pixels[~imu_positive] = np.nan
                imu_pixels *= scale
                imu_color = (
                    TAKE007_IMU_BGR[side]
                    if strict_valid[frame_index]
                    else IMU_STALE_BGR
                )
                viz.draw_hand_skeleton(
                    frame,
                    imu_pixels,
                    GLOVE_CHAINS,
                    imu_color,
                    thickness=3,
                    radius=3,
                    hollow=False,
                )
                for marker_index, glove_index in enumerate(MARKER_TO_GLOVE_INDICES):
                    _draw_connector(
                        frame,
                        imu_pixels[glove_index],
                        marker_pixels[marker_index + 1],
                    )
                current = errors[side][frame_index]
                summaries.append(
                    f"{side[0].upper()} 10pt={np.mean(current):.1f}mm "
                    f"tips={np.mean(current[TIP_MARKER_INDICES]):.1f}mm"
                )
                ages.append(float(nearest[side].age_ms[frame_index]))
                frame_rows.append(
                    {
                        "output_frame": int(frame_index),
                        "source_rgb_frame": int(frame_index),
                        "side": side,
                        "phase": "calibration" if frame_index < calibration_stop else "evaluation",
                        "solver_sample_index": int(nearest[side].indices[frame_index]),
                        "solver_sample_age_ms": float(nearest[side].age_ms[frame_index]),
                        "strict_timing_valid": int(strict_valid[frame_index]),
                        "all_10_correspondence_mean_mm": float(np.mean(current)),
                        "five_tip_mean_mm": float(np.mean(current[TIP_MARKER_INDICES])),
                        "five_base_mean_mm": float(np.mean(current[BASE_MARKER_INDICES])),
                    }
                )
            state = "STRICT" if strict_valid[frame_index] else "STALE DISPLAY / NOT SCORED"
            phase = "CALIBRATION" if frame_index < calibration_stop else "EVALUATION"
            _draw_header(
                frame,
                (
                    (
                        "Take_007 | IMU-SOLVED 20J vs CMM 10 surface markers / hand",
                        (245, 245, 245),
                    ),
                    (
                        "nearest observed solver row | fixed first-20% base-only coordinate transform",
                        IMU_FRESH_BGR if strict_valid[frame_index] else IMU_STALE_BGR,
                    ),
                    ("  ".join(summaries), (225, 225, 225)),
                    (
                        f"age max={max(ages):.2f}ms | {phase} | {state} | "
                        "no per-frame CMM rotation; wrist translation conditioned",
                        (210, 225, 240),
                    ),
                ),
            )
            writer.write(frame)
            if int(frame_index) in snapshot_indices:
                snapshots.append(frame.copy())
    finally:
        source_video.release()
        writer.release()

    _transcode_h264(intermediate, output)
    _contact_sheet(snapshots, poster)
    with frame_map_path.open("w", encoding="utf-8", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(frame_rows)

    evaluation = np.arange(output_count) >= calibration_stop
    strict_evaluation = evaluation & strict_valid
    side_metrics: dict[str, Any] = {}
    marker_names = (
        "thumb_tip", "thumb_base", "index_tip", "index_base",
        "middle_tip", "middle_base", "ring_tip", "ring_base",
        "pinky_tip", "pinky_base",
    )
    for side in ANATOMICAL_SIDES:
        error = errors[side]
        transform = fixed[side]
        side_metrics[side] = {
            "fixed_coordinate_registration": {
                "scale": transform.scale,
                "rotation_mocap_from_solver": transform.rotation.tolist(),
                "input_frames": transform.input_frame_count,
                "retained_frames": transform.retained_frame_count,
                "fit_base_residual_median_mm": transform.residual_median_mm,
                "fit_base_residual_p95_mm": transform.residual_p95_mm,
            },
            "solver_sample_age_ms_all_displayed_frames": _distribution(
                nearest[side].age_ms
            ),
            "evaluation_all_displayed_frames_not_scientific_gate": {
                **_error_metrics(error, evaluation, joint_names=marker_names),
                "five_tip_epe_mm": _distribution(error[evaluation][:, TIP_MARKER_INDICES]),
                "five_base_epe_mm": _distribution(error[evaluation][:, BASE_MARKER_INDICES]),
            },
            "evaluation_strict_20ms_camera_valid_subset": {
                **_error_metrics(error, strict_evaluation, joint_names=marker_names),
                "five_tip_epe_mm": _distribution(
                    error[strict_evaluation][:, TIP_MARKER_INDICES]
                ),
                "five_base_epe_mm": _distribution(
                    error[strict_evaluation][:, BASE_MARKER_INDICES]
                ),
            },
        }

    metrics = {
        "schema": METRICS_SCHEMA_TAKE007,
        "status": "comparison_complete",
        "source_recording": capture.recording,
        "source_take": capture.take,
        "rendered_frames": output_count,
        "visualization": {
            "imu_layer": "nearest observed glove solver world-keypoint row; always drawn",
            "pose_interpolation": False,
            "pose_smoothing": False,
            "validity_gate_hides_pose": False,
            "stale_frames_colored_amber": True,
            "mocap_layer": "ten photographed CMM surface markers per hand",
            "drawn_connectors": "all ten photograph-defined marker/keypoint correspondences",
        },
        "comparison_protocol": {
            "imu_semantics": "20-joint glove solver output derived from IMU sensors; not raw sensor quaternions",
            "coordinate_registration": "fixed rotation and scale fitted on first 20% only using four non-thumb base markers",
            "calibration_frame_interval": [0, calibration_stop],
            "evaluation_frame_interval": [calibration_stop, output_count],
            "per_frame_cmm_rotation_used": False,
            "cmm_virtual_wrist_translation_copied_per_frame": True,
            "absolute_wrist_translation_evaluable": False,
            "absolute_6dof_evaluable": False,
            "scientific_metric_mask": "camera_valid and nearest solver sample <=20 ms",
            "stale_display_frames_in_scientific_metrics": False,
        },
        "original_strict_validity_diagnostics": strict_diagnostics,
        "manual_display_translation": {
            "global_world_xyz_mm": manual.global_world_xyz_mm.tolist(),
            "left_world_xyz_mm": manual.side_world_xyz_mm["left"].tolist(),
            "right_world_xyz_mm": manual.side_world_xyz_mm["right"].tolist(),
            "rear_offset_mm": manual.rear_offset_mm,
        },
        "sides": side_metrics,
        "source_assets": {
            "rgb_video": _source_asset(project_root, capture.rgb_path),
            "cmm_markers": _source_asset(project_root, capture.cmm_path),
            "camera_alignment": _source_asset(project_root, capture.alignment_path),
            "glove_validity_summary": _source_asset(
                project_root, capture.frame_summary_path
            ),
            "left_glove_solver_keypoints": _source_asset(
                project_root, glove_paths["left"]
            ),
            "right_glove_solver_keypoints": _source_asset(
                project_root, glove_paths["right"]
            ),
            "manual_profile": _source_asset(project_root, manual_profile),
        },
        "claim_boundary": (
            "CMM reflectors are glove-surface points rather than anatomical joint centers. "
            "The virtual wrist translation is conditioned on CMM, so this is an articulation/"
            "surface-correspondence comparison, not independent glove wrist or absolute 6DoF accuracy."
        ),
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    strict_summary = {
        side: side_metrics[side]["evaluation_strict_20ms_camera_valid_subset"]
        for side in ANATOMICAL_SIDES
    }
    return {
        "order": 4,
        "id": "take007-imu-solved-vs-cmm",
        "label": "Take_007 · IMU solved pose vs CMM MOCAP markers",
        "source": f"{capture.recording}/{capture.take}",
        "filename": filename,
        "poster": f"posters/{poster_name}",
        "metrics": f"metrics/{metrics_name}",
        "frame_map": f"frame_maps/{frame_map_name}",
        "expected_frames": output_count,
        "summary": {
            side: {
                "strict_frames": strict_summary[side]["frame_count"],
                "joint_median_mm": strict_summary[side]["pooled_joint_epe_mm"]["median"],
                "joint_p95_mm": strict_summary[side]["pooled_joint_epe_mm"]["p95"],
                "tip_median_mm": strict_summary[side]["five_tip_epe_mm"]["median"],
            }
            for side in ANATOMICAL_SIDES
        },
    }


def _probe_video(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames,duration",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    numerator, denominator = (
        int(value) for value in stream["avg_frame_rate"].split("/")
    )
    return {
        "codec": stream["codec_name"],
        "pixel_format": stream.get("pix_fmt"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": numerator / denominator,
        "frame_count": int(stream["nb_frames"]),
        "duration_s": float(stream["duration"]),
        "bytes": Path(path).stat().st_size,
    }


def _faststart(path: Path) -> bool:
    data = Path(path).read_bytes()
    moov = data.find(b"moov")
    mdat = data.find(b"mdat")
    return moov >= 0 and mdat >= 0 and moov < mdat


def write_checksums(destination: Path) -> Path:
    root = Path(destination)
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.name != "SHA256SUMS.txt":
            lines.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    output = root / "SHA256SUMS.txt"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _write_readme(destination: Path) -> None:
    (destination / "README.md").write_text(
        """# IMU-solved pose vs MOCAP supplemental comparison

This folder is separate from the canonical nine-video delivery. It contains
four same-frame overlays: three old takes compare the unmodified glove solver
world pose against the latest Skeleton_0/1 BVH-FK MOCAP reference; Take_007
compares it against the ten photographed CMM surface markers per hand.

The display uses the nearest observed solver row on every RGB frame. It does
not interpolate, smooth, or hide stale rows. Amber means the row failed the
original 20/25 ms scientific timing gate and is excluded from strict metrics.

Important interpretation boundaries:

- “IMU-solved” means the delivered 20-joint glove solver output derived from
  IMUs. It is not the raw sensor quaternion stream.
- Old takes preserve solver wrist orientation and copy only MOCAP wrist
  translation because the glove output has no global translation.
- Their one fixed rotation and scale is fitted directly on Take01 Skeleton BVH
  non-thumb MCP positions, then frozen for Take02/03. The builder recomputes
  and verifies the tracked profile and its source hashes before rendering.
- Primary old-take metrics cover the unambiguous 16 non-thumb joints. Thumb is
  excluded because the glove has three thumb joints and BVH has four.
- Take_007 uses one fixed base-only coordinate registration fitted on the first
  20% and frozen on the remaining 80%; it never copies per-frame CMM rotation.
- Absolute wrist translation and independent 6DoF accuracy are unavailable.
  CMM markers are glove-surface points, not anatomical joint centers.
- The `[0, -44, 0] mm` operator profile moves both compared layers for display
  and cancels from root-normalized error; it is not a metric improvement.

Open `index.html` through the project review server, or use `manifest.json`,
the per-frame CSV files, and `SHA256SUMS.txt` for reproducible review.
""",
        encoding="utf-8",
    )


def _write_webpage(destination: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    cards = []
    for entry in entries:
        left = entry["summary"]["left"]
        right = entry["summary"]["right"]
        cards.append(
            f"""
      <article class="card">
        <header><span>{entry['order']:02d}</span><div><h2>{entry['label']}</h2><p>{entry['source']}</p></div></header>
        <video controls playsinline preload="metadata" poster="{entry['poster']}">
          <source src="videos/{entry['filename']}" type="video/mp4">
        </video>
        <div class="stats">
          <div><b>{left['joint_median_mm']:.2f} / {left['joint_p95_mm']:.2f} mm</b><span>Left strict median / P95</span></div>
          <div><b>{right['joint_median_mm']:.2f} / {right['joint_p95_mm']:.2f} mm</b><span>Right strict median / P95</span></div>
          <div><b>{left['strict_frames']} / {entry['frame_count']}</b><span>strict timing frames</span></div>
        </div>
        <nav><a href="videos/{entry['filename']}" download>下载 MP4</a><a href="{entry['metrics']}">完整 metrics</a><a href="{entry['frame_map']}">逐帧 CSV</a></nav>
      </article>"""
        )
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#07111f"><title>IMU solved pose vs MOCAP</title><link rel="stylesheet" href="styles.css"></head>
<body><header class="top"><a href="/final-nine/">← 返回最终九视频</a><a href="manifest.json">manifest.json</a></header>
<main><section class="hero"><p>SUPPLEMENTAL · NOT PART OF FINAL 9</p><h1>IMU-solved pose<br><em>vs MOCAP reference</em></h1>
<div class="notice"><b>绿色/黄色：IMU solver 20 joints；空心骨架/标记：MOCAP。</b> 每帧直接选择最近的真实 solver row，不插值、不平滑、不过门控隐藏。橙色只表示 stale display，并从 strict 指标排除。</div></section>
<section class="grid">{''.join(cards)}</section>
<section class="boundary"><h2>这些数字是什么</h2><p>旧三段报告 wrist-root-normalized 的 16 个非拇指关节 3D EPE；Take_007 报告 CMM surface-marker 对应残差。腕部平移由 MOCAP 条件化，所以不是独立 absolute wrist / 6DoF GT 精度。</p><a href="README.md">读取完整 README</a></section></main>
<footer>GT_CALIB · raw display continuity + strict metric separation</footer></body></html>"""
    styles = """:root{color-scheme:dark;--bg:#07111f;--panel:#0e2034;--ink:#e9f1fb;--muted:#93a5bb;--cyan:#42d9ee;--green:#4ce69a;--orange:#ffad45;--line:#28405b}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 85% 0,#10314b 0,transparent 32rem),var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,"PingFang SC",sans-serif}.top{position:sticky;top:0;z-index:3;display:flex;justify-content:space-between;padding:18px max(24px,calc((100vw - 1240px)/2));background:#07111fe8;border-bottom:1px solid var(--line);backdrop-filter:blur(15px)}a{color:var(--cyan);font-weight:750;text-decoration:none}.hero,.grid,.boundary{width:min(1240px,calc(100% - 40px));margin:auto}.hero{padding:80px 0 55px}.hero>p{color:var(--cyan);font-weight:900;letter-spacing:.18em;font-size:.75rem}.hero h1{margin:10px 0 28px;font-size:clamp(3rem,7vw,6rem);line-height:.95;letter-spacing:-.055em}.hero em{color:var(--green);font-style:normal}.notice,.boundary{padding:22px 25px;border:1px solid var(--line);border-radius:16px;background:#0b1a2bd9;color:var(--muted)}.notice b{color:var(--ink)}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px;padding-bottom:55px}.card{overflow:hidden;border:1px solid var(--line);border-radius:18px;background:var(--panel)}.card header{display:flex;gap:15px;padding:20px}.card header>span{display:grid;width:42px;height:42px;place-items:center;border:1px solid var(--line);border-radius:50%;color:var(--cyan);font-weight:900}.card h2{margin:0;font-size:1.15rem}.card header p{margin:3px 0 0;color:var(--muted);font-size:.82rem}.card video{display:block;width:100%;aspect-ratio:16/9;background:#000}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line)}.stats div{padding:15px;background:#0b192b}.stats b,.stats span{display:block}.stats b{font-size:.92rem}.stats span{margin-top:3px;color:var(--muted);font-size:.68rem}.card nav{display:flex;gap:18px;padding:16px 20px;font-size:.82rem}.boundary{margin-bottom:70px}.boundary h2{margin-top:0;color:var(--ink)}footer{padding:28px;text-align:center;border-top:1px solid var(--line);color:var(--muted);font-size:.8rem}@media(max-width:850px){.grid{grid-template-columns:1fr}.stats{grid-template-columns:1fr}.card nav{flex-wrap:wrap}}"""
    (destination / "index.html").write_text(html, encoding="utf-8")
    (destination / "styles.css").write_text(styles, encoding="utf-8")


def build_comparison_delivery(
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
    registration_profile = project_root / OLD_REGISTRATION_PROFILE
    registration_source, registrations, registration_payload = (
        _assert_tracked_bvh_registration(project_root, registration_profile)
    )
    if registration_payload.get("pose_mode") != viz.POSE_MODE_GLOVE_WRIST:
        raise ValueError("Comparison registration must preserve glove wrist orientation")

    staging = destination.with_name(destination.name + ".staging")
    if destination.exists() or staging.exists():
        raise FileExistsError(
            f"Refusing to overwrite comparison delivery or staging: {destination}"
        )
    for name in ("videos", "posters", "metrics", "frame_maps", "calibration"):
        (staging / name).mkdir(parents=True, exist_ok=True)
    try:
        packaged_profile = staging / "calibration" / "applied_manual_profile.json"
        shutil.copy2(profile_path, packaged_profile)
        entries: list[dict[str, Any]] = []
        for take in OLD_TAKES:
            translation = final_profile.effective_world_xyz_mm(
                f"take{take.number:02d}-solved"
            )
            if translation is None:
                raise ValueError(f"Manual profile excludes Take {take.number:02d} solved pose")
            entries.append(
                _render_old_comparison(
                    project_root,
                    staging,
                    take,
                    translation_xyz_mm=np.asarray(translation, dtype=np.float64),
                    registrations=registrations,
                    registration_source=registration_source,
                    registration_profile=registration_profile,
                )
            )
        entries.append(
            _render_take007_comparison(
                project_root,
                staging,
                manual_profile=profile_path,
            )
        )

        for entry in entries:
            video = staging / "videos" / entry["filename"]
            poster = staging / entry["poster"]
            metrics = staging / entry["metrics"]
            frame_map = staging / entry["frame_map"]
            probe = _probe_video(video)
            expected_frames = int(entry.pop("expected_frames"))
            if probe["frame_count"] != expected_frames:
                raise ValueError(f"Rendered frame count mismatch: {video}")
            entry.update(probe)
            entry["sha256"] = _sha256(video)
            entry["poster_sha256"] = _sha256(poster)
            entry["metrics_sha256"] = _sha256(metrics)
            entry["frame_map_sha256"] = _sha256(frame_map)
            entry["faststart"] = _faststart(video)
        manifest = {
            "schema": DELIVERY_SCHEMA,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "pass",
            "comparison_count": len(entries),
            "canonical_final_nine_unchanged": True,
            "videos": entries,
            "display_policy": {
                "sampling": "nearest observed solver row",
                "pose_interpolation": False,
                "pose_smoothing": False,
                "validity_gate_hides_pose": False,
                "stale_pose_is_drawn": True,
                "stale_pose_in_strict_metrics": False,
            },
            "manual_profile": {
                "path": packaged_profile.relative_to(staging).as_posix(),
                "bytes": packaged_profile.stat().st_size,
                "sha256": _sha256(packaged_profile),
                "source": _source_asset(project_root, profile_path),
            },
            "claim_boundary": (
                "Supplemental conditioned articulation comparison, separate from the "
                "canonical nine-video package; not independent absolute wrist/6DoF GT accuracy."
            ),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_readme(staging)
        _write_webpage(staging, entries)
        write_checksums(staging)
        first_validation = validate_comparison_delivery(staging, full_decode=False)
        if first_validation["status"] != "pass":
            raise ValueError(f"Comparison pre-validation failed: {first_validation['failures']}")
        validation = validate_comparison_delivery(staging, full_decode=True)
        (staging / "validation.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_checksums(staging)
        final_validation = validate_comparison_delivery(staging, full_decode=False)
        if final_validation["status"] != "pass":
            raise ValueError(f"Comparison final validation failed: {final_validation['failures']}")
        staging.rename(destination)
        return destination
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


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


def validate_comparison_delivery(
    destination: Path,
    *,
    full_decode: bool = False,
) -> dict[str, Any]:
    root_input = Path(destination).expanduser()
    if root_input.is_symlink():
        return {
            "schema": VALIDATION_SCHEMA,
            "status": "fail",
            "comparison_count": 0,
            "full_decode": full_decode,
            "failures": ["comparison delivery root must not be a symlink"],
        }
    root = root_input.resolve()
    failures: list[str] = []
    try:
        manifest_path = _safe_relative(root, "manifest.json", label="manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        return {
            "schema": VALIDATION_SCHEMA,
            "status": "fail",
            "comparison_count": 0,
            "full_decode": full_decode,
            "failures": [str(error)],
        }
    videos = manifest.get("videos")
    if manifest.get("schema") != DELIVERY_SCHEMA:
        failures.append("manifest schema mismatch")
    if manifest.get("status") != "pass":
        failures.append("manifest status is not pass")
    if manifest.get("canonical_final_nine_unchanged") is not True:
        failures.append("comparison must remain outside the canonical final nine")
    display_policy = manifest.get("display_policy")
    expected_policy = {
        "sampling": "nearest observed solver row",
        "pose_interpolation": False,
        "pose_smoothing": False,
        "validity_gate_hides_pose": False,
        "stale_pose_is_drawn": True,
        "stale_pose_in_strict_metrics": False,
    }
    if display_policy != expected_policy:
        failures.append("display/scientific-mask policy mismatch")
    try:
        manual_profile = manifest.get("manual_profile")
        if not isinstance(manual_profile, Mapping):
            raise ValueError("packaged manual profile is missing")
        packaged_profile = _safe_relative(
            root,
            manual_profile.get("path"),
            label="manual_profile.path",
        )
        if (
            type(manual_profile.get("bytes")) is not int
            or packaged_profile.stat().st_size != manual_profile["bytes"]
            or _sha256(packaged_profile) != manual_profile.get("sha256")
        ):
            raise ValueError("packaged manual profile hash/size mismatch")
        packaged_payload = json.loads(packaged_profile.read_text(encoding="utf-8"))
        if packaged_payload.get("schema") != "gt_calib.final_nine_manual_xyz.v1":
            raise ValueError("packaged manual profile schema mismatch")
    except Exception as error:
        failures.append(str(error))
    if (
        manifest.get("comparison_count") != EXPECTED_COMPARISONS
        or not isinstance(videos, list)
        or len(videos) != EXPECTED_COMPARISONS
    ):
        failures.append("manifest must contain exactly four supplemental comparisons")
        videos = videos if isinstance(videos, list) else []
    orders: list[int] = []
    video_names: list[str] = []
    for index, entry in enumerate(videos):
        try:
            if not isinstance(entry, Mapping):
                raise ValueError(f"videos[{index}] must be an object")
            expected_id, expected_filename, expected_frames, expected_metrics_schema = (
                EXPECTED_VIDEO_CONTRACTS[index]
            )
            order = entry["order"]
            if type(order) is not int:
                raise ValueError(f"order must be an integer for videos[{index}]")
            orders.append(order)
            if entry.get("id") != expected_id or entry.get("filename") != expected_filename:
                raise ValueError(f"canonical comparison identity mismatch at index {index}")
            video_names.append(expected_filename)
            video = _safe_relative(
                root, f"videos/{entry['filename']}", label=f"videos[{index}]"
            )
            poster = _safe_relative(root, entry["poster"], label=f"poster[{index}]")
            metrics = _safe_relative(root, entry["metrics"], label=f"metrics[{index}]")
            frame_map = _safe_relative(
                root, entry["frame_map"], label=f"frame_map[{index}]"
            )
            for path, key in (
                (video, "sha256"),
                (poster, "poster_sha256"),
                (metrics, "metrics_sha256"),
                (frame_map, "frame_map_sha256"),
            ):
                if _sha256(path) != entry.get(key):
                    raise ValueError(f"{key} mismatch for {entry.get('id')}")
            probe = _probe_video(video)
            for key in (
                "codec",
                "pixel_format",
                "width",
                "height",
                "frame_count",
                "bytes",
            ):
                if probe[key] != entry.get(key):
                    raise ValueError(f"{key} mismatch for {entry.get('id')}")
            for key in ("fps", "duration_s"):
                value = entry.get(key)
                if type(value) not in (int, float) or not np.isclose(
                    probe[key], float(value), atol=1e-6, rtol=0.0
                ):
                    raise ValueError(f"{key} mismatch for {entry.get('id')}")
            if probe["codec"] != "h264" or probe["pixel_format"] != "yuv420p":
                raise ValueError(f"browser codec contract failed for {entry.get('id')}")
            if (
                probe["width"] != 960
                or probe["height"] != 540
                or not np.isclose(probe["fps"], 30.0, atol=1e-9, rtol=0.0)
                or probe["frame_count"] != expected_frames
            ):
                raise ValueError(f"canonical media geometry mismatch for {entry.get('id')}")
            if not entry.get("faststart") or not _faststart(video):
                raise ValueError(f"fast-start contract failed for {entry.get('id')}")
            if video.stat().st_size > 25 * 1024 * 1024:
                raise ValueError(f"Cloudflare 25 MiB asset limit exceeded: {video.name}")
            if full_decode:
                completed = subprocess.run(
                    [
                        "ffmpeg", "-v", "error", "-i", str(video),
                        "-map", "0:v:0", "-f", "null", "-",
                    ],
                    capture_output=True,
                    text=True,
                )
                if completed.returncode != 0:
                    raise ValueError(f"full decode failed: {video.name}")

            try:
                metrics_payload = json.loads(metrics.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"metrics JSON is invalid for {entry.get('id')}") from error
            if (
                metrics_payload.get("schema") != expected_metrics_schema
                or metrics_payload.get("status") != "comparison_complete"
            ):
                raise ValueError(f"metrics contract mismatch for {entry.get('id')}")
            metric_sides = metrics_payload.get("sides")
            summary = entry.get("summary")
            if not isinstance(metric_sides, Mapping) or not isinstance(summary, Mapping):
                raise ValueError(f"metrics/summary sides missing for {entry.get('id')}")
            metric_key = (
                "strict_25ms_scientific_subset"
                if expected_metrics_schema == METRICS_SCHEMA_OLD
                else "evaluation_strict_20ms_camera_valid_subset"
            )
            for side in ("left", "right"):
                side_metrics = metric_sides.get(side)
                side_summary = summary.get(side)
                if not isinstance(side_metrics, Mapping) or not isinstance(
                    side_summary, Mapping
                ):
                    raise ValueError(f"missing {side} metrics for {entry.get('id')}")
                strict_metrics = side_metrics.get(metric_key)
                if not isinstance(strict_metrics, Mapping):
                    raise ValueError(f"missing strict {side} metrics for {entry.get('id')}")
                pooled = strict_metrics.get("pooled_joint_epe_mm")
                tips = strict_metrics.get(
                    "non_thumb_fingertip_epe_mm"
                    if expected_metrics_schema == METRICS_SCHEMA_OLD
                    else "five_tip_epe_mm"
                )
                if not isinstance(pooled, Mapping) or not isinstance(tips, Mapping):
                    raise ValueError(f"strict summary distributions missing for {entry.get('id')}")
                expected_summary = {
                    "strict_frames": strict_metrics.get("frame_count"),
                    "joint_median_mm": pooled.get("median"),
                    "joint_p95_mm": pooled.get("p95"),
                    "tip_median_mm": tips.get("median"),
                }
                if side_summary != expected_summary:
                    raise ValueError(f"manifest summary mismatch for {entry.get('id')} {side}")

            with frame_map.open("r", encoding="utf-8", newline="") as handle:
                frame_rows = list(csv.DictReader(handle))
            if len(frame_rows) != expected_frames * 2:
                raise ValueError(f"frame-map row count mismatch for {entry.get('id')}")
            frame_keys: set[tuple[int, str]] = set()
            strict_csv_counts = {"left": 0, "right": 0}
            last_sample_index = {"left": -1, "right": -1}
            for row in frame_rows:
                output_frame = int(row["output_frame"])
                side = row["side"]
                solver_sample_index = int(row["solver_sample_index"])
                age_ms = float(row["solver_sample_age_ms"])
                strict_value = int(row["strict_timing_valid"])
                if (
                    output_frame < 0
                    or output_frame >= expected_frames
                    or side not in strict_csv_counts
                    or solver_sample_index < 0
                    or not np.isfinite(age_ms)
                    or age_ms < 0.0
                    or strict_value not in (0, 1)
                ):
                    raise ValueError(f"invalid frame-map row for {entry.get('id')}")
                strict_age_limit_ms = (
                    12.5 if expected_metrics_schema == METRICS_SCHEMA_OLD else 20.0
                )
                if strict_value and age_ms > strict_age_limit_ms + 1e-9:
                    raise ValueError(
                        f"strict frame exceeds solver-age limit for {entry.get('id')}"
                    )
                if solver_sample_index < last_sample_index[side]:
                    raise ValueError(
                        f"solver sample order regressed for {entry.get('id')} {side}"
                    )
                last_sample_index[side] = solver_sample_index
                key = (output_frame, side)
                if key in frame_keys:
                    raise ValueError(f"duplicate frame-map row for {entry.get('id')}: {key}")
                frame_keys.add(key)
                if strict_value and (
                    expected_metrics_schema == METRICS_SCHEMA_OLD
                    or row.get("phase") == "evaluation"
                ):
                    strict_csv_counts[side] += 1
            for side in ("left", "right"):
                if strict_csv_counts[side] != summary[side]["strict_frames"]:
                    raise ValueError(f"strict CSV count mismatch for {entry.get('id')} {side}")
        except Exception as error:
            failures.append(str(error))
    if orders != list(range(1, EXPECTED_COMPARISONS + 1)):
        failures.append("comparison order must be exactly 1..4")
    videos_dir = root / "videos"
    if videos_dir.is_dir() and not videos_dir.is_symlink():
        actual_video_names = {
            path.name
            for path in videos_dir.iterdir()
            if path.is_file() and not path.is_symlink() and path.suffix.lower() == ".mp4"
        }
        if actual_video_names != set(video_names):
            failures.append("videos directory does not match the canonical manifest")

    checksum_path = root / "SHA256SUMS.txt"
    if not checksum_path.is_file() or checksum_path.is_symlink():
        failures.append("SHA256SUMS.txt is missing")
    else:
        declared: dict[str, str] = {}
        for line_number, line in enumerate(
            checksum_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
            if match is None:
                failures.append(f"malformed checksum line {line_number}")
                continue
            digest, relative = match.groups()
            if relative in declared:
                failures.append(f"duplicate checksum entry: {relative}")
                continue
            declared[relative] = digest
        all_regular: dict[str, Path] = {}
        for path in root.rglob("*"):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                failures.append(f"symlinks are forbidden: {relative}")
            elif path.is_dir():
                continue
            elif not path.is_file():
                failures.append(f"non-regular entries are forbidden: {relative}")
            elif relative != "SHA256SUMS.txt":
                all_regular[relative] = path
        actual = all_regular
        if set(declared) != set(actual):
            failures.append("checksum inventory does not match delivery files")
        else:
            for relative, path in actual.items():
                if _sha256(path) != declared[relative]:
                    failures.append(f"checksum mismatch: {relative}")
    return {
        "schema": VALIDATION_SCHEMA,
        "status": "pass" if not failures else "fail",
        "comparison_count": len(videos),
        "full_decode": full_decode,
        "failures": failures,
    }
