#!/usr/bin/env python3
"""Audit candidate MOCAP-world Z translations against raw metric depth.

The audit deliberately keeps the delivered camera pose and timestamp mapping
frozen.  It samples raw ``mono16`` depth frames across each synchronized take,
interpolates the two 21-joint CMAvatar hands at the depth acquisition time,
and compares projected joint camera-Z with the spatially nearest non-zero raw
depth pixel inside a small image-space radius.

This is a visible-first-surface consistency diagnostic, not joint ground
truth.  In particular, an anatomical joint can be behind the visible skin,
occluded, or projected onto another foreground/background surface.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

import depth_mocap_overlay as depth_overlay
import gt_calib_viz as viz
import mocap_video_overlay as video_overlay


SCHEMA = "gt_calib.mocap_wrist_depth_audit.v1"
DEFAULT_SEGMENTS = ("01", "02", "03")
DEFAULT_OFFSET_MIN_MM = -20.0
DEFAULT_OFFSET_MAX_MM = 70.0
DEFAULT_OFFSET_STEP_MM = 5.0
DEFAULT_EXTRA_OFFSET_MM = 37.0
DEFAULT_SAMPLE_COUNT = 301
DEFAULT_LOCAL_RADIUS_PX = 5.0
DEFAULT_SCREENSHOT_OUTPUT_FRAME = 661
DEFAULT_REVIEW_WIDTH = 960


def _offset_key(value: float) -> str:
    return np.format_float_positional(float(value), trim="-")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    source = Path(path)
    return {
        "path": str(source.resolve()),
        "bytes": source.stat().st_size,
        "sha256": _sha256(source),
    }


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(len(finite)),
        "min": float(np.min(finite)),
        "p05": float(np.percentile(finite, 5)),
        "median": float(np.median(finite)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def _fraction(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


@dataclass
class MetricAccumulator:
    requested_count: int = 0
    positive_camera_depth_count: int = 0
    in_frame_count: int = 0
    exact_nonzero_depth_count: int = 0
    local_nonzero_depth_count: int = 0
    residual_chunks: list[np.ndarray] = field(default_factory=list)

    def add(
        self,
        mask: np.ndarray,
        positive: np.ndarray,
        in_frame: np.ndarray,
        exact: np.ndarray,
        local: np.ndarray,
        residual_mm: np.ndarray,
    ) -> None:
        selected = np.asarray(mask, dtype=bool)
        self.requested_count += int(np.count_nonzero(selected))
        self.positive_camera_depth_count += int(
            np.count_nonzero(selected & positive)
        )
        self.in_frame_count += int(np.count_nonzero(selected & in_frame))
        self.exact_nonzero_depth_count += int(
            np.count_nonzero(selected & exact)
        )
        self.local_nonzero_depth_count += int(
            np.count_nonzero(selected & local)
        )
        values = residual_mm[selected & np.isfinite(residual_mm)]
        if len(values):
            self.residual_chunks.append(values.astype(np.float64, copy=True))

    def merge(self, other: "MetricAccumulator") -> None:
        self.requested_count += other.requested_count
        self.positive_camera_depth_count += other.positive_camera_depth_count
        self.in_frame_count += other.in_frame_count
        self.exact_nonzero_depth_count += other.exact_nonzero_depth_count
        self.local_nonzero_depth_count += other.local_nonzero_depth_count
        self.residual_chunks.extend(other.residual_chunks)

    def summary(self) -> dict[str, Any]:
        residual = (
            np.concatenate(self.residual_chunks)
            if self.residual_chunks
            else np.empty(0, dtype=np.float64)
        )
        absolute = np.abs(residual)
        threshold_payload: dict[str, Any] = {}
        for threshold in (30, 50, 80):
            count = int(np.count_nonzero(absolute <= threshold))
            threshold_payload[str(threshold)] = {
                "count": count,
                "fraction_of_requested": _fraction(count, self.requested_count),
                "fraction_of_in_frame": _fraction(count, self.in_frame_count),
                "fraction_of_local_nonzero": _fraction(
                    count, self.local_nonzero_depth_count
                ),
            }
        return {
            "requested_projected_point_count": self.requested_count,
            "positive_camera_depth_count": self.positive_camera_depth_count,
            "positive_camera_depth_fraction_of_requested": _fraction(
                self.positive_camera_depth_count, self.requested_count
            ),
            "in_frame_count": self.in_frame_count,
            "in_frame_fraction_of_requested": _fraction(
                self.in_frame_count, self.requested_count
            ),
            "exact_nonzero_depth_count": self.exact_nonzero_depth_count,
            "exact_nonzero_depth_fraction_of_requested": _fraction(
                self.exact_nonzero_depth_count, self.requested_count
            ),
            "exact_nonzero_depth_fraction_of_in_frame": _fraction(
                self.exact_nonzero_depth_count, self.in_frame_count
            ),
            "local_nonzero_depth_count": self.local_nonzero_depth_count,
            "local_nonzero_depth_fraction_of_requested": _fraction(
                self.local_nonzero_depth_count, self.requested_count
            ),
            "local_nonzero_depth_fraction_of_in_frame": _fraction(
                self.local_nonzero_depth_count, self.in_frame_count
            ),
            "signed_residual_mm_predicted_camera_z_minus_observed": _distribution(
                residual
            ),
            "absolute_residual_mm": _distribution(absolute),
            "absolute_residual_thresholds_mm": threshold_payload,
        }


def _joint_group_masks() -> dict[str, np.ndarray]:
    # Flattening is side-major: left joints 0..20, then right joints 0..20.
    joint_indices = np.tile(np.arange(len(viz.MOCAP_NAMES)), 2)
    tips = np.isin(joint_indices, (4, 8, 12, 16, 20))
    return {
        "all_42_joints": np.ones(42, dtype=bool),
        "wrists": joint_indices == 0,
        "wrist_plus_four_non_thumb_mcps": np.isin(
            joint_indices, (0, 5, 9, 13, 17)
        ),
        "internal_excluding_fingertips": ~tips,
        "fingertips": tips,
    }


def _sample_indices(first: int, stop: int, count: int) -> np.ndarray:
    if stop <= first:
        raise ValueError(f"Invalid synchronized interval [{first}, {stop})")
    if count <= 0:
        raise ValueError("sample_count must be positive")
    if count > stop - first:
        raise ValueError(
            f"sample_count {count} exceeds interval length {stop - first}"
        )
    indices = np.unique(
        np.rint(np.linspace(first, stop - 1, count)).astype(np.int64)
    )
    if len(indices) != count or indices[0] != first or indices[-1] != stop - 1:
        raise AssertionError("Uniform sample construction lost count or endpoints")
    return indices


def _nearest_nonzero_maps(depth_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    invalid = (np.asarray(depth_mm) == 0).astype(np.uint8)
    distance, labels = cv2.distanceTransformWithLabels(
        invalid,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    valid_values = np.asarray(depth_mm)[np.asarray(depth_mm) > 0]
    if not len(valid_values):
        return distance, np.zeros(depth_mm.shape, dtype=np.uint16)
    if int(labels.min()) < 1 or int(labels.max()) > len(valid_values):
        raise ValueError("OpenCV nearest-depth labels do not index valid pixels")
    return distance, valid_values[labels - 1]


def _evaluate_points(
    points_world_mm: np.ndarray,
    prepared: depth_overlay.PreparedDepthMocap,
    depth_mm: np.ndarray,
    nearest_distance_px: np.ndarray,
    nearest_depth_mm: np.ndarray,
    *,
    local_radius_px: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    points = np.asarray(points_world_mm, dtype=np.float64)
    pixels, positive, camera_z_mm = depth_overlay.project_world_to_depth(
        points, prepared.calibration
    )
    in_frame = positive & depth_overlay._inside_frame(
        pixels,
        prepared.calibration.width,
        prepared.calibration.height,
    )
    exact = np.zeros(in_frame.shape, dtype=bool)
    local = np.zeros(in_frame.shape, dtype=bool)
    residual = np.full(in_frame.shape, np.nan, dtype=np.float64)
    if np.any(in_frame):
        rounded = np.rint(pixels[in_frame]).astype(np.int64)
        sampled = depth_mm[rounded[:, 1], rounded[:, 0]]
        distances = nearest_distance_px[rounded[:, 1], rounded[:, 0]]
        nearest = nearest_depth_mm[rounded[:, 1], rounded[:, 0]].astype(
            np.float64
        )
        exact[in_frame] = sampled > 0
        local_values = distances <= local_radius_px
        local[in_frame] = local_values
        selected_indices = np.flatnonzero(in_frame)
        residual[selected_indices[local_values]] = (
            camera_z_mm[in_frame][local_values] - nearest[local_values]
        )
    return pixels, positive, in_frame, exact, local, residual


def _merge_accumulator_maps(
    destination: Mapping[str, MetricAccumulator],
    source: Mapping[str, MetricAccumulator],
) -> None:
    for key in destination:
        destination[key].merge(source[key])


def _metric_maps(
    offsets_mm: Sequence[float], groups: Mapping[str, np.ndarray]
) -> dict[str, dict[str, MetricAccumulator]]:
    return {
        _offset_key(offset): {name: MetricAccumulator() for name in groups}
        for offset in offsets_mm
    }


def _summarize_metric_maps(
    metrics: Mapping[str, Mapping[str, MetricAccumulator]]
) -> dict[str, Any]:
    return {
        offset: {
            "translation_world_xyz_mm": [0.0, 0.0, float(offset)],
            "groups": {name: value.summary() for name, value in groups.items()},
        }
        for offset, groups in metrics.items()
    }


def _candidate_rankings(
    offset_payload: Mapping[str, Mapping[str, Any]],
    group_names: Iterable[str],
) -> dict[str, Any]:
    rankings: dict[str, Any] = {}
    for group_name in group_names:
        candidates = [
            (float(offset), payload["groups"][group_name])
            for offset, payload in offset_payload.items()
        ]
        by_median = min(
            candidates,
            key=lambda item: (
                item[1]["absolute_residual_mm"]["median"],
                abs(item[0]),
                item[0],
            ),
        )
        by_inlier = max(
            candidates,
            key=lambda item: (
                item[1]["absolute_residual_thresholds_mm"]["50"][
                    "fraction_of_requested"
                ],
                -abs(item[0]),
                -item[0],
            ),
        )
        rankings[group_name] = {
            "minimum_local_valid_median_absolute_residual": {
                "offset_world_z_mm": by_median[0],
                "median_absolute_residual_mm": by_median[1][
                    "absolute_residual_mm"
                ]["median"],
                "warning": (
                    "This ranking conditions on a local non-zero depth hit and "
                    "does not penalize missing or occluded projections."
                ),
            },
            "maximum_fraction_of_requested_with_absolute_residual_at_most_50mm": {
                "offset_world_z_mm": by_inlier[0],
                "fraction": by_inlier[1]["absolute_residual_thresholds_mm"][
                    "50"
                ]["fraction_of_requested"],
            },
        }
    return rankings


def _point_rows(
    pixels: np.ndarray,
    camera_z_mm: np.ndarray,
    depth_mm: np.ndarray,
    nearest_distance_px: np.ndarray,
    nearest_depth_mm: np.ndarray,
    *,
    local_radius_px: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for side_index, side in enumerate(("left", "right")):
        pixel = pixels[side_index]
        rounded = np.rint(pixel).astype(np.int64)
        u, vv = int(rounded[0]), int(rounded[1])
        inside = 0 <= u < depth_mm.shape[1] and 0 <= vv < depth_mm.shape[0]
        nearest_value = int(nearest_depth_mm[vv, u]) if inside else 0
        has_nearest_nonzero = inside and nearest_value > 0
        nearest_distance = (
            float(nearest_distance_px[vv, u]) if has_nearest_nonzero else None
        )
        observed = nearest_value if has_nearest_nonzero else None
        accepted = bool(
            has_nearest_nonzero
            and nearest_distance is not None
            and nearest_distance <= local_radius_px
        )
        predicted = float(camera_z_mm[side_index])
        rows.append(
            {
                "side": side,
                "projected_depth_pixel_xy": [float(pixel[0]), float(pixel[1])],
                "predicted_camera_z_mm": predicted,
                "exact_raw_depth_mm": int(depth_mm[vv, u]) if inside else None,
                "nearest_nonzero_distance_px": nearest_distance,
                "nearest_nonzero_depth_mm": observed,
                "nearest_nonzero_accepted_within_local_radius": accepted,
                "signed_residual_mm": (
                    predicted - observed if accepted and observed is not None else None
                ),
            }
        )
    return rows


def _take02_frame661_payload(
    calibration_path: Path,
    prepared_depth: depth_overlay.PreparedDepthMocap,
    *,
    local_radius_px: float,
) -> dict[str, Any]:
    prepared_rgb = video_overlay.prepare_mocap_video(
        prepared_depth.segment_dir,
        calibration_path,
        max_interpolation_gap_ms=prepared_depth.max_interpolation_gap_ms,
    )
    first_rgb, _ = video_overlay.synchronized_interval(prepared_rgb.mocap_valid)
    output_frame = DEFAULT_SCREENSHOT_OUTPUT_FRAME
    source_rgb_frame = first_rgb + output_frame
    if source_rgb_frame >= len(prepared_rgb.mocap_mm):
        raise ValueError("Take 02 does not contain requested screenshot frame")

    base_rgb = prepared_rgb.mocap_mm[source_rgb_frame]
    shifted_rgb = base_rgb.copy()
    shifted_rgb[..., 2] += DEFAULT_EXTRA_OFFSET_MM
    proxy_rgb = video_overlay.visualization_rear_wrist_mount_anchor(base_rgb)
    base_pixels, _ = viz.project_world_to_rgb(base_rgb, prepared_rgb.calibration)
    shifted_pixels, _ = viz.project_world_to_rgb(
        shifted_rgb, prepared_rgb.calibration
    )
    proxy_pixels, _ = viz.project_world_to_rgb(proxy_rgb, prepared_rgb.calibration)
    source_width = prepared_rgb.video.width
    scale = DEFAULT_REVIEW_WIDTH / source_width

    source_depth_frame = source_rgb_frame
    frame = next(
        depth_overlay.iter_bag_depth_frames(
            prepared_depth.bag_path,
            prepared_depth.depth_refs,
            source_depth_frame,
            source_depth_frame + 1,
        )
    )
    nearest_distance, nearest_depth = _nearest_nonzero_maps(frame.depth_mm)
    base_depth = prepared_depth.mocap_mm[source_depth_frame]
    shifted_depth = base_depth.copy()
    shifted_depth[..., 2] += DEFAULT_EXTRA_OFFSET_MM
    proxy_depth = video_overlay.visualization_rear_wrist_mount_anchor(base_depth)

    depth_rows: dict[str, Any] = {}
    for name, points in (
        ("wrist_offset_0", base_depth[:, 0]),
        ("wrist_offset_plus_37", shifted_depth[:, 0]),
        ("rear_mount_proxy", proxy_depth),
    ):
        pixels, _, camera_z = depth_overlay.project_world_to_depth(
            points, prepared_depth.calibration
        )
        depth_rows[name] = _point_rows(
            pixels,
            camera_z,
            frame.depth_mm,
            nearest_distance,
            nearest_depth,
            local_radius_px=local_radius_px,
        )

    def rgb_rows(pixels: np.ndarray) -> list[dict[str, Any]]:
        return [
            {
                "side": side,
                "source_rgb_pixel_xy": [
                    float(pixels[index, 0]),
                    float(pixels[index, 1]),
                ],
                "review_960_pixel_xy": [
                    float(pixels[index, 0] * scale),
                    float(pixels[index, 1] * scale),
                ],
            }
            for index, side in enumerate(("left", "right"))
        ]

    joint_shift = shifted_pixels - base_pixels
    return {
        "segment": prepared_depth.segment_dir.name,
        "output_frame": output_frame,
        "source_rgb_frame": source_rgb_frame,
        "source_depth_frame_same_index": source_depth_frame,
        "rgb_source_resolution": [
            prepared_rgb.video.width,
            prepared_rgb.video.height,
        ],
        "review_width": DEFAULT_REVIEW_WIDTH,
        "nearest_nonzero_depth_radius_px": local_radius_px,
        "rgb_elapsed_s": float(
            prepared_rgb.rgb_device_s[source_rgb_frame]
            - prepared_rgb.rgb_device_s[first_rgb]
        ),
        "depth_minus_same_index_rgb_timestamp_ms": float(
            (
                prepared_depth.depth_timestamp_ns[source_depth_frame]
                - prepared_depth.rgb_timestamp_ns[source_depth_frame]
            )
            / 1e6
        ),
        "plus_37_all_joint_rgb_shift_px_at_source_resolution": {
            "x_min": float(np.min(joint_shift[..., 0])),
            "x_median": float(np.median(joint_shift[..., 0])),
            "x_max": float(np.max(joint_shift[..., 0])),
            "y_min": float(np.min(joint_shift[..., 1])),
            "y_median": float(np.median(joint_shift[..., 1])),
            "y_max": float(np.max(joint_shift[..., 1])),
        },
        "rgb_wrist_or_proxy_projection": {
            "wrist_offset_0": rgb_rows(base_pixels[:, 0]),
            "wrist_offset_plus_37": rgb_rows(shifted_pixels[:, 0]),
            "rear_mount_proxy": rgb_rows(proxy_pixels),
        },
        "raw_depth_same_index_diagnostic": depth_rows,
        "note": (
            "RGB and depth source frames share the validated same-index pairing, "
            "but depth is acquired about 1.31 ms after RGB and therefore uses its "
            "own timestamp-interpolated MOCAP pose."
        ),
    }


def run_audit(
    dataset_root: Path,
    calibration_path: Path,
    segment_keys: Sequence[str],
    *,
    offsets_mm: Sequence[float],
    sample_count: int,
    local_radius_px: float,
    max_interpolation_gap_ms: float,
) -> dict[str, Any]:
    if local_radius_px <= 0.0:
        raise ValueError("local_radius_px must be positive")
    groups = _joint_group_masks()
    aggregate = _metric_maps(offsets_mm, groups)
    aggregate_proxy = MetricAccumulator()
    takes: dict[str, Any] = {}
    prepared_take02: depth_overlay.PreparedDepthMocap | None = None

    for segment_key in segment_keys:
        segment = viz._discover_segment(dataset_root, segment_key)
        print(f"Preparing {segment.name}", flush=True)
        prepared = depth_overlay.prepare_depth_mocap(
            segment,
            calibration_path,
            max_interpolation_gap_ms=max_interpolation_gap_ms,
        )
        first, stop = video_overlay.synchronized_interval(prepared.mocap_valid)
        indices = _sample_indices(first, stop, sample_count)
        take_metrics = _metric_maps(offsets_mm, groups)
        take_proxy = MetricAccumulator()
        sampled_depth_digest = hashlib.sha256()

        for sample_ordinal, source_frame in enumerate(indices):
            frame = next(
                depth_overlay.iter_bag_depth_frames(
                    prepared.bag_path,
                    prepared.depth_refs,
                    int(source_frame),
                    int(source_frame) + 1,
                )
            )
            sampled_depth_digest.update(segment.name.encode("utf-8"))
            sampled_depth_digest.update(int(source_frame).to_bytes(8, "little"))
            sampled_depth_digest.update(
                int(frame.message_timestamp_ns).to_bytes(8, "little")
            )
            sampled_depth_digest.update(np.ascontiguousarray(frame.depth_mm).tobytes())
            nearest_distance, nearest_depth = _nearest_nonzero_maps(frame.depth_mm)
            base = prepared.mocap_mm[int(source_frame)].reshape(42, 3)

            for offset in offsets_mm:
                shifted = base.copy()
                shifted[:, 2] += float(offset)
                _, positive, in_frame, exact, local, residual = _evaluate_points(
                    shifted,
                    prepared,
                    frame.depth_mm,
                    nearest_distance,
                    nearest_depth,
                    local_radius_px=local_radius_px,
                )
                offset_metrics = take_metrics[_offset_key(offset)]
                for group_name, group_mask in groups.items():
                    offset_metrics[group_name].add(
                        group_mask,
                        positive,
                        in_frame,
                        exact,
                        local,
                        residual,
                    )

            proxy = video_overlay.visualization_rear_wrist_mount_anchor(
                prepared.mocap_mm[int(source_frame)]
            )
            _, positive, in_frame, exact, local, residual = _evaluate_points(
                proxy,
                prepared,
                frame.depth_mm,
                nearest_distance,
                nearest_depth,
                local_radius_px=local_radius_px,
            )
            take_proxy.add(
                np.ones(2, dtype=bool),
                positive,
                in_frame,
                exact,
                local,
                residual,
            )
            if (sample_ordinal + 1) % 100 == 0:
                print(
                    f"  {segment.name}: {sample_ordinal + 1}/{len(indices)} sampled",
                    flush=True,
                )

        for offset in aggregate:
            _merge_accumulator_maps(aggregate[offset], take_metrics[offset])
        aggregate_proxy.merge(take_proxy)

        take_number = viz._take_number(segment)
        clock_path = segment / "同步校验" / "camera_cmavatar_alignment.csv"
        common_path = segment / "同步校验" / "common_interval_sync_report.json"
        intrinsics_path = (
            segment / "原始BAG与内参" / "camera_1_intrinsics.json"
        )
        take_offset_payload = _summarize_metric_maps(take_metrics)
        takes[segment.name] = {
            "take_number": take_number,
            "synchronized_source_interval_half_open": [first, stop],
            "sample_count": int(len(indices)),
            "sample_source_frame_indices": [int(value) for value in indices],
            "sampling": (
                "unique rint(linspace(first, stop - 1, sample_count)); "
                "endpoints included"
            ),
            "sampled_decoded_depth_payload_sha256": sampled_depth_digest.hexdigest(),
            "inputs": {
                "raw_rgbd_bag": _artifact(prepared.bag_path),
                "camera_intrinsics": _artifact(intrinsics_path),
                "clock_mapping": _artifact(clock_path),
                "common_interval_sync_report": _artifact(common_path),
                "mocap_human": _artifact(prepared.mocap_path),
            },
            "offsets": take_offset_payload,
            "candidate_rankings": _candidate_rankings(
                take_offset_payload, groups.keys()
            ),
            "rear_mount_proxy": {
                "definition": "wrist - 0.8 * (middle_mcp - wrist)",
                "original_21_joint_positions_modified": False,
                "metrics": take_proxy.summary(),
            },
        }
        if segment_key == "02" or segment.name.startswith("02_"):
            prepared_take02 = prepared

    offset_payload = _summarize_metric_maps(aggregate)
    zero_wrist = offset_payload["0"]["groups"]["wrists"]
    plus37_wrist = offset_payload["37"]["groups"]["wrists"]
    zero_palm = offset_payload["0"]["groups"][
        "wrist_plus_four_non_thumb_mcps"
    ]
    plus37_palm = offset_payload["37"]["groups"][
        "wrist_plus_four_non_thumb_mcps"
    ]
    proxy_summary = aggregate_proxy.summary()

    calibration = depth_overlay.load_depth_calibration(calibration_path)
    rgb_calibration = viz.load_preview_calibration(calibration_path)
    plus37_world = np.asarray([0.0, 0.0, DEFAULT_EXTRA_OFFSET_MM])
    depth_effect = calibration.world_to_depth[:3, :3] @ plus37_world
    rgb_effect = rgb_calibration.world_to_color[:3, :3] @ plus37_world

    return {
        "schema": SCHEMA,
        "artifact_type": "raw_depth_visible_surface_consistency_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "completed_candidate_scan_not_ground_truth",
        "calibration": _artifact(calibration_path),
        "frozen_contract": {
            "world_to_depth_camera_frozen_no_refit": True,
            "depth_timestamp_used_directly_not_rgb_timestamp_substituted": True,
            "mocap_interpolation": (
                "linear interpolation of Human.cma 120 Hz positions at each raw "
                "depth acquisition timestamp"
            ),
            "max_interpolation_gap_ms": max_interpolation_gap_ms,
            "candidate_translation_applied_after_timestamp_interpolation": True,
            "candidate_translation_changes_bone_lengths": False,
        },
        "scan": {
            "segments": [str(value) for value in segment_keys],
            "uniform_samples_per_take": sample_count,
            "total_uniform_sampled_depth_frames": sample_count * len(segment_keys),
            "offsets_world_z_mm": [float(value) for value in offsets_mm],
            "local_nearest_nonzero_radius_px": local_radius_px,
            "joint_order_per_side": list(viz.MOCAP_NAMES),
            "flattening": "left 21 joints followed by right 21 joints",
            "groups": {
                name: [int(value) for value in np.flatnonzero(mask)]
                for name, mask in groups.items()
            },
        },
        "formula": {
            "candidate_world_point": "p_W(delta_z) = p_W + [0, 0, delta_z] mm",
            "frozen_camera_transform": "p_D = R_D_from_W * p_W(delta_z) + t_D_from_W",
            "projection": (
                "(u,v) = OrbbecDistort(K * [p_D.x/p_D.z, p_D.y/p_D.z]); "
                "evaluate round(u,v)"
            ),
            "observed_surface": (
                "spatially nearest non-zero raw mono16 millimetre pixel to "
                "round(u,v), accepted only when OpenCV L2 distanceTransform "
                f"distance <= {local_radius_px:g} px"
            ),
            "signed_residual": "r_mm = predicted p_D.z - observed raw depth mm",
            "signed_residual_interpretation": (
                "positive means the projected point is farther from the camera "
                "than the observed first surface; negative means it is nearer"
            ),
            "plus_37_camera_frame_translation_mm": {
                "depth_camera_xyz": [float(value) for value in depth_effect],
                "color_camera_xyz": [float(value) for value in rgb_effect],
            },
        },
        "takes": takes,
        "aggregate": {
            "offsets": offset_payload,
            "candidate_rankings": _candidate_rankings(
                offset_payload, groups.keys()
            ),
            "rear_mount_proxy": {
                "definition": "wrist - 0.8 * (middle_mcp - wrist)",
                "hand_local_pose_dependent": True,
                "original_21_joint_positions_modified": False,
                "additional_measured_mocap_joint": False,
                "metrics": proxy_summary,
            },
        },
        "take02_output661": (
            _take02_frame661_payload(
                calibration_path,
                prepared_take02,
                local_radius_px=local_radius_px,
            )
            if prepared_take02 is not None
            else None
        ),
        "decision": {
            "activate_global_world_z_plus_37_as_ground_truth": False,
            "activate_global_world_z_plus_37_as_default_overlay": False,
            "recommended_visualization": "rear_mount_proxy_plus_original_21_joints",
            "recommended_original_joint_translation_world_xyz_mm": [0.0, 0.0, 0.0],
            "rear_mount_proxy_must_be_labeled": "VIZ-only, not GT",
            "evidence": {
                "raw_wrist_offset_0": zero_wrist,
                "raw_wrist_offset_plus_37": plus37_wrist,
                "wrist_plus_four_mcps_offset_0": zero_palm,
                "wrist_plus_four_mcps_offset_plus_37": plus37_palm,
                "rear_mount_proxy": proxy_summary,
            },
            "reason": (
                "+37 mm can move fingertips onto some foreground surfaces, but "
                "it introduces a large negative visible-surface residual at wrists "
                "and palm anchors. The hand-local rear proxy reaches the visible "
                "rear module region without modifying the delivered 21-joint hand."
            ),
        },
        "limits": [
            "Raw depth records the first visible surface on each camera ray, not an anatomical joint centre.",
            "Human.cma has no per-joint visibility, occlusion, residual, or gap-fill quality field.",
            "Self-occluded joints and fingertips can project onto another finger, the palm, clothing, table, or background.",
            f"A {local_radius_px:g} px nearest-valid search can bridge sensor holes but can also select a neighbouring surface.",
            "Black or reflective wrist modules can produce invalid depth; non-zero coverage alone is not accuracy.",
            "The rear mount proxy is extrapolated from wrist and middle MCP; it is not a measured rigid-body/module pose.",
            "The scan uses the available three recordings and is not an independent labelled calibration/holdout experiment.",
            "Neither the lowest residual candidate nor the proxy may be promoted to camera-to-MOCAP ground truth without independent module/joint labels or a measured rigid mount transform.",
        ],
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Audit MOCAP world-Z candidates against synchronized raw depth."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "同步整理_20260829_三段",
    )
    parser.add_argument("--calibration", type=Path)
    parser.add_argument(
        "--segment",
        action="append",
        default=[],
        help="01, 02, or 03; repeatable. Default audits all three.",
    )
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument(
        "--local-radius-px", type=float, default=DEFAULT_LOCAL_RADIUS_PX
    )
    parser.add_argument(
        "--max-interpolation-gap-ms", type=float, default=25.0
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "outputs"
        / "mocap_wrist_depth_audit"
        / "depth_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    calibration = (
        args.calibration.expanduser().resolve()
        if args.calibration is not None
        else viz._default_calibration(dataset_root)
    )
    offsets = sorted(
        {
            *np.arange(
                DEFAULT_OFFSET_MIN_MM,
                DEFAULT_OFFSET_MAX_MM + DEFAULT_OFFSET_STEP_MM / 2.0,
                DEFAULT_OFFSET_STEP_MM,
            ).tolist(),
            DEFAULT_EXTRA_OFFSET_MM,
        }
    )
    payload = run_audit(
        dataset_root,
        calibration,
        args.segment or DEFAULT_SEGMENTS,
        offsets_mm=offsets,
        sample_count=args.sample_count,
        local_radius_px=args.local_radius_px,
        max_interpolation_gap_ms=args.max_interpolation_gap_ms,
    )
    output = args.output.expanduser().resolve()
    _write_json_atomic(output, payload)
    print(f"Wrote {output}")
    print(f"SHA256 {_sha256(output)}")


if __name__ == "__main__":
    main()
