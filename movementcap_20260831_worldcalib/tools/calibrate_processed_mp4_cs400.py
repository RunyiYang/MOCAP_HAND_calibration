#!/usr/bin/env python3
"""Calibrate the 2026-08-31 processed MP4 set from a visually labelled CS-400.

The processed delivery contains RGB.mp4 and a lossy JET-colored Depth.mp4, but
not the original 16-bit depth BAG.  The metric camera/world transform is solved
from the four CS-400 marker centers in RGB and their known ruler geometry.  The
same camera's previously recorded factory depth-to-color transform is then used
to express the result in depth-camera coordinates.

The world origin is the CS-400 vertex axis projected onto the tabletop.  The
marker plane and black vertex hole are 45 mm above world Z=0.  Any point cloud
exported here is explicitly approximate because it decodes the lossy preview.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from calibrate_manual_cs400 import json_ready, pixel_rays, project_camera_points
from export_world_point_cloud import (
    axis_point_samples,
    render_world_preview,
    transform_points,
    write_axes_line_ply,
    write_binary_ply,
)


AXIS_COLORS_BGR = ((45, 45, 255), (70, 230, 45), (255, 130, 60))
MARKER_NAMES = ("long_far", "long_near", "short_near", "short_far")


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    workspace = project.parent
    parser = argparse.ArgumentParser(
        description="Calibrate a processed RGB/Depth MP4 set from a visual CS-400 observation."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=workspace / "20260831_processed" / "thor_new4_20260831_processed",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project / "configs" / "manual_cs400_points_20260831.json",
    )
    parser.add_argument(
        "--prior-camera-calibration",
        type=Path,
        default=project / "results" / "manual_final" / "camera_to_world.json",
        help="Same-device calibration containing the factory depth-to-color rotation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project / "results" / "20260831_processed_tabletop_origin",
    )
    parser.add_argument("--axis-length-mm", type=float, default=300.0)
    parser.add_argument("--max-points", type=int, default=400_000)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_video_frame(path: Path, frame_index: int) -> tuple[np.ndarray, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if frame_index < 0 or frame_index >= frame_count:
        capture.release()
        raise IndexError(f"Frame {frame_index} outside [0, {frame_count}) for {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed reading frame {frame_index} from {path}")
    return frame, frame_count


def camera_parameters(intrinsics_json: dict[str, Any]) -> dict[str, np.ndarray]:
    camera = intrinsics_json["camera_param"]
    rgb = camera["rgb_intrinsic"]
    depth = camera["depth_intrinsic"]
    depth_distortion = camera["depth_distortion"]
    color_distortion = camera["rgb_distortion"]
    return {
        "color_matrix": np.asarray(rgb["camera_matrix"], dtype=np.float64),
        "color_intrinsics": np.asarray(
            [rgb["fx"], rgb["fy"], rgb["cx"], rgb["cy"]], dtype=np.float64
        ),
        "color_distortion_opencv": np.asarray(
            color_distortion["opencv_coefficients"], dtype=np.float64
        ),
        "color_distortion_orbbec": np.asarray(
            [
                color_distortion["k1"],
                color_distortion["k2"],
                color_distortion["k3"],
                color_distortion["k4"],
                color_distortion["k5"],
                color_distortion["k6"],
                color_distortion["p1"],
                color_distortion["p2"],
            ],
            dtype=np.float64,
        ),
        "depth_intrinsics": np.asarray(
            [depth["fx"], depth["fy"], depth["cx"], depth["cy"]], dtype=np.float64
        ),
        "depth_distortion_orbbec": np.asarray(
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
        ),
    }


def homogeneous(linear: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(linear, dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return matrix


def solve_color_pose(
    config: dict[str, Any], parameters: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, Any]]:
    object_points = np.asarray(
        [config["marker_centers_world_mm"][name] for name in MARKER_NAMES],
        dtype=np.float64,
    )
    image_points = np.asarray(
        [config["rgb_marker_centers_px"][name] for name in MARKER_NAMES],
        dtype=np.float64,
    )
    ok, rotation_vector, translation = cv2.solvePnP(
        object_points,
        image_points,
        parameters["color_matrix"],
        parameters["color_distortion_opencv"],
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("CS-400 solvePnP failed")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation,
        parameters["color_matrix"],
        parameters["color_distortion_opencv"],
    )
    projected = projected[:, 0]
    residual = np.linalg.norm(projected - image_points, axis=1)
    color_from_world = homogeneous(rotation, translation[:, 0])
    camera_origin_world = -rotation.T @ translation[:, 0]
    if camera_origin_world[2] <= 0.0:
        raise RuntimeError("PnP selected a physically invalid below-table camera solution")
    quality = {
        "solver": "OpenCV SOLVEPNP_ITERATIVE",
        "marker_order": MARKER_NAMES,
        "observed_rgb_pixels": image_points,
        "reprojected_rgb_pixels": projected,
        "per_marker_reprojection_error_px": {
            name: float(value) for name, value in zip(MARKER_NAMES, residual, strict=True)
        },
        "reprojection_rms_px": float(np.sqrt(np.mean(residual**2))),
        "reprojection_max_px": float(np.max(residual)),
        "color_camera_origin_in_world_mm": camera_origin_world,
        "proper_rotation_determinant": float(np.linalg.det(rotation)),
    }
    return color_from_world, quality


def load_same_camera_depth_to_color(
    prior_path: Path,
    prior: dict[str, Any],
    intrinsics_json: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    serial = intrinsics_json["device"]["serial_number"]
    consistency = prior.get("camera_consistency", {})
    known_serials = set(
        consistency.get("serial_numbers", consistency.get("device_serial_numbers", []))
    )
    if serial not in known_serials:
        raise RuntimeError(
            f"Prior calibration {prior_path} does not document current device serial {serial}"
        )
    transform = np.asarray(
        prior["transforms"]["depth_camera_to_color_camera"], dtype=np.float64
    )
    current_translation = np.asarray(
        intrinsics_json["camera_param"]["depth_to_color_extrinsic"]["translation_mm"],
        dtype=np.float64,
    )
    if not np.allclose(transform[:3, 3], current_translation, atol=1e-6):
        raise RuntimeError("Current and prior depth-to-color translations do not match")
    quality = {
        "source": str(prior_path.resolve()),
        "device_serial_number": serial,
        "translation_match": True,
        "recorded_linear_determinant": float(np.linalg.det(transform[:3, :3])),
        "recorded_linear_orthogonality_max_error": float(
            np.max(np.abs(transform[:3, :3].T @ transform[:3, :3] - np.eye(3)))
        ),
        "reason": (
            "The processed intrinsics JSON retains the translation but omits the rotation. "
            "The factory transform is reused from a raw BAG calibration of the same serial-numbered device."
        ),
    }
    return transform, quality


def project_world_points(
    points_world: np.ndarray,
    color_from_world: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    points_color = transform_points(points, color_from_world)
    return project_camera_points(
        points_color,
        parameters["color_intrinsics"],
        parameters["color_distortion_orbbec"],
    )


def dashed_line(
    image: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    dash_px: float = 18.0,
) -> None:
    vector = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length < 1.0:
        return
    direction = vector / length
    cursor = 0.0
    while cursor < length:
        a = np.asarray(start) + direction * cursor
        b = np.asarray(start) + direction * min(length, cursor + dash_px)
        cv2.line(
            image,
            tuple(np.rint(a).astype(int)),
            tuple(np.rint(b).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )
        cursor += 2.0 * dash_px


def draw_axes_overlay(
    image: np.ndarray,
    color_from_world: np.ndarray,
    parameters: dict[str, np.ndarray],
    *,
    axis_length_mm: float,
    marker_height_mm: float,
    observations: dict[str, list[float]] | None,
    show_height_guide: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = image.copy()
    world_points = np.asarray(
        [
            [0, 0, 0],
            [axis_length_mm, 0, 0],
            [0, axis_length_mm, 0],
            [0, 0, axis_length_mm],
            [0, 0, marker_height_mm],
            [axis_length_mm, 0, marker_height_mm],
            [0, axis_length_mm, marker_height_mm],
        ],
        dtype=np.float64,
    )
    pixels = project_world_points(world_points, color_from_world, parameters)
    rounded = np.rint(pixels).astype(int)
    origin = rounded[0]
    for endpoint, color, name in zip(
        rounded[1:4], AXIS_COLORS_BGR, ("+X", "+Y", "+Z"), strict=True
    ):
        cv2.arrowedLine(
            result,
            tuple(origin),
            tuple(endpoint),
            color,
            7,
            cv2.LINE_AA,
            tipLength=0.08,
        )
        cv2.putText(
            result,
            name,
            tuple(endpoint + np.asarray([8, -8])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            3,
            cv2.LINE_AA,
        )
    cv2.circle(result, tuple(origin), 11, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(
        result,
        "O: tabletop Z=0",
        tuple(origin + np.asarray([16, 12])),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if show_height_guide:
        marker_origin = rounded[4]
        dashed_line(result, pixels[4], pixels[5], (170, 170, 255), 4)
        dashed_line(result, pixels[4], pixels[6], (170, 255, 170), 4)
        cv2.circle(result, tuple(marker_origin), 9, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(result, tuple(marker_origin), tuple(origin), (255, 220, 80), 3, cv2.LINE_AA)
        cv2.putText(
            result,
            "black-hole axis / marker plane (+45 mm)",
            tuple(marker_origin + np.asarray([18, -15])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if observations:
        for name, point in observations.items():
            center = tuple(np.rint(point).astype(int))
            cv2.circle(result, center, 12, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(
                result,
                name,
                (center[0] + 10, center[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    metadata = {
        "tabletop_origin_rgb_pixel": pixels[0],
        "x_endpoint_rgb_pixel": pixels[1],
        "y_endpoint_rgb_pixel": pixels[2],
        "z_endpoint_rgb_pixel": pixels[3],
        "marker_plane_origin_rgb_pixel": pixels[4],
        "marker_plane_x_endpoint_rgb_pixel": pixels[5],
        "marker_plane_y_endpoint_rgb_pixel": pixels[6],
    }
    return result, metadata


def nearest_jet_indices(depth_preview_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    palette = cv2.applyColorMap(
        np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_JET
    ).reshape(256, 3).astype(np.int32)
    pixels = depth_preview_bgr.reshape(-1, 3).astype(np.int32)
    indices = np.empty(len(pixels), dtype=np.uint8)
    errors = np.empty(len(pixels), dtype=np.float32)
    for start in range(0, len(pixels), 30_000):
        values = pixels[start : start + 30_000]
        distances = np.sum((values[:, None, :] - palette[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(distances, axis=1)
        indices[start : start + len(values)] = nearest
        errors[start : start + len(values)] = np.sqrt(
            distances[np.arange(len(values)), nearest]
        )
    shape = depth_preview_bgr.shape[:2]
    return indices.reshape(shape), errors.reshape(shape)


def expected_table_geometry(
    image_shape: tuple[int, int],
    world_from_depth: np.ndarray,
    depth_from_color: np.ndarray,
    parameters: dict[str, np.ndarray],
    rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image_shape
    vv, uu = np.mgrid[:height, :width]
    rays = pixel_rays(
        uu,
        vv,
        parameters["depth_intrinsics"],
        parameters["depth_distortion_orbbec"],
    )
    denominator = np.einsum("...j,j->...", rays, world_from_depth[2, :3])
    expected_z = -world_from_depth[2, 3] / denominator
    points_depth = rays * expected_z[..., None]
    color_from_depth = np.linalg.inv(depth_from_color)
    points_color = transform_points(points_depth.reshape(-1, 3), color_from_depth).reshape(
        height, width, 3
    )
    color_pixels = project_camera_points(
        points_color.reshape(-1, 3),
        parameters["color_intrinsics"],
        parameters["color_distortion_orbbec"],
    ).reshape(height, width, 2)
    color_u = np.rint(color_pixels[..., 0]).astype(np.int64)
    color_v = np.rint(color_pixels[..., 1]).astype(np.int64)
    inside = (
        (color_u >= 0)
        & (color_u < rgb.shape[1])
        & (color_v >= 0)
        & (color_v < rgb.shape[0])
        & (expected_z > 650.0)
        & (expected_z < 1300.0)
    )
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    hsv[inside] = rgb_hsv[color_v[inside], color_u[inside]]
    green_table = (
        inside
        & (hsv[..., 0] >= 72)
        & (hsv[..., 0] <= 90)
        & (hsv[..., 1] >= 45)
        & (hsv[..., 2] >= 70)
    )
    return expected_z, green_table, rays


def fit_preview_depth_mapping(
    jet_indices: np.ndarray,
    palette_error: np.ndarray,
    expected_table_z: np.ndarray,
    green_table: np.ndarray,
) -> tuple[float, float, np.ndarray, dict[str, Any]]:
    fit_mask = (
        green_table
        & (palette_error < 12.0)
        & (jet_indices > 2)
        & (jet_indices < 250)
    )
    x = expected_table_z[fit_mask]
    y = jet_indices[fit_mask].astype(np.float64)
    if len(x) < 1000:
        raise RuntimeError(f"Only {len(x)} table samples available for preview-depth fit")
    keep = np.ones(len(x), dtype=bool)
    for _ in range(10):
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        residual = y - (slope * x + intercept)
        center = float(np.median(residual[keep]))
        mad = 1.4826 * float(np.median(np.abs(residual[keep] - center)))
        updated = np.abs(residual - center) < max(2.5, 3.0 * mad)
        if np.array_equal(updated, keep):
            break
        keep = updated
    residual = y - (slope * x + intercept)
    depth_error = residual[keep] / slope
    stats = {
        "mapping": "jet_index = slope_per_mm * depth_mm + intercept",
        "slope_per_mm": float(slope),
        "intercept": float(intercept),
        "millimetres_per_color_index": float(1.0 / slope),
        "candidate_table_pixels": int(np.sum(fit_mask)),
        "robust_fit_pixels": int(np.sum(keep)),
        "table_fit_depth_error_mm_p50": float(np.percentile(np.abs(depth_error), 50)),
        "table_fit_depth_error_mm_p95": float(np.percentile(np.abs(depth_error), 95)),
        "warning": (
            "This mapping recovers a lossy 8-bit preview, not original 16-bit depth. "
            "It is used only for the diagnostic point cloud."
        ),
    }
    return float(slope), float(intercept), fit_mask, stats


def decode_approximate_depth(
    jet_indices: np.ndarray,
    palette_error: np.ndarray,
    slope: float,
    intercept: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth_mm = (jet_indices.astype(np.float64) - intercept) / slope
    valid = (
        (palette_error < 35.0)
        & (jet_indices > 2)
        & (jet_indices < 250)
        & (depth_mm >= 400.0)
        & (depth_mm <= 4500.0)
    )
    depth_mm[~valid] = 0.0
    return depth_mm, valid


def make_depth_diagnostics(
    depth_preview: np.ndarray,
    approximate_depth_mm: np.ndarray,
    fit_mask: np.ndarray,
    output_dir: Path,
) -> None:
    overlay = depth_preview.copy()
    tint = np.zeros_like(overlay)
    tint[..., 1] = 255
    overlay[fit_mask] = cv2.addWeighted(
        overlay[fit_mask], 0.45, tint[fit_mask], 0.55, 0.0
    )
    cv2.putText(
        overlay,
        "green: tabletop pixels used to fit the lossy depth preview",
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_dir / "reference_depth_table_fit_mask.png"), overlay)
    encoded = np.clip(np.rint(approximate_depth_mm), 0, 65535).astype(np.uint16)
    cv2.imwrite(str(output_dir / "reference_approx_depth_mm.png"), encoded)
    normalized = np.clip((approximate_depth_mm - 400.0) / 3600.0, 0.0, 1.0)
    preview = cv2.applyColorMap(
        (255.0 * normalized).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    preview[approximate_depth_mm <= 0.0] = 0
    cv2.putText(
        preview,
        "APPROXIMATE: decoded from lossy JET Depth.mp4",
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_dir / "reference_approx_depth_preview.png"), preview)


def build_approximate_point_cloud(
    depth_mm: np.ndarray,
    valid_depth: np.ndarray,
    rgb: np.ndarray,
    parameters: dict[str, np.ndarray],
    color_from_depth: np.ndarray,
    world_from_depth: np.ndarray,
    output_dir: Path,
    *,
    axis_length_mm: float,
    max_points: int,
) -> dict[str, Any]:
    vv, uu = np.mgrid[: depth_mm.shape[0], : depth_mm.shape[1]]
    u = uu[valid_depth].astype(np.float64)
    v = vv[valid_depth].astype(np.float64)
    z = depth_mm[valid_depth]
    rays = pixel_rays(
        u,
        v,
        parameters["depth_intrinsics"],
        parameters["depth_distortion_orbbec"],
    )
    points_depth = rays * z[:, None]
    points_color = transform_points(points_depth, color_from_depth)
    color_pixels = project_camera_points(
        points_color,
        parameters["color_intrinsics"],
        parameters["color_distortion_orbbec"],
    )
    rounded = np.rint(color_pixels).astype(np.int64)
    inside = (
        (points_color[:, 2] > 0.0)
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < rgb.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < rgb.shape[0])
    )
    points_depth = points_depth[inside]
    rounded = rounded[inside]
    colors_rgb = rgb[rounded[:, 1], rounded[:, 0], ::-1].copy()
    if len(points_depth) > max_points:
        keep = np.linspace(0, len(points_depth) - 1, max_points).astype(np.int64)
        points_depth = points_depth[keep]
        colors_rgb = colors_rgb[keep]
    points_world = transform_points(points_depth, world_from_depth)
    axis_points, axis_colors = axis_point_samples(axis_length_mm)

    depth_path = output_dir / "reference_approx_depth_camera_coordinates.ply"
    world_path = output_dir / "reference_approx_world_coordinates.ply"
    world_axes_path = output_dir / "reference_approx_world_coordinates_with_axes.ply"
    write_binary_ply(depth_path, points_depth, colors_rgb)
    write_binary_ply(world_path, points_world, colors_rgb)
    write_binary_ply(
        world_axes_path,
        np.vstack([points_world, axis_points]),
        np.vstack([colors_rgb, axis_colors]),
    )
    write_axes_line_ply(output_dir / "world_axes_lines.ply", axis_length_mm)
    render_world_preview(
        points_world,
        colors_rgb,
        output_dir / "reference_approx_world_point_cloud_preview.png",
        axis_length_mm=axis_length_mm,
    )
    table_like = np.abs(points_world[:, 2]) <= 60.0
    return {
        "point_count": int(len(points_world)),
        "world_bounds_min_xyz_mm": np.min(points_world, axis=0),
        "world_bounds_max_xyz_mm": np.max(points_world, axis=0),
        "table_candidate_count_within_60mm": int(np.sum(table_like)),
        "table_world_z_median_mm": (
            float(np.median(points_world[table_like, 2])) if np.any(table_like) else None
        ),
        "table_world_z_p95_absolute_mm": (
            float(np.percentile(np.abs(points_world[table_like, 2]), 95))
            if np.any(table_like)
            else None
        ),
        "files": [
            depth_path.name,
            world_path.name,
            world_axes_path.name,
            "world_axes_lines.ply",
            "reference_approx_world_point_cloud_preview.png",
        ],
        "accuracy_warning": (
            "Coordinates use depth recovered from a lossy 8-bit color preview; "
            "use this cloud for visual review only."
        ),
    }


def estimate_fixed_camera_stability(
    reference: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray_reference)
    mask[:430, :] = 255
    mask[:, :250] = 255
    mask[:, 1650:] = 255
    orb = cv2.ORB_create(8000, fastThreshold=10)
    keypoints_reference, descriptors_reference = orb.detectAndCompute(gray_reference, mask)
    keypoints_target, descriptors_target = orb.detectAndCompute(gray_target, mask)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        descriptors_reference, descriptors_target, k=2
    )
    good = [first for first, second in matches if first.distance < 0.7 * second.distance]
    points_reference = np.float32(
        [keypoints_reference[item.queryIdx].pt for item in good]
    )
    points_target = np.float32([keypoints_target[item.trainIdx].pt for item in good])
    homography, inliers = cv2.findHomography(
        points_reference, points_target, cv2.RANSAC, 1.5
    )
    if homography is None or inliers is None:
        raise RuntimeError("Unable to estimate fixed-camera validation homography")
    height, width = gray_reference.shape
    probes = np.asarray(
        [[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1], [width / 2, height / 2]]],
        dtype=np.float32,
    )
    moved = cv2.perspectiveTransform(probes, homography)[0]
    displacement = np.linalg.norm(moved - probes[0], axis=1)
    return {
        "orb_good_matches": len(good),
        "ransac_inliers": int(np.sum(inliers)),
        "homography_reference_to_target": homography,
        "probe_displacement_px": displacement,
        "max_probe_displacement_px": float(np.max(displacement)),
        "center_displacement_px": float(displacement[-1]),
    }


def camera_consistency(dataset_root: Path) -> dict[str, Any]:
    records = []
    fingerprints = []
    for recording in sorted(dataset_root.glob("camera_glove_recording_*")):
        path = recording / "rgbd_unpack" / "camera_1_intrinsics.json"
        intrinsics = read_json(path)
        parameters = camera_parameters(intrinsics)
        fingerprint = np.concatenate(
            [
                parameters["color_intrinsics"],
                parameters["depth_intrinsics"],
                np.asarray(
                    intrinsics["camera_param"]["depth_to_color_extrinsic"][
                        "translation_mm"
                    ],
                    dtype=np.float64,
                ),
            ]
        )
        fingerprints.append(fingerprint)
        records.append(
            {
                "recording": recording.name,
                "device_serial_number": intrinsics["device"]["serial_number"],
                "intrinsics_path": str(path.resolve()),
            }
        )
    reference = fingerprints[0]
    passed = all(np.allclose(value, reference, atol=0.0) for value in fingerprints[1:])
    return {"passed": passed, "recordings": records}


def run(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    config_path = args.config.resolve()
    prior_path = args.prior_camera_calibration.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(config_path)
    reference_recording = dataset_root / config["reference_recording"]
    reference_unpack = reference_recording / "rgbd_unpack"
    intrinsics_path = reference_unpack / "camera_1_intrinsics.json"
    intrinsics_json = read_json(intrinsics_path)
    parameters = camera_parameters(intrinsics_json)
    prior = read_json(prior_path)

    reference_rgb, rgb_frame_count = read_video_frame(
        reference_unpack / "RGB.mp4", int(config["reference_rgb_frame_index"])
    )
    reference_depth, depth_frame_count = read_video_frame(
        reference_unpack / "Depth.mp4", int(config["reference_depth_frame_index"])
    )
    color_from_world, pnp_quality = solve_color_pose(config, parameters)
    world_from_color = np.linalg.inv(color_from_world)
    color_from_depth, extrinsic_quality = load_same_camera_depth_to_color(
        prior_path, prior, intrinsics_json
    )
    depth_from_color = np.linalg.inv(color_from_depth)
    depth_from_world = depth_from_color @ color_from_world
    world_from_depth = np.linalg.inv(depth_from_world)

    table_normal = world_from_depth[2, :3].copy()
    table_d = float(world_from_depth[2, 3])
    scale = float(np.linalg.norm(table_normal))
    table_normal /= scale
    table_d /= scale

    reference_overlay, axis_pixels = draw_axes_overlay(
        reference_rgb,
        color_from_world,
        parameters,
        axis_length_mm=args.axis_length_mm,
        marker_height_mm=float(config["marker_center_height_above_table_mm"]),
        observations=config["rgb_marker_centers_px"],
        show_height_guide=True,
    )
    cv2.imwrite(str(output_dir / "reference_rgb.png"), reference_rgb)
    cv2.imwrite(str(output_dir / "reference_depth_preview_original.png"), reference_depth)
    cv2.imwrite(
        str(output_dir / "reference_rgb_markers_and_world_axes.png"), reference_overlay
    )
    # Keep the origin-height explanation and axis labels inside the review crop.
    crop = reference_overlay[675:1030, 430:1500]
    cv2.imwrite(
        str(output_dir / "reference_rgb_ruler_axes_crop.png"),
        cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC),
    )

    jet_indices, palette_error = nearest_jet_indices(reference_depth)
    expected_table_z, green_table, _ = expected_table_geometry(
        reference_depth.shape[:2],
        world_from_depth,
        depth_from_color,
        parameters,
        reference_rgb,
    )
    slope, intercept, fit_mask, depth_fit = fit_preview_depth_mapping(
        jet_indices, palette_error, expected_table_z, green_table
    )
    approximate_depth, valid_depth = decode_approximate_depth(
        jet_indices, palette_error, slope, intercept
    )
    make_depth_diagnostics(
        reference_depth, approximate_depth, fit_mask, output_dir
    )

    point_cloud = build_approximate_point_cloud(
        approximate_depth,
        valid_depth,
        reference_rgb,
        parameters,
        color_from_depth,
        world_from_depth,
        output_dir,
        axis_length_mm=args.axis_length_mm,
        max_points=args.max_points,
    )

    validations = []
    for recording in sorted(dataset_root.glob("camera_glove_recording_*")):
        if recording.name == config["reference_recording"]:
            continue
        video = recording / "rgbd_unpack" / "RGB.mp4"
        target, target_count = read_video_frame(
            video, int(config["validation_rgb_frame_index"])
        )
        overlay, _ = draw_axes_overlay(
            target,
            color_from_world,
            parameters,
            axis_length_mm=args.axis_length_mm,
            marker_height_mm=float(config["marker_center_height_above_table_mm"]),
            observations=None,
            show_height_guide=False,
        )
        output_name = f"validation_{recording.name}_world_axes.png"
        cv2.imwrite(str(output_dir / output_name), overlay)
        stability = estimate_fixed_camera_stability(reference_rgb, target)
        validations.append(
            {
                "recording": recording.name,
                "rgb_video": str(video.resolve()),
                "frame_index": int(config["validation_rgb_frame_index"]),
                "frame_count": target_count,
                "diagnostic_image": output_name,
                "fixed_camera_visual_stability": stability,
            }
        )

    consistency = camera_consistency(dataset_root)
    if not consistency["passed"]:
        raise RuntimeError("Camera core parameters differ across recordings")

    marker_height = float(config["marker_center_height_above_table_mm"])
    marker_origin_rgb = project_world_points(
        np.asarray([[0.0, 0.0, marker_height]]), color_from_world, parameters
    )[0]
    table_origin_rgb = project_world_points(
        np.asarray([[0.0, 0.0, 0.0]]), color_from_world, parameters
    )[0]
    result = {
        "schema": "movementcap.camera_to_mocap_world.v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "manual_visual_calibration_complete",
        "camera_type": "third_view",
        "method": "multi_frame_visual_CS400_RGB_PnP_tabletop_origin",
        "applies_to": [
            path.name for path in sorted(dataset_root.glob("camera_glove_recording_*"))
        ],
        "fixed_camera_assumption": True,
        "coordinate_system": {
            "origin": "CS-400 black vertex-hole axis projected vertically onto the tabletop",
            "origin_height_mode": "tabletop",
            "tabletop_world_z_mm": 0.0,
            "marker_plane_world_z_mm": marker_height,
            "x_axis": "from origin outward along the CS-400 long arm",
            "y_axis": "from origin outward along the CS-400 short arm",
            "z_axis": "tabletop normal, upward",
            "handedness": "right_handed",
            "units": "millimetres",
        },
        "input": {
            "dataset_root": str(dataset_root),
            "reference_recording": config["reference_recording"],
            "reference_rgb_mp4": str((reference_unpack / "RGB.mp4").resolve()),
            "reference_depth_preview_mp4": str(
                (reference_unpack / "Depth.mp4").resolve()
            ),
            "reference_rgb_frame_index": int(config["reference_rgb_frame_index"]),
            "reference_depth_frame_index": int(config["reference_depth_frame_index"]),
            "rgb_frame_count": rgb_frame_count,
            "depth_preview_frame_count": depth_frame_count,
            "intrinsics_json": str(intrinsics_path.resolve()),
            "visual_observation_config": str(config_path),
            "metric_pose_uses_depth_preview": False,
        },
        "camera_consistency": consistency,
        "manual_visual_observations": {
            "marker_model": config["marker_model"],
            "marker_center_height_above_table_mm": marker_height,
            "marker_centers_world_mm": config["marker_centers_world_mm"],
            "rgb_marker_centers_px": config["rgb_marker_centers_px"],
            "multi_frame_detection": config["multi_frame_visual_detection"],
            "model_geometry_provenance": config["model_geometry_provenance"],
        },
        "pose_quality": pnp_quality,
        "depth_to_color_extrinsic_provenance": extrinsic_quality,
        "table_plane_in_depth_camera": {
            "equation": "normal dot point_mm + d_mm = 0",
            "normal_up": table_normal,
            "d_mm": table_d,
            "source": "CS-400 RGB PnP tabletop plane; not fitted from lossy Depth.mp4",
        },
        "world_origin_in_color_camera_mm": color_from_world[:3, 3],
        "world_origin_in_depth_camera_mm": depth_from_world[:3, 3],
        "marker_plane_origin_in_color_camera_mm": transform_points(
            np.asarray([[0.0, 0.0, marker_height]]), color_from_world
        )[0],
        "marker_plane_origin_in_depth_camera_mm": transform_points(
            np.asarray([[0.0, 0.0, marker_height]]), depth_from_world
        )[0],
        "transforms": {
            "convention": "column point p_target = T_target_from_source @ [p_source; 1]",
            "depth_camera_to_world": world_from_depth,
            "world_to_depth_camera": depth_from_world,
            "color_camera_to_world": world_from_color,
            "world_to_color_camera": color_from_world,
            "depth_camera_to_color_camera": color_from_depth,
            "color_camera_to_depth_camera": depth_from_color,
        },
        "camera_origins_in_world_mm": {
            "depth_camera": world_from_depth[:3, 3],
            "color_camera": world_from_color[:3, 3],
        },
        "approximate_depth_preview_decode": depth_fit,
        "approximate_point_cloud": point_cloud,
        "diagnostics": {
            "axis_rgb_pixels": axis_pixels,
            "black_hole_marker_plane_origin_rgb_pixel": marker_origin_rgb,
            "tabletop_origin_rgb_pixel": table_origin_rgb,
            "black_hole_to_tabletop_pixel_displacement": table_origin_rgb
            - marker_origin_rgb,
            "fixed_camera_validation": validations,
            "files": [
                "reference_rgb.png",
                "reference_depth_preview_original.png",
                "reference_rgb_markers_and_world_axes.png",
                "reference_rgb_ruler_axes_crop.png",
                "reference_depth_table_fit_mask.png",
                "reference_approx_depth_mm.png",
                "reference_approx_depth_preview.png",
                *[item["diagnostic_image"] for item in validations],
                *point_cloud["files"],
            ],
        },
        "review_notes": [
            "World Z=0 is the tabletop, not the ruler or reflective-marker height.",
            "The black vertex hole represents the XY origin axis at Z=+45 mm; its vertical projection is the tabletop origin.",
            "The metric camera/world transform is solved from RGB marker geometry and does not use the lossy depth preview.",
            "The approximate point cloud is for visual review only because Depth.mp4 has about 20 mm color-index quantization.",
        ],
    }
    result_path = output_dir / "camera_to_world.json"
    result_path.write_text(
        json.dumps(json_ready(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    point_cloud_metadata = {
        "schema": "movementcap.approximate_world_point_cloud.v1",
        "calibration": str(result_path),
        "source_depth": str((reference_unpack / "Depth.mp4").resolve()),
        "source_rgb": str((reference_unpack / "RGB.mp4").resolve()),
        "frame_index": int(config["reference_depth_frame_index"]),
        "depth_decode": depth_fit,
        "point_cloud": point_cloud,
        "coordinate_system": result["coordinate_system"],
        "depth_camera_to_world_matrix_4x4_mm": world_from_depth,
    }
    (output_dir / "point_cloud_metadata.json").write_text(
        json.dumps(json_ready(point_cloud_metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = f"""# 2026-08-31 CS-400 camera/world calibration

Status: complete (multi-frame visual marker labeling + metric CS-400 RGB PnP).

- Reference: `{config['reference_recording']}` frame {config['reference_rgb_frame_index']}
- World origin: black vertex-hole axis projected onto the tabletop
- Tabletop world Z: 0 mm
- Marker/black-hole plane world Z: +{marker_height:.0f} mm
- RGB marker reprojection RMS: {pnp_quality['reprojection_rms_px']:.3f} px
- Fixed-camera validation recordings: {len(validations)}
- Maximum static-scene probe displacement: {max(item['fixed_camera_visual_stability']['max_probe_displacement_px'] for item in validations):.3f} px
- Approximate preview-depth quantization: {depth_fit['millimetres_per_color_index']:.3f} mm/index
- Approximate point count: {point_cloud['point_count']}

Primary result: `camera_to_world.json`.

The transform is metric and does not depend on Depth.mp4.  PLY files are visual-review artifacts only because the delivered depth video is an 8-bit lossy color preview rather than original 16-bit depth.
"""
    (output_dir / "CALIBRATION_SUMMARY.md").write_text(summary, encoding="utf-8")
    print(
        json.dumps(
            json_ready(
                {
                    "result": str(result_path),
                    "origin_mode": "tabletop",
                    "tabletop_world_z_mm": 0.0,
                    "marker_plane_world_z_mm": marker_height,
                    "pnp_rms_px": pnp_quality["reprojection_rms_px"],
                    "fixed_camera_max_probe_displacement_px": max(
                        item["fixed_camera_visual_stability"][
                            "max_probe_displacement_px"
                        ]
                        for item in validations
                    ),
                    "approx_depth_mm_per_index": depth_fit[
                        "millimetres_per_color_index"
                    ],
                    "approx_point_count": point_cloud["point_count"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
