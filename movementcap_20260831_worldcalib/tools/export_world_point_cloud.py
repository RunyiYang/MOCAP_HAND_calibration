#!/usr/bin/env python3
"""Export one RGB-D BAG frame as a colored point cloud in the mocap world frame.

The output follows the earlier world-calibration preview convention: millimetre
coordinates, a camera/depth-frame PLY, a world-frame PLY, and a world-frame PLY
with dense colored XYZ axis samples.  No Open3D dependency is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rosbags.rosbag1 import Reader

from calibrate_manual_cs400 import pixel_rays, project_camera_points
from export_rosbag_rgbd import (
    COLOR_PROFILE_TOPIC,
    DEPTH_PROFILE_TOPIC,
    DEPTH_TOPIC,
    RGB_TOPIC,
    decode_depth,
    decode_rgb,
    parse_orbbec_image,
    parse_video_profile,
)


AXIS_COLORS_RGB = np.asarray(
    [[255, 45, 45], [45, 230, 70], [60, 130, 255]], dtype=np.uint8
)


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def load_indexed_rgbd_frame(bag_path: Path, frame_index: int) -> dict[str, Any]:
    if frame_index < 0:
        raise ValueError("frame index must be non-negative")
    profiles: dict[str, Any] = {}
    selected: dict[str, Any] = {}
    counts = {"rgb": 0, "depth": 0}
    with Reader(bag_path) as reader:
        wanted = [
            connection
            for connection in reader.connections
            if connection.topic
            in {RGB_TOPIC, DEPTH_TOPIC, COLOR_PROFILE_TOPIC, DEPTH_PROFILE_TOPIC}
        ]
        for connection, bag_timestamp_ns, payload in reader.messages(connections=wanted):
            if connection.topic in {COLOR_PROFILE_TOPIC, DEPTH_PROFILE_TOPIC}:
                name = "color" if connection.topic == COLOR_PROFILE_TOPIC else "depth"
                profiles[name] = parse_video_profile(payload)
                continue
            stream = "rgb" if connection.topic == RGB_TOPIC else "depth"
            index = counts[stream]
            counts[stream] += 1
            if index != frame_index:
                continue
            message = parse_orbbec_image(payload)
            selected[stream] = {
                "index": index,
                "bag_timestamp_ns": int(bag_timestamp_ns),
                "device_timestamp_us": int(message.timestamp_usec),
                "frame_number": int(message.frame_number),
                "image": decode_rgb(message) if stream == "rgb" else decode_depth(message),
            }
            if set(selected) == {"rgb", "depth"} and set(profiles) == {"color", "depth"}:
                break
    if set(profiles) != {"color", "depth"}:
        raise RuntimeError(f"Missing BAG stream profiles; found {sorted(profiles)}")
    if set(selected) != {"rgb", "depth"}:
        raise IndexError(
            f"Frame {frame_index} not available in both streams; selected={sorted(selected)}, "
            f"frames_seen={counts}"
        )
    return {"profiles": profiles, **selected}


def colored_depth_cloud(
    rgb_bgr: np.ndarray,
    depth_mm: np.ndarray,
    depth_profile: Any,
    color_profile: Any,
    *,
    stride: int,
    min_depth_mm: float,
    max_depth_mm: float,
    max_points: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stride < 1:
        raise ValueError("stride must be at least 1")
    yy, xx = np.mgrid[0 : depth_mm.shape[0] : stride, 0 : depth_mm.shape[1] : stride]
    values = depth_mm[yy, xx].astype(np.float64)
    valid = (values >= min_depth_mm) & (values <= max_depth_mm)
    u = xx[valid].astype(np.float64)
    v = yy[valid].astype(np.float64)
    z = values[valid]
    rays = pixel_rays(u, v, depth_profile.intrinsics, depth_profile.distortion)
    points_depth = rays * z[:, None]

    rotation = np.asarray(depth_profile.rotation_matrix, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(depth_profile.translation_mm, dtype=np.float64)
    points_color = points_depth @ rotation.T + translation
    color_uv = project_camera_points(
        points_color,
        np.asarray(color_profile.intrinsics, dtype=np.float64),
        np.asarray(color_profile.distortion, dtype=np.float64),
    )
    rounded = np.rint(color_uv).astype(np.int64)
    inside = (
        (points_color[:, 2] > 0.0)
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < rgb_bgr.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < rgb_bgr.shape[0])
    )
    points_depth = points_depth[inside]
    rounded = rounded[inside]
    colors_rgb = rgb_bgr[rounded[:, 1], rounded[:, 0], ::-1].copy()
    source_depth_pixels = np.column_stack([u[inside], v[inside]])

    if max_points is not None and len(points_depth) > max_points:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(points_depth), size=max_points, replace=False))
        points_depth = points_depth[keep]
        colors_rgb = colors_rgb[keep]
        source_depth_pixels = source_depth_pixels[keep]
    return points_depth, colors_rgb, source_depth_pixels


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def write_binary_ply(path: Path, points: np.ndarray, colors_rgb: np.ndarray) -> None:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.clip(np.asarray(colors_rgb), 0, 255).astype(np.uint8).reshape(-1, 3)
    if len(values) != len(colors):
        raise ValueError("point and color counts differ")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment coordinates are millimetres\n"
        f"element vertex {len(values)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    records = np.empty(
        len(values),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")],
    )
    records["x"], records["y"], records["z"] = values.T
    records["r"], records["g"], records["b"] = colors.T
    with path.open("wb") as handle:
        handle.write(header)
        records.tofile(handle)


def axis_point_samples(axis_length_mm: float, spacing_mm: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    steps = max(2, int(np.ceil(axis_length_mm / spacing_mm)) + 1)
    distances = np.linspace(0.0, axis_length_mm, steps)
    all_points = []
    all_colors = []
    # A small cross section makes the axes remain visible in point-cloud viewers.
    offsets = np.asarray(
        [[0, 0, 0], [2, 0, 0], [-2, 0, 0], [0, 2, 0], [0, -2, 0], [0, 0, 2], [0, 0, -2]],
        dtype=np.float64,
    )
    for axis_index in range(3):
        line = np.zeros((steps, 3), dtype=np.float64)
        line[:, axis_index] = distances
        thick = (line[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        all_points.append(thick)
        all_colors.append(np.repeat(AXIS_COLORS_RGB[axis_index][None, :], len(thick), axis=0))
    # Add a compact white origin marker.
    grid = np.arange(-5.0, 5.1, 2.5)
    origin = np.asarray(np.meshgrid(grid, grid, grid, indexing="ij")).reshape(3, -1).T
    origin = origin[np.linalg.norm(origin, axis=1) <= 6.0]
    all_points.append(origin)
    all_colors.append(np.full((len(origin), 3), 255, dtype=np.uint8))
    return np.vstack(all_points), np.vstack(all_colors)


def write_axes_line_ply(path: Path, axis_length_mm: float) -> None:
    points = [[0, 0, 0], [axis_length_mm, 0, 0], [0, axis_length_mm, 0], [0, 0, axis_length_mm]]
    edges = [(0, 1, *AXIS_COLORS_RGB[0]), (0, 2, *AXIS_COLORS_RGB[1]), (0, 3, *AXIS_COLORS_RGB[2])]
    lines = [
        "ply",
        "format ascii 1.0",
        "comment coordinates are millimetres",
        "element vertex 4",
        "property float x",
        "property float y",
        "property float z",
        "element edge 3",
        "property int vertex1",
        "property int vertex2",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
        *[" ".join(str(value) for value in point) for point in points],
        *[" ".join(str(int(value)) for value in edge) for edge in edges],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def render_world_preview(
    points_world: np.ndarray,
    colors_rgb: np.ndarray,
    output_path: Path,
    *,
    axis_length_mm: float,
    width: int = 1200,
    height: int = 900,
) -> None:
    points = np.asarray(points_world, dtype=np.float64)
    colors = np.asarray(colors_rgb, dtype=np.uint8)
    if len(points) > 450_000:
        keep = np.linspace(0, len(points) - 1, 450_000).astype(np.int64)
        points, colors = points[keep], colors[keep]

    # Virtual camera: above the +X/+Y quadrant looking back toward the origin.
    view_direction = np.asarray([1.35, 1.7, 1.05], dtype=np.float64)
    view_direction /= np.linalg.norm(view_direction)
    right = np.cross(np.asarray([0.0, 0.0, 1.0]), view_direction)
    right /= np.linalg.norm(right)
    up = np.cross(view_direction, right)
    up /= np.linalg.norm(up)
    basis = np.vstack([right, up, view_direction])
    view = points @ basis.T

    low = np.percentile(view[:, :2], 0.5, axis=0)
    high = np.percentile(view[:, :2], 99.5, axis=0)
    span = np.maximum(high - low, 1.0)
    margin = 55
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])
    center_view = 0.5 * (low + high)
    u = np.rint((view[:, 0] - center_view[0]) * scale + width / 2).astype(np.int64)
    v = np.rint(height / 2 - (view[:, 1] - center_view[1]) * scale).astype(np.int64)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, depth, colors = u[inside], v[inside], view[inside, 2], colors[inside]
    # Draw far points first; nearby samples overwrite them.
    order = np.argsort(depth)
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    canvas[v[order], u[order]] = colors[order, ::-1]
    occupied = np.zeros((height, width), dtype=np.uint8)
    occupied[v, u] = 255
    expanded = cv2.dilate(canvas, np.ones((2, 2), np.uint8), iterations=1)
    expanded_mask = cv2.dilate(occupied, np.ones((2, 2), np.uint8), iterations=1)
    canvas[expanded_mask > 0] = expanded[expanded_mask > 0]

    def project(values: np.ndarray) -> np.ndarray:
        q = np.asarray(values, dtype=np.float64) @ basis.T
        px = np.rint((q[:, 0] - center_view[0]) * scale + width / 2).astype(int)
        py = np.rint(height / 2 - (q[:, 1] - center_view[1]) * scale).astype(int)
        return np.column_stack([px, py])

    axes = np.asarray(
        [[0, 0, 0], [axis_length_mm, 0, 0], [0, axis_length_mm, 0], [0, 0, axis_length_mm]],
        dtype=np.float64,
    )
    pixels = project(axes)
    origin = tuple(pixels[0])
    for index, label in enumerate(("X", "Y", "Z"), start=1):
        endpoint = tuple(pixels[index])
        color = tuple(int(value) for value in AXIS_COLORS_RGB[index - 1][::-1])
        cv2.arrowedLine(canvas, origin, endpoint, color, 5, cv2.LINE_AA, tipLength=0.10)
        cv2.putText(canvas, f"+{label}", endpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.circle(canvas, origin, 7, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(canvas, "world origin", (origin[0] + 10, origin[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "world-frame colored point cloud (units: mm)", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), canvas):
        raise IOError(f"Failed to write {output_path}")


def depth_preview(depth_mm: np.ndarray) -> np.ndarray:
    valid = depth_mm > 0
    normalized = np.zeros(depth_mm.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth_mm[valid], [2, 98])
        high = max(high, low + 1.0)
        normalized[valid] = np.clip((depth_mm[valid] - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
    preview = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    return preview


def draw_video_world_axes(
    rgb_bgr: np.ndarray,
    depth_to_world: np.ndarray,
    depth_profile: Any,
    color_profile: Any,
    axis_length_mm: float,
) -> np.ndarray:
    """Project the tabletop world axes into the selected RGB video frame."""
    points_world = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [axis_length_mm, 0.0, 0.0],
            [0.0, axis_length_mm, 0.0],
            [0.0, 0.0, axis_length_mm],
        ],
        dtype=np.float64,
    )
    world_to_depth = np.linalg.inv(depth_to_world)
    points_depth = transform_points(points_world, world_to_depth)
    depth_to_color_rotation = np.asarray(
        depth_profile.rotation_matrix, dtype=np.float64
    ).reshape(3, 3)
    depth_to_color_translation = np.asarray(
        depth_profile.translation_mm, dtype=np.float64
    )
    points_color = points_depth @ depth_to_color_rotation.T + depth_to_color_translation
    pixels = np.rint(
        project_camera_points(
            points_color,
            np.asarray(color_profile.intrinsics, dtype=np.float64),
            np.asarray(color_profile.distortion, dtype=np.float64),
        )
    ).astype(int)
    output = rgb_bgr.copy()
    origin = tuple(pixels[0])
    for index, label in enumerate(("X", "Y", "Z"), start=1):
        endpoint = tuple(pixels[index])
        color_bgr = tuple(int(value) for value in AXIS_COLORS_RGB[index - 1][::-1])
        cv2.arrowedLine(output, origin, endpoint, color_bgr, 6, cv2.LINE_AA, tipLength=0.10)
        cv2.putText(
            output,
            f"+{label}",
            (endpoint[0] + 8, endpoint[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color_bgr,
            3,
            cv2.LINE_AA,
        )
    cv2.circle(output, origin, 9, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(
        output,
        "world O",
        (origin[0] + 12, origin[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def calibration_depth_to_world(calibration: dict[str, Any]) -> np.ndarray:
    transforms = calibration.get("transforms", {})
    if "depth_camera_to_world" in transforms:
        return np.asarray(transforms["depth_camera_to_world"], dtype=np.float64)
    transform = calibration.get("transform", {})
    if "raw_depth_optical_to_world_recorded_affine_4x4" in transform:
        matrix_m = np.asarray(
            transform["raw_depth_optical_to_world_recorded_affine_4x4"], dtype=np.float64
        )
        matrix_mm = matrix_m.copy()
        matrix_mm[:3, 3] *= 1000.0
        return matrix_mm
    raise ValueError("Calibration JSON has no supported depth-camera-to-world transform")


def run(args: argparse.Namespace) -> None:
    calibration_path = args.calibration.resolve()
    bag_path = args.bag.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    frame = load_indexed_rgbd_frame(bag_path, args.frame_index)
    rgb = frame["rgb"]["image"]
    depth = frame["depth"]["image"]
    points_depth, colors, source_depth_pixels = colored_depth_cloud(
        rgb,
        depth,
        frame["profiles"]["depth"],
        frame["profiles"]["color"],
        stride=args.stride,
        min_depth_mm=args.min_depth_mm,
        max_depth_mm=args.max_depth_mm,
        max_points=args.max_points,
        seed=args.seed,
    )
    depth_to_world = calibration_depth_to_world(calibration)
    points_world = transform_points(points_depth, depth_to_world)
    axis_points, axis_colors = axis_point_samples(args.axis_length_mm)

    camera_path = output_dir / "selected_frame_depth_camera_coordinates.ply"
    world_path = output_dir / "selected_frame_world_coordinates.ply"
    world_axes_path = output_dir / "selected_frame_world_coordinates_with_axes.ply"
    axes_line_path = output_dir / "world_axes_lines.ply"
    write_binary_ply(camera_path, points_depth, colors)
    write_binary_ply(world_path, points_world, colors)
    write_binary_ply(
        world_axes_path,
        np.vstack([points_world, axis_points]),
        np.vstack([colors, axis_colors]),
    )
    write_axes_line_ply(axes_line_path, args.axis_length_mm)

    cv2.imwrite(str(output_dir / "selected_rgb.png"), rgb)
    cv2.imwrite(
        str(output_dir / "selected_rgb_world_axes_overlay.png"),
        draw_video_world_axes(
            rgb,
            depth_to_world,
            frame["profiles"]["depth"],
            frame["profiles"]["color"],
            args.axis_length_mm,
        ),
    )
    cv2.imwrite(str(output_dir / "selected_raw_depth_preview.png"), depth_preview(depth))
    render_world_preview(
        points_world,
        colors,
        output_dir / "world_point_cloud_preview.png",
        axis_length_mm=args.axis_length_mm,
    )

    table_normal = np.asarray(
        calibration.get("table_plane_in_depth_camera", {}).get("normal_up", [np.nan] * 3),
        dtype=np.float64,
    )
    table_d = float(calibration.get("table_plane_in_depth_camera", {}).get("d_mm", np.nan))
    table_residual = points_depth @ table_normal + table_d
    table_like = np.isfinite(table_residual) & (np.abs(table_residual) <= 6.0)
    coordinate_system = calibration.get(
        "coordinate_system", calibration.get("world_definition", {})
    ) or {}
    expected_tabletop_world_z_mm = float(
        coordinate_system.get("tabletop_world_z_mm", 0.0)
    )
    table_world_z = points_world[table_like, 2]
    table_world_z_error = table_world_z - expected_tabletop_world_z_mm
    rgb_time = frame["rgb"]["device_timestamp_us"]
    depth_time = frame["depth"]["device_timestamp_us"]
    metadata = {
        "schema": "movementcap.world_point_cloud.v1",
        "units": "millimetres",
        "input": {
            "bag": str(bag_path),
            "frame_index": args.frame_index,
            "rgb_device_timestamp_us": rgb_time,
            "depth_device_timestamp_us": depth_time,
            "depth_minus_rgb_timestamp_us": depth_time - rgb_time,
            "calibration": str(calibration_path),
            "calibration_schema": calibration.get("schema"),
        },
        "point_cloud": {
            "point_count": len(points_world),
            "stride": args.stride,
            "min_depth_mm": args.min_depth_mm,
            "max_depth_mm": args.max_depth_mm,
            "max_points": args.max_points,
            "world_bounds_min_xyz_mm": np.min(points_world, axis=0),
            "world_bounds_max_xyz_mm": np.max(points_world, axis=0),
            "source_depth_pixel_bounds_uv": [
                np.min(source_depth_pixels, axis=0),
                np.max(source_depth_pixels, axis=0),
            ],
        },
        "coordinate_system": coordinate_system,
        "depth_camera_to_world_matrix_4x4_mm": depth_to_world,
        "table_plane_check": {
            "candidate_count_within_6mm": int(np.sum(table_like)),
            "expected_world_z_mm": expected_tabletop_world_z_mm,
            "world_z_median_mm": float(np.median(table_world_z)) if np.any(table_like) else None,
            "world_z_median_error_mm": (
                float(np.median(table_world_z_error)) if np.any(table_like) else None
            ),
            "world_z_p95_absolute_error_mm": (
                float(np.percentile(np.abs(table_world_z_error), 95))
                if np.any(table_like)
                else None
            ),
        },
        "outputs": {
            "depth_camera_coordinates": camera_path.name,
            "world_coordinates": world_path.name,
            "world_coordinates_with_axes": world_axes_path.name,
            "world_axes_lines": axes_line_path.name,
            "rgb": "selected_rgb.png",
            "rgb_world_axes_overlay": "selected_rgb_world_axes_overlay.png",
            "depth_preview": "selected_raw_depth_preview.png",
            "world_preview": "world_point_cloud_preview.png",
        },
    }
    (output_dir / "point_cloud_metadata.json").write_text(
        json.dumps(json_ready(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "points": len(points_world),
                "rgb_depth_delta_us": depth_time - rgb_time,
                "table_plane_check": metadata["table_plane_check"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-depth-mm", type=float, default=250.0)
    parser.add_argument("--max-depth-mm", type=float, default=3500.0)
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--axis-length-mm", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
