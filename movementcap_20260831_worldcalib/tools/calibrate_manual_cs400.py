#!/usr/bin/env python3
"""One-off CS-400 camera/world calibration for the 2026-08-29 movementcap set.

The Orbbec bag is a ROS1 bag whose Image message definition differs from the
standard sensor_msgs/Image definition.  The raw bytes are decoded here rather
than routed through OrbbecSDK playback, which rejects this recorder version.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import io
import json
import math
from pathlib import Path
import struct
import warnings
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw
from rosbags.highlevel import AnyReader


DEPTH_TOPIC = "/cam/sensor_3/frameType_3"
COLOR_TOPIC = "/cam/sensor_2/frameType_2"
DEPTH_PROFILE_TOPIC = "/cam/streamProfileType_3"
COLOR_PROFILE_TOPIC = "/cam/streamProfileType_2"


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    default_dataset = repo / "movementcap--同步整理_20260829_三段"
    default_config = Path(__file__).resolve().parents[1] / "configs" / "manual_cs400_points.json"
    parser = argparse.ArgumentParser(
        description="Calibrate the fixed Femto Bolt to the CS-400/mocap world frame."
    )
    parser.add_argument("--dataset-root", type=Path, default=default_dataset)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to DATASET_ROOT/world_calibration_cs400_manual_20260830.",
    )
    parser.add_argument(
        "--skip-take-validation",
        action="store_true",
        help="Only calibrate from the 3-second reference bag.",
    )
    parser.add_argument(
        "--origin-height-mode",
        choices=("tabletop", "marker-plane"),
        default="tabletop",
        help=(
            "World-origin height. 'tabletop' applies the standard CS-400 "
            "45 mm vertical offset; 'marker-plane' places the origin at the "
            "virtual arm intersection near the black vertex hole."
        ),
    )
    return parser.parse_args()


def _parse_header(raw: bytes | memoryview, offset: int = 0) -> tuple[dict[str, Any], int]:
    seq, sec, nsec = struct.unpack_from("<III", raw, offset)
    offset += 12
    name_length = struct.unpack_from("<I", raw, offset)[0]
    offset += 4
    frame_id = bytes(raw[offset : offset + name_length]).decode("utf-8")
    offset += name_length
    return {
        "seq": int(seq),
        "sec": int(sec),
        "nsec": int(nsec),
        "timestamp_us": int(sec) * 1_000_000 + int(round(int(nsec) / 1000.0)),
        "frame_id": frame_id,
    }, offset


def parse_orbbec_image(raw: bytes | memoryview) -> dict[str, Any]:
    header, offset = _parse_header(raw)
    height, width = struct.unpack_from("<II", raw, offset)
    offset += 8
    encoding_length = struct.unpack_from("<I", raw, offset)[0]
    offset += 4
    encoding = bytes(raw[offset : offset + encoding_length]).decode("ascii")
    offset += encoding_length
    is_bigendian = int(raw[offset])
    offset += 1
    step, metadata_size, data_size = struct.unpack_from("<III", raw, offset)
    offset += 12
    payload = memoryview(raw)[offset : offset + data_size]
    if len(payload) != data_size or metadata_size > data_size:
        raise ValueError("Invalid Orbbec Image payload lengths")
    pixels = payload[metadata_size:]
    return {
        "header": header,
        "height": int(height),
        "width": int(width),
        "encoding": encoding,
        "is_bigendian": is_bigendian,
        "step": int(step),
        "metadata": bytes(payload[:metadata_size]),
        "pixels": pixels,
    }


def parse_stream_profile(raw: bytes | memoryview) -> dict[str, Any]:
    header, offset = _parse_header(raw)
    stream_type, pixel_format = struct.unpack_from("<BB", raw, offset)
    offset += 2
    rotation = np.asarray(struct.unpack_from("<9f", raw, offset), dtype=np.float64).reshape(3, 3)
    offset += 36
    translation = np.asarray(struct.unpack_from("<3f", raw, offset), dtype=np.float64)
    offset += 12
    width, height, fps = struct.unpack_from("<3H", raw, offset)
    offset += 6
    fx, fy, cx, cy = struct.unpack_from("<4f", raw, offset)
    offset += 16
    # Orbbec order: k1,k2,k3,k4,k5,k6,p1,p2.
    distortion = np.asarray(struct.unpack_from("<8f", raw, offset), dtype=np.float64)
    offset += 32
    distortion_model = int(raw[offset])
    return {
        "header": header,
        "stream_type": int(stream_type),
        "pixel_format": int(pixel_format),
        "rotation": rotation,
        "translation_mm": translation,
        "width": int(width),
        "height": int(height),
        "fps": int(fps),
        "intrinsics": np.asarray([fx, fy, cx, cy], dtype=np.float64),
        "distortion": distortion,
        "distortion_model": distortion_model,
    }


def _connection(reader: AnyReader, topic: str) -> Any:
    matches = [item for item in reader.connections if item.topic == topic]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {topic!r} connection, found {len(matches)}")
    return matches[0]


def load_reference_bag(bag_path: Path) -> dict[str, Any]:
    depth_frames: list[np.ndarray] = []
    color_frames: list[Image.Image] = []
    with AnyReader([bag_path]) as reader:
        profiles: dict[str, dict[str, Any]] = {}
        for name, topic in (("depth", DEPTH_PROFILE_TOPIC), ("color", COLOR_PROFILE_TOPIC)):
            connection = _connection(reader, topic)
            raw = next(reader.messages(connections=[connection]))[2]
            profiles[name] = parse_stream_profile(raw)

        depth_connection = _connection(reader, DEPTH_TOPIC)
        for _, _, raw in reader.messages(connections=[depth_connection]):
            frame = parse_orbbec_image(raw)
            if frame["encoding"] != "mono16":
                raise RuntimeError(f"Unexpected depth encoding: {frame['encoding']}")
            expected = frame["height"] * frame["width"] * 2
            if len(frame["pixels"]) != expected:
                raise RuntimeError("Unexpected depth byte count")
            depth_frames.append(
                np.frombuffer(frame["pixels"], dtype="<u2")
                .reshape(frame["height"], frame["width"])
                .copy()
            )

        color_connection = _connection(reader, COLOR_TOPIC)
        for index, (_, _, raw) in enumerate(reader.messages(connections=[color_connection])):
            if index in {0, len(depth_frames) // 2, len(depth_frames) - 1}:
                frame = parse_orbbec_image(raw)
                if frame["encoding"] != "mjpg":
                    raise RuntimeError(f"Unexpected color encoding: {frame['encoding']}")
                color_frames.append(Image.open(io.BytesIO(bytes(frame["pixels"]))).convert("RGB"))

    stack = np.stack(depth_frames)
    work = stack.astype(np.float32)
    work[work == 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(work, axis=0)
    median = np.nan_to_num(median, nan=0.0).astype(np.uint16)
    valid_counts = np.count_nonzero(stack, axis=0)
    return {
        "profiles": profiles,
        "depth_stack": stack,
        "median_depth_mm": median,
        "valid_counts": valid_counts,
        "representative_rgb": color_frames[len(color_frames) // 2],
        "frame_count": int(stack.shape[0]),
    }


def undistort_normalized(
    x_distorted: np.ndarray | float,
    y_distorted: np.ndarray | float,
    coefficients: np.ndarray,
    iterations: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    xd = np.asarray(x_distorted, dtype=np.float64)
    yd = np.asarray(y_distorted, dtype=np.float64)
    x = xd.copy()
    y = yd.copy()
    k1, k2, k3, k4, k5, k6, p1, p2 = coefficients
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
    x: np.ndarray | float,
    y: np.ndarray | float,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    k1, k2, k3, k4, k5, k6, p1, p2 = coefficients
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (
        1.0 + k4 * r2 + k5 * r4 + k6 * r6
    )
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xd, yd


def pixel_rays(
    u: np.ndarray | float,
    v: np.ndarray | float,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
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
    fx, fy, cx, cy = intrinsics
    return np.stack([fx * xd + cx, fy * yd + cy], axis=-1)


def median_depth_points(
    median_depth: np.ndarray,
    profile: dict[str, Any],
    fit_config: dict[str, Any],
) -> np.ndarray:
    vv, uu = np.mgrid[: median_depth.shape[0], : median_depth.shape[1]]
    z = median_depth.astype(np.float64)
    mask = (
        (z > float(fit_config["depth_min_mm"]))
        & (z < float(fit_config["depth_max_mm"]))
        & (vv > int(fit_config["depth_image_v_min"]))
    )
    rays = pixel_rays(
        uu[mask], v=vv[mask], intrinsics=profile["intrinsics"], distortion=profile["distortion"]
    )
    return rays * z[mask, None]


def fit_plane_ransac(
    points: np.ndarray,
    threshold_mm: float,
    refine_threshold_mm: float,
    seed: int,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    sample = points[rng.choice(len(points), min(30_000, len(points)), replace=False)]
    best_count = -1
    best_normal: np.ndarray | None = None
    best_d = 0.0
    for _ in range(500):
        tri = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        d = -float(normal @ tri[0])
        count = int(np.count_nonzero(np.abs(sample @ normal + d) < threshold_mm))
        if count > best_count:
            best_count, best_normal, best_d = count, normal, d
    if best_normal is None:
        raise RuntimeError("RANSAC failed to find a table plane")

    first_residual = np.abs(points @ best_normal + best_d)
    inliers = points[first_residual < refine_threshold_mm]
    centroid = np.mean(inliers, axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vh[-1]
    # Upward points approximately back towards the camera, not into the table.
    if float(normal @ centroid) > 0.0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    d = -float(normal @ centroid)
    residual = np.abs(points @ normal + d)
    bounded = residual[residual < 30.0]
    stats = {
        "candidate_point_count": int(len(points)),
        "refined_inlier_count": int(len(inliers)),
        "inlier_ratio_below_5mm": float(np.mean(residual < 5.0)),
        "residual_mm_p50": float(np.percentile(bounded, 50)),
        "residual_mm_p90": float(np.percentile(bounded, 90)),
        "residual_mm_p95": float(np.percentile(bounded, 95)),
        "residual_mm_p99": float(np.percentile(bounded, 99)),
    }
    return normal, d, stats


def reconstruct_marker(
    pixel: Iterable[float],
    marker_height_mm: float,
    plane_normal: np.ndarray,
    plane_d: float,
    depth_profile: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    u, v = [float(item) for item in pixel]
    ray = pixel_rays(u, v, depth_profile["intrinsics"], depth_profile["distortion"])
    scale = (marker_height_mm - plane_d) / float(plane_normal @ ray)
    center = ray * scale
    foot = center - marker_height_mm * plane_normal
    return center, foot


def line_intersection(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    first_direction = p2 - p1
    first_direction /= np.linalg.norm(first_direction)
    second_direction = q2 - q1
    second_direction /= np.linalg.norm(second_direction)
    coefficients = np.linalg.lstsq(
        np.column_stack([first_direction, -second_direction]), q1 - p1, rcond=None
    )[0]
    first_closest = p1 + coefficients[0] * first_direction
    second_closest = q1 + coefficients[1] * second_direction
    origin = 0.5 * (first_closest + second_closest)
    gap = float(np.linalg.norm(first_closest - second_closest))
    return origin, first_direction, second_direction, gap


def homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def rotate_about_axis(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a vector with Rodrigues' formula."""
    unit_axis = np.asarray(axis, dtype=np.float64)
    unit_axis /= np.linalg.norm(unit_axis)
    value = np.asarray(vector, dtype=np.float64)
    return (
        value * math.cos(angle_rad)
        + np.cross(unit_axis, value) * math.sin(angle_rad)
        + unit_axis * float(unit_axis @ value) * (1.0 - math.cos(angle_rad))
    )


def project_depth_to_color(
    points_depth: np.ndarray,
    depth_to_color_rotation: np.ndarray,
    depth_to_color_translation: np.ndarray,
    color_profile: dict[str, Any],
) -> np.ndarray:
    points = np.asarray(points_depth, dtype=np.float64)
    points_color = points @ depth_to_color_rotation.T + depth_to_color_translation
    return project_camera_points(points_color, color_profile["intrinsics"], color_profile["distortion"])


def draw_circle_label(
    draw: ImageDraw.ImageDraw,
    xy: Iterable[float],
    label: str,
    color: tuple[int, int, int],
    radius: int = 8,
) -> None:
    x, y = [float(item) for item in xy]
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=4)
    draw.text((x + radius + 4, y - radius - 2), label, fill=color, stroke_width=2, stroke_fill=(0, 0, 0))


def draw_line(
    draw: ImageDraw.ImageDraw,
    a: Iterable[float],
    b: Iterable[float],
    color: tuple[int, int, int],
    width: int = 6,
) -> None:
    draw.line([tuple(float(item) for item in a), tuple(float(item) for item in b)], fill=color, width=width)


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    a: Iterable[float],
    b: Iterable[float],
    color: tuple[int, int, int],
    width: int = 4,
    dash_px: float = 16.0,
    gap_px: float = 10.0,
) -> None:
    start = np.asarray(list(a), dtype=np.float64)
    end = np.asarray(list(b), dtype=np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 0.0:
        return
    direction = delta / length
    distance = 0.0
    while distance < length:
        segment_end = min(distance + dash_px, length)
        p0 = start + direction * distance
        p1 = start + direction * segment_end
        draw.line([tuple(p0), tuple(p1)], fill=color, width=width)
        distance += dash_px + gap_px


def draw_marker_plane_guides(
    image: Image.Image,
    tabletop_origin_depth: np.ndarray,
    marker_height_mm: float,
    basis_depth_from_world: np.ndarray,
    depth_to_color_rotation: np.ndarray,
    depth_to_color_translation: np.ndarray,
    color_profile: dict[str, Any],
    origin_height_mode: str = "tabletop",
    axis_length_mm: float = 300.0,
) -> dict[str, list[float]]:
    """Compare the tabletop and marker-height arm-intersection planes."""
    z_axis = basis_depth_from_world[:, 2]
    marker_plane_origin = tabletop_origin_depth + marker_height_mm * z_axis
    points = np.vstack(
        [
            tabletop_origin_depth,
            marker_plane_origin,
            tabletop_origin_depth + axis_length_mm * basis_depth_from_world[:, 0],
            tabletop_origin_depth + axis_length_mm * basis_depth_from_world[:, 1],
            marker_plane_origin + axis_length_mm * basis_depth_from_world[:, 0],
            marker_plane_origin + axis_length_mm * basis_depth_from_world[:, 1],
        ]
    )
    pixels = project_depth_to_color(
        points,
        depth_to_color_rotation,
        depth_to_color_translation,
        color_profile,
    )
    draw = ImageDraw.Draw(image)
    if origin_height_mode == "tabletop":
        guide_origin = pixels[1]
        guide_x = pixels[4]
        guide_y = pixels[5]
        guide_label = f"marker/black-hole plane (+{marker_height_mm:g} mm)"
        caption = "solid: world axes on tabletop   dashed: marker/black-hole plane"
    else:
        guide_origin = pixels[0]
        guide_x = pixels[2]
        guide_y = pixels[3]
        guide_label = f"tabletop plane (-{marker_height_mm:g} mm)"
        caption = "solid: alternate world axes at black hole   dashed: tabletop plane"
    draw_dashed_line(draw, guide_origin, guide_x, (255, 150, 150), width=4)
    draw_dashed_line(draw, guide_origin, guide_y, (150, 255, 160), width=4)
    draw_dashed_line(draw, pixels[0], pixels[1], (255, 220, 35), width=3, dash_px=10, gap_px=7)
    if origin_height_mode == "marker-plane":
        guide_x_px, guide_y_px = [float(item) for item in guide_origin]
        draw.ellipse(
            (guide_x_px - 7, guide_y_px - 7, guide_x_px + 7, guide_y_px + 7),
            outline=(255, 255, 255),
            width=4,
        )
        draw.text(
            (guide_x_px - 255, guide_y_px + 14),
            guide_label,
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    else:
        draw_circle_label(draw, guide_origin, guide_label, (255, 255, 255), radius=7)
    draw.text(
        (30, 30),
        caption,
        fill=(255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    return {
        "ground_origin": pixels[0].tolist(),
        "marker_plane_origin": pixels[1].tolist(),
        "ground_plane_x_300mm": pixels[2].tolist(),
        "ground_plane_y_300mm": pixels[3].tolist(),
        "marker_plane_x_300mm": pixels[4].tolist(),
        "marker_plane_y_300mm": pixels[5].tolist(),
        "selected_origin_height_mode": origin_height_mode,
    }


def draw_world_axes_rgb(
    image: Image.Image,
    origin_depth: np.ndarray,
    basis_depth_from_world: np.ndarray,
    depth_to_color_rotation: np.ndarray,
    depth_to_color_translation: np.ndarray,
    color_profile: dict[str, Any],
    axis_length_mm: float = 300.0,
) -> tuple[Image.Image, dict[str, list[float]]]:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    points = np.vstack(
        [
            origin_depth,
            origin_depth + axis_length_mm * basis_depth_from_world[:, 0],
            origin_depth + axis_length_mm * basis_depth_from_world[:, 1],
            origin_depth + axis_length_mm * basis_depth_from_world[:, 2],
        ]
    )
    pixels = project_depth_to_color(
        points,
        depth_to_color_rotation,
        depth_to_color_translation,
        color_profile,
    )
    colors = {"x": (255, 45, 45), "y": (45, 230, 70), "z": (60, 130, 255)}
    for index, name in enumerate(("x", "y", "z"), start=1):
        draw_line(draw, pixels[0], pixels[index], colors[name])
        draw_circle_label(draw, pixels[index], f"+{name.upper()} 300mm", colors[name], radius=6)
    draw_circle_label(draw, pixels[0], "O", (255, 220, 35), radius=9)
    return output, {
        "origin": pixels[0].tolist(),
        "x_300mm": pixels[1].tolist(),
        "y_300mm": pixels[2].tolist(),
        "z_300mm": pixels[3].tolist(),
    }


def depth_color_image(depth: np.ndarray) -> Image.Image:
    value = np.clip((depth.astype(np.float64) - 500.0) / 2500.0, 0.0, 1.0)
    red = np.clip(1.5 - 2.0 * np.abs(value - 0.15), 0.0, 1.0)
    green = np.clip(1.4 - 2.5 * np.abs(value - 0.5), 0.0, 1.0)
    blue = np.clip(1.5 - 2.0 * np.abs(value - 0.85), 0.0, 1.0)
    rgb = (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)
    rgb[depth == 0] = 0
    return Image.fromarray(rgb)


def make_depth_diagnostic(
    median_depth: np.ndarray,
    marker_pixels: dict[str, list[float]],
    origin_depth: np.ndarray,
    basis_depth_from_world: np.ndarray,
    depth_profile: dict[str, Any],
) -> Image.Image:
    output = depth_color_image(median_depth)
    draw = ImageDraw.Draw(output)
    for name, pixel in marker_pixels.items():
        draw_circle_label(draw, pixel, name, (255, 255, 255), radius=5)
    axis_points = np.vstack(
        [
            origin_depth,
            origin_depth + 300.0 * basis_depth_from_world[:, 0],
            origin_depth + 300.0 * basis_depth_from_world[:, 1],
            origin_depth + 300.0 * basis_depth_from_world[:, 2],
        ]
    )
    pixels = project_camera_points(
        axis_points, depth_profile["intrinsics"], depth_profile["distortion"]
    )
    for index, (label, color) in enumerate(
        (("+X", (255, 45, 45)), ("+Y", (45, 230, 70)), ("+Z", (60, 130, 255))), start=1
    ):
        draw_line(draw, pixels[0], pixels[index], color, width=3)
        draw_circle_label(draw, pixels[index], label, color, radius=4)
    draw_circle_label(draw, pixels[0], "O", (255, 220, 35), radius=5)
    return output


def make_rgb_depth_overlay(
    rgb: Image.Image,
    median_depth: np.ndarray,
    depth_profile: dict[str, Any],
    color_profile: dict[str, Any],
) -> Image.Image:
    vv, uu = np.nonzero((median_depth > 0) & (median_depth < 4000))
    # Sparse splat keeps the RGB content legible.
    select = np.arange(len(uu)) % 2 == 0
    uu = uu[select]
    vv = vv[select]
    z = median_depth[vv, uu].astype(np.float64)
    rays = pixel_rays(uu, vv, depth_profile["intrinsics"], depth_profile["distortion"])
    points_depth = rays * z[:, None]
    rotation = depth_profile["rotation"]
    translation = depth_profile["translation_mm"]
    points_color = points_depth @ rotation.T + translation
    valid = points_color[:, 2] > 0
    points_color = points_color[valid]
    pixels = project_camera_points(
        points_color, color_profile["intrinsics"], color_profile["distortion"]
    )
    u = np.rint(pixels[:, 0]).astype(np.int64)
    v = np.rint(pixels[:, 1]).astype(np.int64)
    valid = (u >= 0) & (u < rgb.width) & (v >= 0) & (v < rgb.height)
    u, v = u[valid], v[valid]
    distance = points_color[valid, 2]
    value = np.clip((distance - 500.0) / 2500.0, 0.0, 1.0)
    colors = np.stack(
        [255.0 * (1.0 - value), 255.0 * (1.0 - np.abs(value - 0.5) * 2.0), 255.0 * value],
        axis=-1,
    ).clip(0, 255).astype(np.uint8)
    output = (np.asarray(rgb, dtype=np.float64) * 0.55).astype(np.uint8)
    output[v, u] = (0.2 * output[v, u] + 0.8 * colors).astype(np.uint8)
    return Image.fromarray(output)


def read_rgb_near_timestamp(bag_path: Path, target_timestamp_us: int) -> tuple[Image.Image, int, int]:
    best: tuple[int, dict[str, Any]] | None = None
    with AnyReader([bag_path]) as reader:
        connection = _connection(reader, COLOR_TOPIC)
        for index, (_, _, raw) in enumerate(reader.messages(connections=[connection])):
            frame = parse_orbbec_image(raw)
            timestamp = int(frame["header"]["timestamp_us"])
            delta = abs(timestamp - target_timestamp_us)
            if best is None or delta < best[0]:
                best = (delta, frame)
            if timestamp > target_timestamp_us and best is not None and delta > best[0]:
                break
    if best is None:
        raise RuntimeError(f"No RGB frames found in {bag_path}")
    frame = best[1]
    image = Image.open(io.BytesIO(bytes(frame["pixels"]))).convert("RGB")
    return image, int(frame["header"]["timestamp_us"]), int(best[0])


def load_alignment_midpoint(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    good = [row for row in rows if row.get("within_20ms") == "1"]
    if not good:
        raise RuntimeError(f"No valid camera/mocap alignment rows in {path}")
    return good[len(good) // 2]


def load_body_frame(path: Path, frame_counter: int) -> dict[str, np.ndarray]:
    best: tuple[int, dict[str, str]] | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            current = int(row["FrameCounter"])
            delta = abs(current - frame_counter)
            if best is None or delta < best[0]:
                best = (delta, row)
            if current > frame_counter and best is not None and delta > best[0]:
                break
    if best is None:
        raise RuntimeError(f"No body rows in {path}")
    row = best[1]
    result: dict[str, np.ndarray] = {}
    for name in ("LeftHand", "RightHand"):
        result[name] = np.asarray(
            [float(row[f"{name}_Pos_{axis}(mm)"]) for axis in "XYZ"], dtype=np.float64
        )
    result["frame_counter"] = np.asarray([int(row["FrameCounter"])], dtype=np.int64)
    return result


def project_world_points_to_rgb(
    world_points: np.ndarray,
    world_to_depth: np.ndarray,
    depth_to_color_rotation: np.ndarray,
    depth_to_color_translation: np.ndarray,
    color_profile: dict[str, Any],
) -> np.ndarray:
    points = np.asarray(world_points, dtype=np.float64)
    depth = points @ world_to_depth[:3, :3].T + world_to_depth[:3, 3]
    return project_depth_to_color(
        depth,
        depth_to_color_rotation,
        depth_to_color_translation,
        color_profile,
    )


def validate_take(
    take_dir: Path,
    output_dir: Path,
    world_to_depth: np.ndarray,
    origin_depth: np.ndarray,
    basis_depth_from_world: np.ndarray,
    depth_profile: dict[str, Any],
    color_profile: dict[str, Any],
) -> dict[str, Any]:
    alignment_path = take_dir / "同步校验" / "camera_cmavatar_alignment.csv"
    alignment = load_alignment_midpoint(alignment_path)
    target_timestamp = int(alignment["color_device_timestamp_us"])
    bag_path = take_dir / "原始BAG与内参" / "camera_1_rgb_depth.bag"
    image, actual_timestamp, timestamp_delta = read_rgb_near_timestamp(bag_path, target_timestamp)

    take_number = take_dir.name.split("Take_")[-1]
    body_path = take_dir / "动捕" / f"Take_{take_number}" / f"Take_{take_number}_Body.cma"
    body = load_body_frame(body_path, int(alignment["cmavatar_frame_counter"]))
    names = ["LeftHand", "RightHand"]
    world_points = np.vstack([body[name] for name in names])
    pixels = project_world_points_to_rgb(
        world_points,
        world_to_depth,
        depth_profile["rotation"],
        depth_profile["translation_mm"],
        color_profile,
    )
    output, axis_pixels = draw_world_axes_rgb(
        image,
        origin_depth,
        basis_depth_from_world,
        depth_profile["rotation"],
        depth_profile["translation_mm"],
        color_profile,
    )
    draw = ImageDraw.Draw(output)
    colors = {"LeftHand": (255, 80, 220), "RightHand": (40, 240, 240)}
    for name, pixel, point in zip(names, pixels, world_points, strict=True):
        text = f"mocap {name} ({point[0]:.0f},{point[1]:.0f},{point[2]:.0f})mm"
        draw_circle_label(draw, pixel, text, colors[name], radius=14)
    output_name = f"validation_{take_dir.name}.png"
    output.save(output_dir / output_name)
    return {
        "take": take_dir.name,
        "diagnostic_image": output_name,
        "alignment_row_frame_counter": int(alignment["cmavatar_frame_counter"]),
        "body_frame_counter_used": int(body["frame_counter"][0]),
        "target_color_timestamp_us": target_timestamp,
        "actual_bag_color_timestamp_us": actual_timestamp,
        "bag_frame_timestamp_delta_us": timestamp_delta,
        "mocap_world_points_mm": {name: body[name].tolist() for name in names},
        "projected_rgb_pixels": {name: pixel.tolist() for name, pixel in zip(names, pixels, strict=True)},
        "axis_rgb_pixels": axis_pixels,
    }


def intrinsics_consistency(dataset_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for recording in sorted(path for path in dataset_root.iterdir() if path.is_dir() and path.name[:2].isdigit()):
        candidates = list(recording.glob("原始BAG与内参/camera_1_intrinsics.json"))
        if not candidates:
            continue
        data = json.loads(candidates[0].read_text(encoding="utf-8"))
        rgb = data["camera_param"]["rgb_intrinsic"]
        depth = data["camera_param"]["depth_intrinsic"]
        rows.append(
            {
                "recording": recording.name,
                "serial_number": data["device"]["serial_number"],
                "rgb": [rgb["fx"], rgb["fy"], rgb["cx"], rgb["cy"]],
                "depth": [depth["fx"], depth["fy"], depth["cx"], depth["cy"]],
            }
        )
    serials = sorted({row["serial_number"] for row in rows})
    same_rgb = len({tuple(row["rgb"]) for row in rows}) == 1
    same_depth = len({tuple(row["depth"]) for row in rows}) == 1
    return {
        "recordings_checked": len(rows),
        "serial_numbers": serials,
        "same_rgb_intrinsics": same_rgb,
        "same_depth_intrinsics": same_depth,
        "passed": len(serials) == 1 and same_rgb and same_depth and len(rows) == 4,
        "details": rows,
    }


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


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    config_path = args.config.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else dataset_root / "world_calibration_cs400_manual_20260830"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    reference_dir = dataset_root / config["reference_recording"]
    bag_path = reference_dir / "原始BAG与内参" / "camera_1_rgb_depth.bag"

    print(f"Reading reference bag: {bag_path}")
    reference = load_reference_bag(bag_path)
    depth_profile = reference["profiles"]["depth"]
    color_profile = reference["profiles"]["color"]
    fit_config = config["table_plane_fit"]
    plane_points = median_depth_points(reference["median_depth_mm"], depth_profile, fit_config)
    normal, plane_d, plane_stats = fit_plane_ransac(
        plane_points,
        threshold_mm=float(fit_config["ransac_threshold_mm"]),
        refine_threshold_mm=float(fit_config["refine_threshold_mm"]),
        seed=int(fit_config["random_seed"]),
    )

    marker_height = float(config["marker_center_height_above_table_mm"])
    marker_centers: dict[str, np.ndarray] = {}
    marker_feet: dict[str, np.ndarray] = {}
    for name, pixel in config["depth_marker_centers_px"].items():
        center, foot = reconstruct_marker(
            pixel, marker_height, normal, plane_d, depth_profile
        )
        marker_centers[name] = center
        marker_feet[name] = foot

    tabletop_origin, long_line_direction, short_line_direction, intersection_gap = line_intersection(
        marker_feet["long_far"],
        marker_feet["long_near"],
        marker_feet["short_near"],
        marker_feet["short_far"],
    )
    # Choose signs from the virtual vertex towards each physical arm.
    x_measured = 0.5 * (marker_feet["long_far"] + marker_feet["long_near"]) - tabletop_origin
    x_measured -= normal * float(normal @ x_measured)
    x_measured /= np.linalg.norm(x_measured)
    y_measured = 0.5 * (marker_feet["short_near"] + marker_feet["short_far"]) - tabletop_origin
    y_measured -= normal * float(normal @ y_measured)
    y_measured /= np.linalg.norm(y_measured)
    measured_axis_angle = math.degrees(math.acos(np.clip(float(x_measured @ y_measured), -1.0, 1.0)))

    z_axis = normal
    y_from_x = np.cross(z_axis, x_measured)
    y_from_x /= np.linalg.norm(y_from_x)
    if float(y_from_x @ y_measured) < 0.0:
        # This should never trigger for this manually labelled CS-400, but makes
        # a point-label error explicit instead of silently mirroring the world.
        raise RuntimeError("Manual marker labels imply a left-handed or reversed Y axis")
    # The two independently observed marker lines differ slightly from a
    # perfect right angle.  Split that residual equally across X and Y instead
    # of locking X and placing all visible error on Y.
    signed_y_residual = math.atan2(
        float(z_axis @ np.cross(y_from_x, y_measured)),
        float(y_from_x @ y_measured),
    )
    shared_axis_correction = 0.5 * signed_y_residual
    x_axis = rotate_about_axis(x_measured, z_axis, shared_axis_correction)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    basis_depth_from_world = np.column_stack([x_axis, y_axis, z_axis])

    marker_plane_origin = tabletop_origin + marker_height * z_axis
    if args.origin_height_mode == "marker-plane":
        origin = marker_plane_origin
        origin_description = (
            "CS-400 long/short arm virtual intersection at marker-center height "
            "near the black vertex hole"
        )
        tabletop_world_z_mm = -marker_height
    else:
        origin = tabletop_origin
        origin_description = "CS-400 long/short arm virtual intersection on the tabletop"
        tabletop_world_z_mm = 0.0

    world_to_depth = homogeneous(basis_depth_from_world, origin)
    depth_to_world_rotation = basis_depth_from_world.T
    depth_to_world_translation = -depth_to_world_rotation @ origin
    depth_to_world = homogeneous(depth_to_world_rotation, depth_to_world_translation)

    depth_to_color_rotation = depth_profile["rotation"]
    depth_to_color_translation = depth_profile["translation_mm"]
    world_to_color_rotation = depth_to_color_rotation @ basis_depth_from_world
    world_to_color_translation = depth_to_color_rotation @ origin + depth_to_color_translation
    world_to_color = homogeneous(world_to_color_rotation, world_to_color_translation)
    # The recorded Orbbec field is named rotationMatrix, but this particular
    # factory matrix is not perfectly orthonormal.  Its exact inverse must be
    # used; a transpose creates roughly 1% error in the color inverse transform.
    color_to_world_linear = np.linalg.inv(world_to_color_rotation)
    color_to_world_translation = -color_to_world_linear @ world_to_color_translation
    color_to_world = homogeneous(color_to_world_linear, color_to_world_translation)
    depth_to_color = homogeneous(depth_to_color_rotation, depth_to_color_translation)
    color_to_depth_linear = np.linalg.inv(depth_to_color_rotation)
    color_to_depth_translation = -color_to_depth_linear @ depth_to_color_translation
    color_to_depth = homogeneous(color_to_depth_linear, color_to_depth_translation)

    Image.fromarray(reference["median_depth_mm"]).save(
        output_dir / "reference_median_depth_mm.png"
    )
    reference["representative_rgb"].save(output_dir / "reference_rgb.png")
    make_depth_diagnostic(
        reference["median_depth_mm"],
        config["depth_marker_centers_px"],
        origin,
        basis_depth_from_world,
        depth_profile,
    ).save(output_dir / "reference_depth_markers_and_axes.png")
    axes_image, reference_axis_pixels = draw_world_axes_rgb(
        reference["representative_rgb"],
        origin,
        basis_depth_from_world,
        depth_to_color_rotation,
        depth_to_color_translation,
        color_profile,
    )
    axes_draw = ImageDraw.Draw(axes_image)
    marker_color_pixels = project_depth_to_color(
        np.vstack([marker_centers[name] for name in marker_centers]),
        depth_to_color_rotation,
        depth_to_color_translation,
        color_profile,
    )
    for name, pixel in zip(marker_centers, marker_color_pixels, strict=True):
        draw_circle_label(axes_draw, pixel, name, (255, 255, 255), radius=7)
    marker_plane_pixels = draw_marker_plane_guides(
        axes_image,
        tabletop_origin,
        marker_height,
        basis_depth_from_world,
        depth_to_color_rotation,
        depth_to_color_translation,
        color_profile,
        origin_height_mode=args.origin_height_mode,
    )
    axes_image.save(output_dir / "reference_rgb_markers_and_world_axes.png")
    axes_image.crop((430, 650, 1120, 1030)).resize((1380, 760)).save(
        output_dir / "reference_rgb_marker_vs_ground_axes_crop.png"
    )
    make_rgb_depth_overlay(
        reference["representative_rgb"],
        reference["median_depth_mm"],
        depth_profile,
        color_profile,
    ).save(output_dir / "reference_depth_to_rgb_registration.png")

    validation: list[dict[str, Any]] = []
    if not args.skip_take_validation:
        for take_dir in sorted(dataset_root.glob("0[1-9]_*Take_*")):
            print(f"Validating fixed calibration on {take_dir.name}")
            validation.append(
                validate_take(
                    take_dir,
                    output_dir,
                    world_to_depth,
                    origin,
                    basis_depth_from_world,
                    depth_profile,
                    color_profile,
                )
            )

    marker_reconstruction = {
        name: {
            "depth_pixel": config["depth_marker_centers_px"][name],
            "center_depth_camera_mm": marker_centers[name],
            "table_foot_depth_camera_mm": marker_feet[name],
            "distance_from_selected_origin_mm": float(
                np.linalg.norm(
                    (marker_centers[name] if args.origin_height_mode == "marker-plane" else marker_feet[name])
                    - origin
                )
            ),
            "distance_from_tabletop_origin_mm": float(
                np.linalg.norm(marker_feet[name] - tabletop_origin)
            ),
        }
        for name in marker_centers
    }
    result = {
        "schema": "movementcap.camera_to_mocap_world.v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "manual_visual_calibration_complete",
        "camera_type": "third_view",
        "method": (
            "manual_visual_CS400_markers_plus_109_frame_raw_depth_median_"
            f"{args.origin_height_mode.replace('-', '_')}_origin"
        ),
        "applies_to": [
            config["reference_recording"],
            *[path.name for path in sorted(dataset_root.glob("0[1-9]_*Take_*"))],
        ],
        "fixed_camera_assumption": True,
        "coordinate_system": {
            "origin": origin_description,
            "origin_height_mode": args.origin_height_mode,
            "tabletop_world_z_mm": tabletop_world_z_mm,
            "x_axis": "from origin outward along the CS-400 long arm",
            "y_axis": "from origin outward along the CS-400 short arm",
            "z_axis": "tabletop normal, upward",
            "handedness": "right_handed",
            "units": "millimetres",
        },
        "input": {
            "dataset_root": str(dataset_root),
            "reference_bag": str(bag_path),
            "manual_observation_config": str(config_path),
            "raw_depth_frame_count": reference["frame_count"],
            "depth_source": "raw mono16 millimetres from camera_1_rgb_depth.bag",
            "depth_preview_used_for_geometry": False,
        },
        "camera_consistency": intrinsics_consistency(dataset_root),
        "orbbec_profiles_from_bag": {
            "depth": {
                "resolution": [depth_profile["width"], depth_profile["height"]],
                "fps": depth_profile["fps"],
                "intrinsics_fx_fy_cx_cy": depth_profile["intrinsics"],
                "distortion_orbbec_k1_k2_k3_k4_k5_k6_p1_p2": depth_profile["distortion"],
            },
            "color": {
                "resolution": [color_profile["width"], color_profile["height"]],
                "fps": color_profile["fps"],
                "intrinsics_fx_fy_cx_cy": color_profile["intrinsics"],
                "distortion_orbbec_k1_k2_k3_k4_k5_k6_p1_p2": color_profile["distortion"],
            },
        },
        "table_plane_in_depth_camera": {
            "equation": "normal dot point_mm + d_mm = 0",
            "normal_up": normal,
            "d_mm": plane_d,
            **plane_stats,
        },
        "manual_visual_observations": {
            "marker_model": config["marker_model"],
            "marker_center_height_above_table_mm": marker_height,
            "markers": marker_reconstruction,
        },
        "axis_geometry_quality": {
            "measured_long_short_angle_deg": measured_axis_angle,
            "deviation_from_90_deg": abs(measured_axis_angle - 90.0),
            "line_intersection_gap_mm": intersection_gap,
            "axis_fit_method": "equal_weight_marker_line_orthogonal_bisector",
            "shared_axis_correction_deg": math.degrees(shared_axis_correction),
            "orthonormal_basis_determinant": float(np.linalg.det(basis_depth_from_world)),
            "measured_x_alignment_after_orthogonalization": float(x_axis @ x_measured),
            "measured_y_alignment_after_orthogonalization": float(y_axis @ y_measured),
            "recorded_depth_to_color_linear_orthogonality_error": float(
                np.max(np.abs(depth_to_color_rotation.T @ depth_to_color_rotation - np.eye(3)))
            ),
        },
        "world_origin_in_depth_camera_mm": origin,
        "tabletop_origin_in_depth_camera_mm": tabletop_origin,
        "marker_plane_origin_in_depth_camera_mm": marker_plane_origin,
        "world_axes_in_depth_camera_columns_xyz": basis_depth_from_world,
        "transforms": {
            "convention": "column point p_target = T_target_from_source @ [p_source; 1]",
            "depth_camera_to_world": depth_to_world,
            "world_to_depth_camera": world_to_depth,
            "color_camera_to_world": color_to_world,
            "world_to_color_camera": world_to_color,
            "depth_camera_to_color_camera": depth_to_color,
            "color_camera_to_depth_camera": color_to_depth,
        },
        "camera_origins_in_world_mm": {
            "depth_camera": depth_to_world_translation,
            "color_camera": color_to_world_translation,
        },
        "diagnostics": {
            "reference_axis_rgb_pixels": reference_axis_pixels,
            "marker_plane_guide_rgb_pixels": marker_plane_pixels,
            "files": [
                "reference_rgb.png",
                "reference_median_depth_mm.png",
                "reference_depth_markers_and_axes.png",
                "reference_rgb_markers_and_world_axes.png",
                "reference_rgb_marker_vs_ground_axes_crop.png",
                "reference_depth_to_rgb_registration.png",
                *[item["diagnostic_image"] for item in validation],
            ],
            "formal_take_mocap_reprojection": validation,
        },
        "review_notes": [
            "Marker identities were selected by visual inspection of the RGB video and median raw-depth holes.",
            (
                "ALTERNATIVE interpretation: Z=0 is the virtual arm intersection at marker-center height near "
                "the black vertex hole; the tabletop is Z=-45 mm. This does not apply the standard Motive "
                "CS-400 ground-plane origin convention."
                if args.origin_height_mode == "marker-plane"
                else "The CS-400 45 mm offset was applied so Z=0 is the tabletop, not the reflective marker-center plane."
            ),
            "The same transform is reused only because the camera and table/ruler frame did not move across the three formal takes.",
        ],
    }
    result_path = output_dir / "camera_to_world.json"
    result_path.write_text(
        json.dumps(json_ready(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = f"""# CS-400 camera/world calibration\n\nStatus: complete (manual visual marker classification + multi-frame metric depth).\n\n- Reference: `{config['reference_recording']}`\n- Origin height mode: `{args.origin_height_mode}`\n- Origin definition: {origin_description}\n- Tabletop world Z (mm): {tabletop_world_z_mm:.3f}\n- Raw depth frames: {reference['frame_count']}\n- World origin in depth camera (mm): `{np.array2string(origin, precision=3)}`\n- Long/short measured angle: {measured_axis_angle:.3f} deg\n- Table residual P50/P95: {plane_stats['residual_mm_p50']:.3f} / {plane_stats['residual_mm_p95']:.3f} mm\n- Camera consistency across all four recordings: {result['camera_consistency']['passed']}\n- Formal Take validation images: {len(validation)}\n\nPrimary machine-readable result: `camera_to_world.json`.\n\nThe depth-camera transform is primary. Color-camera transforms are included because the reference markers and validation overlays are viewed in RGB.\n"""
    (output_dir / "CALIBRATION_SUMMARY.md").write_text(summary, encoding="utf-8")

    print(f"Calibration written to: {result_path}")
    print(f"World origin in depth camera (mm): {origin}")
    print(f"Measured long/short angle: {measured_axis_angle:.3f} deg")
    print(
        "Table residual P50/P95 (mm): "
        f"{plane_stats['residual_mm_p50']:.3f}/{plane_stats['residual_mm_p95']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
