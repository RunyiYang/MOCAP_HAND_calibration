from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tarfile
from typing import Iterable, Sequence

import cv2
import numpy as np


CALIBRATION_MEMBER = (
    "movementcap_20260831_worldcalib/results/"
    "20260831_processed_tabletop_origin/camera_to_world.json"
)

# These are the only IDs present in every CMM frame across the synchronized
# Take_006 interval.  Other IDs are transient/ghost tracks and are excluded.
VIEWER_LEFT_MARKERS = (
    "11614", "11616", "11619", "11625", "11627", "11629",
    "11630", "11678", "11685", "12058", "12069",
)
VIEWER_RIGHT_MARKERS = (
    "11781", "11978", "12001", "12003", "12006", "12013",
    "12015", "12045", "12051", "12064", "12076",
)
PERSISTENT_MARKERS = VIEWER_LEFT_MARKERS + VIEWER_RIGHT_MARKERS

# Visual inspection identifies the proximal markers behind the black wrist
# blocks.  The remaining four IDs are a fixed same-take palm-only assignment;
# no fingertip marker participates in the root fit.
HAND_ROOT_CONDITIONING = {
    "right": {
        "root_marker": "12058",
        "palm_markers": ("11614", "11678", "11616", "11619"),
        "scale": 0.8986121252142041,
        "fit_residual_median_mm": 25.05,
        "fit_residual_p95_mm": 56.32,
    },
    "left": {
        "root_marker": "11781",
        "palm_markers": ("12003", "12001", "12013", "12015"),
        "scale": 0.8865779393164441,
        "fit_residual_median_mm": 24.50,
        "fit_residual_p95_mm": 56.26,
    },
}

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
    rgb_path: Path
    cmm_path: Path
    alignment_path: Path
    frame_summary_path: Path
    camera: CameraModel
    clock: FrameClock
    marker_world_mm: np.ndarray
    marker_ids: tuple[str, ...]
    source_rgb_frame_count: int
    source_depth_frame_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_camera_model(calibration_archive: Path, intrinsics_path: Path) -> CameraModel:
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
    expected = "camera_glove_recording_20260831_161610"
    if expected not in applies_to:
        raise ValueError(f"Calibration does not declare applicability to {expected}")
    return CameraModel(matrix, distortion, world_to_color, calibration, intrinsics)


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
    if not np.all(np.isin(steps, (1, 2, 3))):
        raise ValueError(f"Unexpected camera frame steps: {np.unique(steps)}")
    anchor_indices = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(steps)))
    # This independently reconstructed endpoint closes exactly on Depth index
    # 2618; RGB has one extra, unsynchronized tail frame and is intentionally cut.
    if anchor_indices[-1] != depth_frame_count - 1:
        raise ValueError(
            f"Derived last anchor {anchor_indices[-1]} does not match depth endpoint {depth_frame_count - 1}"
        )
    if rgb_frame_count != depth_frame_count + 1:
        raise ValueError("Expected the delivered RGB to contain exactly one unsynchronized tail frame")
    output_indices = np.arange(anchor_indices[-1] + 1, dtype=np.int64)
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


def load_persistent_markers(
    cmm_path: Path,
    target_counters: np.ndarray,
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
        required = {
            f"Marker_{marker}_Pos_{axis}(mm)"
            for marker in PERSISTENT_MARKERS
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
            points = []
            for marker in PERSISTENT_MARKERS:
                values = [row[f"Marker_{marker}_Pos_{axis}(mm)"].strip() for axis in "XYZ"]
                if not all(values):
                    raise ValueError(
                        f"Persistent marker {marker} is missing at CMM frame {counter}"
                    )
                points.append([float(value) for value in values])
            counters.append(counter)
            samples.append(np.asarray(points, dtype=np.float64))
    source_counters = np.asarray(counters, dtype=np.float64)
    source = np.asarray(samples, dtype=np.float64)
    if len(source_counters) < 2 or np.any(np.diff(source_counters) != 1):
        raise ValueError("Selected CMM counters are not contiguous")
    output = np.empty((len(target_counters), len(PERSISTENT_MARKERS), 3), dtype=np.float64)
    for marker_index in range(len(PERSISTENT_MARKERS)):
        for axis in range(3):
            output[:, marker_index, axis] = np.interp(
                target_counters,
                source_counters,
                source[:, marker_index, axis],
            )
    return output


def load_new_capture(
    dataset_root: Path,
    calibration_archive: Path,
) -> NewCapture:
    root = Path(dataset_root) / "camera_glove_recording_20260831_161610"
    rgb_path = root / "rgbd_unpack" / "RGB.mp4"
    depth_path = root / "rgbd_unpack" / "Depth.mp4"
    intrinsics_path = root / "rgbd_unpack" / "camera_1_intrinsics.json"
    cmm_path = root / "mocap" / "Take_006" / "Take_006.cmm"
    alignment_path = (
        root / "mocap" / "Take_006" / "alignment" / "camera_cmavatar_alignment.csv"
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
    camera = load_camera_model(calibration_archive, intrinsics_path)
    markers = load_persistent_markers(cmm_path, clock.target_cmm_counter)
    return NewCapture(
        root=root,
        rgb_path=rgb_path,
        cmm_path=cmm_path,
        alignment_path=alignment_path,
        frame_summary_path=frame_summary_path,
        camera=camera,
        clock=clock,
        marker_world_mm=markers,
        marker_ids=PERSISTENT_MARKERS,
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


def condition_solved_pose(
    local_points_mm: np.ndarray,
    markers_world_mm: np.ndarray,
    side: str,
) -> np.ndarray:
    config = HAND_ROOT_CONDITIONING[side]
    marker_index = {name: index for index, name in enumerate(PERSISTENT_MARKERS)}
    root = markers_world_mm[marker_index[str(config["root_marker"])]]
    palm_indices = [marker_index[name] for name in config["palm_markers"]]
    target = markers_world_mm[palm_indices] - root
    scale = float(config["scale"])
    source = local_points_mm[GLOVE_PALM_INDICES] * scale
    rotation = _proper_kabsch_rotation(source, target)
    return local_points_mm @ rotation.T * scale + root


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
) -> None:
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
    all_pixels, all_positive = project_world(capture.marker_world_mm, capture.camera)
    scale = output_width / source_width
    all_pixels *= scale
    try:
        for frame_index in capture.clock.output_indices:
            ok, frame = capture_video.read()
            if not ok:
                raise RuntimeError(f"RGB decode stopped at frame {frame_index}")
            frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
            for marker_index, marker in enumerate(capture.marker_ids):
                color = RAW_LEFT_BGR if marker in VIEWER_LEFT_MARKERS else RAW_RIGHT_BGR
                for trail_offset, radius in ((8, 2), (4, 3)):
                    prior = max(int(frame_index) - trail_offset, 0)
                    if all_positive[prior, marker_index]:
                        x, y = np.rint(all_pixels[prior, marker_index]).astype(int)
                        cv2.circle(frame, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
                if not all_positive[frame_index, marker_index]:
                    continue
                x, y = np.rint(all_pixels[frame_index, marker_index]).astype(int)
                cv2.circle(frame, (int(x), int(y)), 7, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, (int(x), int(y)), 5, color, -1, cv2.LINE_AA)
                if marker in {"12058", "11781"}:
                    cv2.circle(frame, (int(x), int(y)), 8, (255, 255, 255), 2, cv2.LINE_AA)
            nearest = int(capture.clock.nearest_anchor_row[frame_index])
            kind = (
                "ANCHOR" if frame_index in set(capture.clock.anchor_indices)
                else "INFERRED"
            )
            _draw_header(
                frame,
                (
                    ("Take_006 | RAW MOCAP MARKERS -> RGB | 22 persistent points", (245, 245, 245)),
                    (
                        f"RGB {frame_index}/{len(capture.clock.output_indices)-1} | "
                        f"CMM {capture.clock.target_cmm_counter[frame_index]:.3f} | {kind} | "
                        f"nearest dt={capture.clock.anchor_alignment_error_ms[nearest]:+.3f} ms",
                        (210, 230, 245),
                    ),
                    ("viewer-left +world-X / viewer-right -world-X | NO JOINT TOPOLOGY / NOT 21-JOINT GT", (90, 220, 255)),
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
            "schema": "gt_calib.take006_raw_markers.v1",
            "status": "visualization_complete",
            "source_rgb_frames": source_count,
            "rendered_frames": int(len(capture.clock.output_indices)),
            "excluded_unsynchronized_rgb_frames": [source_count - 1],
            "persistent_marker_count": len(capture.marker_ids),
            "viewer_left_marker_ids": list(VIEWER_LEFT_MARKERS),
            "viewer_right_marker_ids": list(VIEWER_RIGHT_MARKERS),
            "mocap_representation": "anonymous persistent raw marker clusters; no anatomical topology",
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
                "CMM marker IDs have no delivered left/right anatomy or joint-edge semantics.",
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
) -> None:
    poses, valid, diagnostics = load_solved_pose(capture)
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
                    world = condition_solved_pose(
                        poses[side][frame_index],
                        capture.marker_world_mm[frame_index],
                        side,
                    )
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
                    ("Take_006 | GLOVE SOLVED ARTICULATION 20-joint", (245, 245, 245)),
                    ("MOCAP wrist + 4 palm markers -> root SE(3) | scale frozen from Take_000", (105, 245, 160)),
                    ("FINGERTIPS NEVER FITTED | VIZ-only; not independent glove wrist 6DoF or GT", (90, 220, 255)),
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
            "schema": "gt_calib.take006_mocap_palm_conditioned_glove_pose.v1",
            "status": "visualization_complete",
            "rendered_frames": int(len(capture.clock.output_indices)),
            "validity": diagnostics,
            "root_conditioning": {
                "conditional_on_mocap_wrist_translation": True,
                "conditional_on_mocap_palm_orientation": True,
                "per_frame_fit_uses_only_wrist_and_four_palm_markers": True,
                "finger_or_fingertip_markers_used_in_fit": False,
                "glove_input": "root-local solved 20-joint keypoints",
                "sides": HAND_ROOT_CONDITIONING,
            },
            "evaluation_scope": "same-take visualization fit; not an independent accuracy evaluation",
            "limitations": [
                "The CMM does not deliver an anatomical wrist rigid body or joint topology.",
                "Global wrist/palm pose is conditioned on MOCAP markers and is not glove-only 6DoF.",
                "Invalid glove/camera intervals are shown as unavailable and are not held or extrapolated.",
            ],
        },
    )


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
            for point in marker_pixels:
                x, y = np.rint(point).astype(int)
                cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, (int(x), int(y)), 6, (40, 220, 255), 2, cv2.LINE_AA)
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
                    ("NO-GLOVE CALIBRATION | CS-400 world -> RGB", (245, 245, 245)),
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


def render_new_three(
    project_root: Path,
    destination: Path,
) -> dict[str, Path]:
    project_root = Path(project_root)
    destination = Path(destination)
    videos = destination / "videos"
    posters = destination / "posters"
    metrics = destination / "metrics"
    frame_maps = destination / "frame_maps"
    for directory in (videos, posters, metrics, frame_maps):
        directory.mkdir(parents=True, exist_ok=True)
    dataset_root = project_root / "thor_new4_20260831_processed"
    archive = project_root / "movementcap_20260831_worldcalib_tabletop_final.tar.gz"
    capture = load_new_capture(dataset_root, archive)
    frame_map = frame_maps / "take006_rgb_to_cmm.csv"
    write_frame_map(capture, frame_map)

    raw_video = videos / "07_take006_raw_mocap_markers.mp4"
    render_raw_markers(
        capture,
        raw_video,
        posters / "07_take006_raw_mocap_markers.jpg",
        metrics / "07_take006_raw_mocap_markers.json",
    )
    solved_video = videos / "08_take006_solved_hand_pose.mp4"
    render_solved_pose(
        capture,
        solved_video,
        posters / "08_take006_solved_hand_pose.jpg",
        metrics / "08_take006_solved_hand_pose.json",
    )
    no_glove_video = videos / "09_no_glove_world_calibration.mp4"
    no_glove_rgb = (
        dataset_root / "camera_glove_recording_20260831_155410"
        / "rgbd_unpack" / "RGB.mp4"
    )
    render_no_glove_calibration(
        no_glove_rgb,
        capture.camera,
        no_glove_video,
        posters / "09_no_glove_world_calibration.jpg",
        metrics / "09_no_glove_world_calibration.json",
    )
    return {
        "raw_markers": raw_video,
        "solved_pose": solved_video,
        "no_glove_calibration": no_glove_video,
        "frame_map": frame_map,
    }
