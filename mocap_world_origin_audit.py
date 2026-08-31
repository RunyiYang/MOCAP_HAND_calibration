#!/usr/bin/env python3
"""Render a reproducible MOCAP wrist-semantics audit.

This script is intentionally separate from the production video overlay.  It
reuses each formal take's existing strict-25-ms alignment CSV, interpolates the
same two Human.cma rows with the recorded alpha, and renders three views on
the exact same RGB frame:

* raw: the 42 delivered Human.cma joints, unchanged;
* rear proxy: the same unchanged 42 joints plus one display-only point behind
  each wrist, ``wrist - 0.8 * (middle_mcp - wrist)``;
* Z37 counterexample: every joint receives the same ``[0, 0, 37]`` mm world
  translation before projection.

The generated artifacts prove selection, timing, interpolation provenance, and
geometric invariants.  They are automated evidence only.  They do not provide
subjective pixel labels or independent 2D ground truth.  In particular, the
Z37 view is a 2D counterexample, not a recommended global calibration fix.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

import gt_calib_viz as viz


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "同步整理_20260829_三段"
DEFAULT_ALIGNMENT_DIR = PROJECT_ROOT / "outputs" / "mocap_video_alignment_review"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mocap_wrist_semantics_audit"

SUMMARY_SCHEMA = "gt_calib.mocap_wrist_semantics_audit.v1"
PROVENANCE_SCHEMA = "gt_calib.mocap_wrist_semantics_audit_frame.v1"
CANDIDATE_TRANSLATION_WORLD_MM = np.asarray((0.0, 0.0, 37.0), dtype=np.float64)
REAR_PROXY_MIDDLE_MCP_SCALE = 0.8
MIDDLE_MCP_INDEX = 9
SAMPLE_QUANTILES = (0.06, 0.18, 0.30, 0.42, 0.54, 0.66, 0.78, 0.90)
CROP_XYXY = (350, 360, 1570, 900)
CONTACT_PANEL_SIZE = (480, 270)
RAW_COLOR_BGR = (25, 25, 245)
REAR_PROXY_SKELETON_COLOR_BGR = (245, 220, 45)
REAR_PROXY_COLOR_BGR = (0, 165, 255)
Z37_COUNTEREXAMPLE_COLOR_BGR = (30, 230, 30)
OUTLINE_BGR = (8, 8, 8)

TAKES: Mapping[str, tuple[str, str]] = {
    "01": ("01_210814_Take_000", "Take_000"),
    "02": ("02_210955_Take_001", "Take_001"),
    "03": ("03_211139_Take_002", "Take_002"),
}

# Take02 output 661 motivated the hypothesis and is therefore explicitly
# excluded from this held-out rendering set even if a future count change makes
# a configured quantile land on it.
EXCLUDED_OUTPUT_FRAMES: Mapping[str, frozenset[int]] = {
    "01": frozenset(),
    "02": frozenset((661,)),
    "03": frozenset(),
}

ALIGNMENT_FIELDS = (
    "source_video_frame",
    "mocap_low_frame_counter",
    "mocap_high_frame_counter",
    "interpolation_alpha",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_relative(path: Path) -> str:
    source = Path(path).resolve()
    try:
        return source.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(source)


def _canonical_row_sha256(row: Mapping[str, str]) -> str:
    payload = json.dumps(dict(row), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _selected_output_frames(row_count: int, take_key: str) -> list[int]:
    if row_count < 2:
        raise ValueError(f"Alignment for Take {take_key} has too few rows")
    selected: list[int] = []
    excluded = EXCLUDED_OUTPUT_FRAMES[take_key]
    for quantile in SAMPLE_QUANTILES:
        index = int(round((row_count - 1) * quantile))
        while index in excluded or index in selected:
            index += 1
        if index >= row_count:
            raise ValueError(
                f"Cannot select held-out output frame for Take {take_key}: {index}"
            )
        selected.append(index)
    if len(selected) != len(SAMPLE_QUANTILES) or selected != sorted(selected):
        raise AssertionError("Held-out frame selection is not unique and ordered")
    if excluded.intersection(selected):
        raise AssertionError("A hypothesis-development frame entered held-out audit")
    return selected


def _human_entities() -> tuple[tuple[str, ...], tuple[str, ...]]:
    result: list[tuple[str, ...]] = []
    for root in ("Skeleton_0_LeftHand", "Skeleton_1_RightHand"):
        result.append(
            (
                root,
                *(
                    f"{root}{finger}{joint}"
                    for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky")
                    for joint in range(1, 5)
                ),
            )
        )
    return result[0], result[1]


def _load_selected_human_points(
    path: Path,
    counters: set[int],
) -> dict[int, np.ndarray]:
    entities = _human_entities()
    selected: dict[int, np.ndarray] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Human.cma has no header: {path}")
        required = {
            f"{entity}_Pos_{axis}(mm)"
            for side in entities
            for entity in side
            for axis in "XYZ"
        } | {"FrameCounter"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Human.cma lacks fields {sorted(missing)}: {path}")
        for row in reader:
            counter = int(row["FrameCounter"])
            if counter not in counters:
                continue
            hands = []
            for side in entities:
                hands.append(
                    np.asarray(
                        [
                            [
                                float(row[f"{entity}_Pos_{axis}(mm)"])
                                for axis in "XYZ"
                            ]
                            for entity in side
                        ],
                        dtype=np.float64,
                    )
                )
            points = np.stack(hands)
            if points.shape != (2, 21, 3) or not np.all(np.isfinite(points)):
                raise ValueError(
                    f"Invalid Human.cma hand points at counter {counter}: {path}"
                )
            selected[counter] = points
    if set(selected) != counters:
        missing_counters = sorted(counters - set(selected))
        raise ValueError(f"Human.cma lacks selected counters {missing_counters}: {path}")
    return selected


def _draw_pose(
    frame: np.ndarray,
    points_world_mm: np.ndarray,
    calibration: viz.PreviewCalibration,
    *,
    color: tuple[int, int, int],
    title: str,
) -> np.ndarray:
    pixels, positive = viz.project_world_to_rgb(points_world_mm, calibration)
    pixels[~positive] = np.nan
    for side_index in range(2):
        side = pixels[side_index]
        for chain in viz.MOCAP_CHAINS:
            chain_pixels = side[list(chain)]
            if not np.all(np.isfinite(chain_pixels)):
                continue
            integer = np.rint(chain_pixels).astype(np.int32)
            cv2.polylines(frame, [integer], False, OUTLINE_BGR, 10, cv2.LINE_AA)
            cv2.polylines(frame, [integer], False, color, 5, cv2.LINE_AA)
        for pixel in side:
            if not np.all(np.isfinite(pixel)):
                continue
            center = tuple(np.rint(pixel).astype(int))
            cv2.circle(frame, center, 7, OUTLINE_BGR, -1, cv2.LINE_AA)
            cv2.circle(frame, center, 4, color, -1, cv2.LINE_AA)
        root = side[0]
        if np.all(np.isfinite(root)):
            center = tuple(np.rint(root).astype(int))
            cv2.circle(frame, center, 18, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.circle(frame, center, 12, color, 3, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 78), (0, 0, 0), -1)
    cv2.putText(
        frame,
        title,
        (18, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return pixels


def _rear_proxy_points(points_world_mm: np.ndarray) -> np.ndarray:
    points = np.asarray(points_world_mm, dtype=np.float64)
    if points.shape != (2, 21, 3):
        raise ValueError(f"Expected two 21-joint hands, got {points.shape}")
    wrist = points[:, 0]
    middle_mcp = points[:, MIDDLE_MCP_INDEX]
    return wrist - REAR_PROXY_MIDDLE_MCP_SCALE * (middle_mcp - wrist)


def _draw_rear_proxy(
    frame: np.ndarray,
    raw_joint_pixels: np.ndarray,
    proxy_world_mm: np.ndarray,
    calibration: viz.PreviewCalibration,
) -> np.ndarray:
    proxy_pixels, positive = viz.project_world_to_rgb(proxy_world_mm, calibration)
    proxy_pixels[~positive] = np.nan
    for side_index in range(2):
        wrist = raw_joint_pixels[side_index, 0]
        proxy = proxy_pixels[side_index]
        if not np.all(np.isfinite(wrist)) or not np.all(np.isfinite(proxy)):
            continue
        wrist_xy = tuple(np.rint(wrist).astype(int))
        proxy_xy = tuple(np.rint(proxy).astype(int))
        cv2.arrowedLine(
            frame,
            wrist_xy,
            proxy_xy,
            OUTLINE_BGR,
            12,
            cv2.LINE_AA,
            tipLength=0.18,
        )
        cv2.arrowedLine(
            frame,
            wrist_xy,
            proxy_xy,
            REAR_PROXY_COLOR_BGR,
            6,
            cv2.LINE_AA,
            tipLength=0.18,
        )
        cv2.drawMarker(
            frame,
            proxy_xy,
            (255, 255, 255),
            cv2.MARKER_DIAMOND,
            30,
            8,
            cv2.LINE_AA,
        )
        cv2.drawMarker(
            frame,
            proxy_xy,
            REAR_PROXY_COLOR_BGR,
            cv2.MARKER_DIAMOND,
            24,
            4,
            cv2.LINE_AA,
        )
    return proxy_pixels


def _title_bar(width: int, message: str, *, height: int = 58) -> np.ndarray:
    result = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        result,
        message,
        (12, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def _label_panel(
    image: np.ndarray,
    label: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        result,
        label,
        (10, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        color,
        2,
        cv2.LINE_AA,
    )
    return result


@contextmanager
def _atomic_destination(path: Path, *, suffix: str = ".tmp"):
    """Yield a same-directory temporary path and atomically publish it.

    Keeping the temporary file beside the destination makes ``os.replace`` an
    atomic same-filesystem operation.  Failed encodes/serialization leave the
    previous artifact untouched and remove the temporary file.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=suffix,
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_png(path: Path, image: np.ndarray) -> None:
    destination = Path(path)
    # OpenCV chooses its encoder from the extension, so the temporary file must
    # retain the final image suffix rather than ending in a generic ``.tmp``.
    with _atomic_destination(destination, suffix=destination.suffix) as temporary:
        if not cv2.imwrite(str(temporary), image):
            raise OSError(f"OpenCV failed to write {destination}")


def _write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, str | int | float]],
) -> None:
    if not rows:
        raise ValueError("Cannot write provenance CSV without rows")
    with _atomic_destination(path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    with _atomic_destination(path) as temporary:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )


def _bone_lengths(points: np.ndarray) -> np.ndarray:
    edges = [
        (chain[index], chain[index + 1])
        for chain in viz.MOCAP_CHAINS
        for index in range(len(chain) - 1)
    ]
    return np.asarray(
        [
            np.linalg.norm(points[:, child] - points[:, parent], axis=-1)
            for parent, child in edges
        ],
        dtype=np.float64,
    )


def _assert_provenance_matches_alignment(
    provenance_path: Path,
    alignment_paths: Mapping[str, Path],
) -> int:
    with provenance_path.open("r", encoding="utf-8", newline="") as stream:
        provenance = list(csv.DictReader(stream))
    matches = 0
    cached: dict[str, list[dict[str, str]]] = {}
    for row in provenance:
        take_key = row["take_key"]
        if take_key not in cached:
            with alignment_paths[take_key].open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                cached[take_key] = list(csv.DictReader(stream))
        original = cached[take_key][int(row["output_frame"])]
        for field in ALIGNMENT_FIELDS:
            if row[field] != original[field]:
                raise AssertionError(
                    f"Provenance mismatch for Take {take_key} output "
                    f"{row['output_frame']} field {field}: "
                    f"{row[field]!r} != {original[field]!r}"
                )
        if row["alignment_row_sha256"] != _canonical_row_sha256(original):
            raise AssertionError("Alignment row digest changed during audit")
        matches += 1
    return matches


def run_audit(
    dataset_root: Path,
    alignment_dir: Path,
    calibration_path: Path,
    output_dir: Path,
) -> Path:
    dataset_root = Path(dataset_root).resolve()
    alignment_dir = Path(alignment_dir).resolve()
    calibration_path = Path(calibration_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration = viz.load_preview_calibration(calibration_path)
    provenance_rows: list[dict[str, str | int | float]] = []
    take_summaries: list[dict[str, Any]] = []
    alignment_paths: dict[str, Path] = {}
    decoded_source_frame_assertions = 0
    rigid_translation_max_abs_error_mm = 0.0
    bone_length_max_abs_error_mm = 0.0
    rear_proxy_joint_mutation_max_abs_error_mm = 0.0
    rear_proxy_joint_pixel_max_abs_error_px = 0.0

    for take_key, (segment_name, take_name) in TAKES.items():
        segment = dataset_root / segment_name
        alignment_path = (
            alignment_dir
            / f"{segment_name}_mocap_video_aligned_strict25.alignment.csv"
        )
        video_path = segment / "视频" / "RGB.mp4"
        human_path = segment / "动捕" / take_name / f"{take_name}_Human.cma"
        for required in (alignment_path, video_path, human_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        alignment_paths[take_key] = alignment_path

        with alignment_path.open("r", encoding="utf-8", newline="") as stream:
            alignment_rows = list(csv.DictReader(stream))
        selected = _selected_output_frames(len(alignment_rows), take_key)
        counters = {
            int(alignment_rows[index][field])
            for index in selected
            for field in ("mocap_low_frame_counter", "mocap_high_frame_counter")
        }
        human = _load_selected_human_points(human_path, counters)
        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            raise OSError(f"Cannot open video {video_path}")
        cells: list[np.ndarray] = []
        frame_artifacts: list[dict[str, Any]] = []

        for output_frame in selected:
            alignment_row = alignment_rows[output_frame]
            if int(alignment_row["output_frame"]) != output_frame:
                raise AssertionError(
                    f"Alignment row ordinal mismatch in {alignment_path}: {output_frame}"
                )
            source_frame = int(alignment_row["source_video_frame"])
            low_counter = int(alignment_row["mocap_low_frame_counter"])
            high_counter = int(alignment_row["mocap_high_frame_counter"])
            alpha = float(alignment_row["interpolation_alpha"])
            if not (0.0 <= alpha <= 1.0):
                raise ValueError(f"Invalid interpolation alpha {alpha}")
            points = human[low_counter] * (1.0 - alpha) + human[high_counter] * alpha
            rear_panel_points = points.copy()
            rear_proxy = _rear_proxy_points(points)
            z37_counterexample = points + CANDIDATE_TRANSLATION_WORLD_MM
            rear_proxy_joint_mutation_max_abs_error_mm = max(
                rear_proxy_joint_mutation_max_abs_error_mm,
                float(np.max(np.abs(rear_panel_points - points))),
            )
            translation_error = float(
                np.max(
                    np.abs(
                        (z37_counterexample - points)
                        - CANDIDATE_TRANSLATION_WORLD_MM.reshape(1, 1, 3)
                    )
                )
            )
            rigid_translation_max_abs_error_mm = max(
                rigid_translation_max_abs_error_mm, translation_error
            )
            bone_error = float(
                np.max(
                    np.abs(
                        _bone_lengths(z37_counterexample) - _bone_lengths(points)
                    )
                )
            )
            bone_length_max_abs_error_mm = max(
                bone_length_max_abs_error_mm, bone_error
            )

            video.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            decoded, frame = video.read()
            if not decoded:
                raise OSError(f"Cannot decode {video_path} frame {source_frame}")
            decoder_next_frame = int(round(video.get(cv2.CAP_PROP_POS_FRAMES)))
            if decoder_next_frame != source_frame + 1:
                raise AssertionError(
                    f"Video seek decoded an unexpected frame in {video_path}: "
                    f"requested={source_frame}, next={decoder_next_frame}"
                )
            decoded_source_frame_assertions += 1

            raw_frame = frame.copy()
            rear_proxy_frame = frame.copy()
            z37_frame = frame.copy()
            raw_pixels = _draw_pose(
                raw_frame,
                points,
                calibration,
                color=RAW_COLOR_BGR,
                title="RAW: delivered 21 joints per hand",
            )
            rear_joint_pixels = _draw_pose(
                rear_proxy_frame,
                rear_panel_points,
                calibration,
                color=REAR_PROXY_SKELETON_COLOR_BGR,
                title="RAW 21 + rear proxy; joints unchanged",
            )
            rear_proxy_pixels = _draw_rear_proxy(
                rear_proxy_frame,
                rear_joint_pixels,
                rear_proxy,
                calibration,
            )
            rear_proxy_joint_pixel_max_abs_error_px = max(
                rear_proxy_joint_pixel_max_abs_error_px,
                float(np.nanmax(np.abs(rear_joint_pixels - raw_pixels))),
            )
            z37_pixels = _draw_pose(
                z37_frame,
                z37_counterexample,
                calibration,
                color=Z37_COUNTEREXAMPLE_COLOR_BGR,
                title="2D COUNTEREXAMPLE: global world +Z 37 mm",
            )

            x0, y0, x1, y1 = CROP_XYXY
            if not (
                0 <= x0 < x1 <= frame.shape[1]
                and 0 <= y0 < y1 <= frame.shape[0]
            ):
                raise ValueError(f"Audit crop {CROP_XYXY} exceeds video frame")
            raw_crop = _label_panel(
                raw_frame[y0:y1, x0:x1],
                "RAW 21",
                RAW_COLOR_BGR,
            )
            rear_proxy_crop = _label_panel(
                rear_proxy_frame[y0:y1, x0:x1],
                "RAW 21 + REAR PROXY (JOINTS UNCHANGED)",
                REAR_PROXY_COLOR_BGR,
            )
            z37_crop = _label_panel(
                z37_frame[y0:y1, x0:x1],
                "GLOBAL +Z37: 2D COUNTEREXAMPLE",
                Z37_COUNTEREXAMPLE_COLOR_BGR,
            )
            message = (
                f"{segment_name} | output {output_frame} | source RGB "
                f"{source_frame} | CMA {low_counter}->{high_counter} "
                f"alpha={alpha:.3f}"
            )
            full_triplet = np.hstack((raw_crop, rear_proxy_crop, z37_crop))
            pair_name = (
                f"{take_key}_output{output_frame:04d}_source{source_frame:04d}"
                "_raw-rearproxy-z37.png"
            )
            pair_path = output_dir / pair_name
            _write_png(
                pair_path,
                np.vstack(
                    (_title_bar(full_triplet.shape[1], message), full_triplet)
                ),
            )
            small_triplet = np.hstack(
                (
                    cv2.resize(raw_crop, CONTACT_PANEL_SIZE, interpolation=cv2.INTER_AREA),
                    cv2.resize(
                        rear_proxy_crop,
                        CONTACT_PANEL_SIZE,
                        interpolation=cv2.INTER_AREA,
                    ),
                    cv2.resize(
                        z37_crop,
                        CONTACT_PANEL_SIZE,
                        interpolation=cv2.INTER_AREA,
                    ),
                )
            )
            cells.append(
                np.vstack(
                    (_title_bar(small_triplet.shape[1], message), small_triplet)
                )
            )

            provenance_rows.append(
                {
                    "schema": PROVENANCE_SCHEMA,
                    "take_key": take_key,
                    "segment": segment_name,
                    "output_frame": alignment_row["output_frame"],
                    "source_video_frame": alignment_row["source_video_frame"],
                    "rgb_device_timestamp_us": alignment_row[
                        "rgb_device_timestamp_us"
                    ],
                    "cmavatar_target_timestamp_ns": alignment_row[
                        "cmavatar_target_timestamp_ns"
                    ],
                    "mocap_low_frame_counter": alignment_row[
                        "mocap_low_frame_counter"
                    ],
                    "mocap_high_frame_counter": alignment_row[
                        "mocap_high_frame_counter"
                    ],
                    "interpolation_alpha": alignment_row["interpolation_alpha"],
                    "alignment_row_sha256": _canonical_row_sha256(alignment_row),
                    "candidate_translation_world_x_mm": 0.0,
                    "candidate_translation_world_y_mm": 0.0,
                    "candidate_translation_world_z_mm": 37.0,
                    "rear_proxy_formula": "wrist - 0.8 * (middle_mcp - wrist)",
                    "rear_proxy_modifies_delivered_joints": "false",
                    "raw_left_root_x": float(raw_pixels[0, 0, 0]),
                    "raw_left_root_y": float(raw_pixels[0, 0, 1]),
                    "raw_right_root_x": float(raw_pixels[1, 0, 0]),
                    "raw_right_root_y": float(raw_pixels[1, 0, 1]),
                    "rear_proxy_left_world_x_mm": float(rear_proxy[0, 0]),
                    "rear_proxy_left_world_y_mm": float(rear_proxy[0, 1]),
                    "rear_proxy_left_world_z_mm": float(rear_proxy[0, 2]),
                    "rear_proxy_right_world_x_mm": float(rear_proxy[1, 0]),
                    "rear_proxy_right_world_y_mm": float(rear_proxy[1, 1]),
                    "rear_proxy_right_world_z_mm": float(rear_proxy[1, 2]),
                    "rear_proxy_left_x": float(rear_proxy_pixels[0, 0]),
                    "rear_proxy_left_y": float(rear_proxy_pixels[0, 1]),
                    "rear_proxy_right_x": float(rear_proxy_pixels[1, 0]),
                    "rear_proxy_right_y": float(rear_proxy_pixels[1, 1]),
                    "z37_left_root_x": float(z37_pixels[0, 0, 0]),
                    "z37_left_root_y": float(z37_pixels[0, 0, 1]),
                    "z37_right_root_x": float(z37_pixels[1, 0, 0]),
                    "z37_right_root_y": float(z37_pixels[1, 0, 1]),
                    "pair_image": pair_name,
                    "automated_evidence_not_subjective_pixel_gt": "true",
                    "z37_is_2d_counterexample_not_recommended_global_fix": "true",
                }
            )
            frame_artifacts.append(
                {
                    "output_frame": output_frame,
                    "source_video_frame": source_frame,
                    "pair_image": pair_name,
                }
            )

        video.release()
        if len(cells) != len(SAMPLE_QUANTILES):
            raise AssertionError("Unexpected contact-sheet cell count")
        contact = np.vstack(
            [np.hstack(cells[index : index + 2]) for index in range(0, 8, 2)]
        )
        contact_name = (
            f"{take_key}_{segment_name}_heldout_raw-rearproxy-z37_contact.png"
        )
        contact_path = output_dir / contact_name
        _write_png(contact_path, contact)
        take_summaries.append(
            {
                "take_key": take_key,
                "segment": segment_name,
                "alignment_csv": {
                    "path": _project_relative(alignment_path),
                    "sha256": _sha256(alignment_path),
                    "row_count": len(alignment_rows),
                },
                "rgb_video": {
                    "path": _project_relative(video_path),
                    "sha256": _sha256(video_path),
                },
                "human_cma": {
                    "path": _project_relative(human_path),
                    "sha256": _sha256(human_path),
                },
                "selected_output_frames": selected,
                "excluded_hypothesis_development_frames": sorted(
                    EXCLUDED_OUTPUT_FRAMES[take_key]
                ),
                "contact_sheet": contact_name,
                "contact_sheet_sha256": _sha256(contact_path),
                "frames": frame_artifacts,
            }
        )

    provenance_path = output_dir / "frame_provenance.csv"
    _write_csv_atomic(provenance_path, provenance_rows)
    exact_mapping_matches = _assert_provenance_matches_alignment(
        provenance_path, alignment_paths
    )
    if exact_mapping_matches != len(provenance_rows):
        raise AssertionError("Not every provenance row matched its alignment source")

    z37_root_shifts: dict[str, dict[str, dict[str, float]]] = {}
    rear_proxy_shifts: dict[str, dict[str, dict[str, float]]] = {}
    for take_key in TAKES:
        selected_rows = [row for row in provenance_rows if row["take_key"] == take_key]
        z37_root_shifts[take_key] = {}
        rear_proxy_shifts[take_key] = {}
        for side in ("left", "right"):
            dx = np.asarray(
                [
                    float(row[f"z37_{side}_root_x"])
                    - float(row[f"raw_{side}_root_x"])
                    for row in selected_rows
                ]
            )
            dy = np.asarray(
                [
                    float(row[f"z37_{side}_root_y"])
                    - float(row[f"raw_{side}_root_y"])
                    for row in selected_rows
                ]
            )
            z37_root_shifts[take_key][side] = {
                "dx_median_px": float(np.median(dx)),
                "dx_min_px": float(np.min(dx)),
                "dx_max_px": float(np.max(dx)),
                "dy_median_px": float(np.median(dy)),
                "dy_min_px": float(np.min(dy)),
                "dy_max_px": float(np.max(dy)),
            }
            rear_dx = np.asarray(
                [
                    float(row[f"rear_proxy_{side}_x"])
                    - float(row[f"raw_{side}_root_x"])
                    for row in selected_rows
                ]
            )
            rear_dy = np.asarray(
                [
                    float(row[f"rear_proxy_{side}_y"])
                    - float(row[f"raw_{side}_root_y"])
                    for row in selected_rows
                ]
            )
            rear_proxy_shifts[take_key][side] = {
                "dx_median_px": float(np.median(rear_dx)),
                "dx_min_px": float(np.min(rear_dx)),
                "dx_max_px": float(np.max(rear_dx)),
                "dy_median_px": float(np.median(rear_dy)),
                "dy_min_px": float(np.min(rear_dy)),
                "dy_max_px": float(np.max(rear_dy)),
            }

    summary_path = output_dir / "summary.json"
    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "automated_render_and_provenance_checks_passed",
        "purpose": (
            "Held-out wrist-semantics rendering of delivered 21-joint hands, "
            "a display-only rear-mount proxy, and a uniform +37 mm world-Z "
            "2D counterexample."
        ),
        "evidence_boundary": {
            "automated_evidence": [
                "exact reuse of strict25 source RGB frame",
                "exact reuse of CMA low/high frame counters and interpolation alpha",
                "rear proxy is derived from wrist and middle MCP without modifying any delivered joint",
                "Z37 counterexample differs only by one uniform [0,0,37] mm translation",
                "bone lengths are invariant under the Z37 counterexample translation",
                "deterministic contact sheets for human review",
            ],
            "not_provided": [
                "subjective pixel ground truth",
                "independent annotated wrist-module centers",
                "independent 2D joint reprojection error",
            ],
            "interpretation": (
                "Automated checks make the comparison reproducible but do not decide "
                "which overlay looks better or establish calibrated accuracy. The "
                "Z37 rendering is not a recommended global fix."
            ),
        },
        "selection": {
            "quantiles": list(SAMPLE_QUANTILES),
            "frames_per_take": len(SAMPLE_QUANTILES),
            "total_frames": len(provenance_rows),
            "take02_output_661_excluded": True,
        },
        "views": {
            "raw_21_joints": {
                "source": "unchanged interpolated Human.cma positions",
                "delivered_joint_count": 42,
            },
            "rear_mount_proxy": {
                "formula": "wrist - 0.8 * (middle_mcp - wrist)",
                "middle_mcp_index": MIDDLE_MCP_INDEX,
                "scale": REAR_PROXY_MIDDLE_MCP_SCALE,
                "additional_points": 2,
                "modifies_delivered_21_joint_hands": False,
                "status": "display_only_semantic_proxy_not_physical_ground_truth",
            },
            "global_z37_counterexample": {
            "translation_world_mm": CANDIDATE_TRANSLATION_WORLD_MM.tolist(),
            "application_order": (
                "after unchanged CMA timestamp interpolation, before camera projection"
            ),
            "applied_to": "all 42 joints identically",
                "status": "2d_counterexample_not_recommended_global_fix",
            },
        },
        "camera_calibration": {
            "path": _project_relative(calibration_path),
            "sha256": _sha256(calibration_path),
        },
        "render": {
            "crop_xyxy": list(CROP_XYXY),
            "contact_panel_size_wh": list(CONTACT_PANEL_SIZE),
            "raw_color_bgr": list(RAW_COLOR_BGR),
            "rear_proxy_skeleton_color_bgr": list(
                REAR_PROXY_SKELETON_COLOR_BGR
            ),
            "rear_proxy_color_bgr": list(REAR_PROXY_COLOR_BGR),
            "z37_counterexample_color_bgr": list(
                Z37_COUNTEREXAMPLE_COLOR_BGR
            ),
        },
        "assertions": {
            "exact_alignment_mapping_matches": exact_mapping_matches,
            "decoded_exact_source_frame_assertions": decoded_source_frame_assertions,
            "rear_proxy_joint_mutation_max_abs_error_mm": (
                rear_proxy_joint_mutation_max_abs_error_mm
            ),
            "rear_proxy_joint_pixel_max_abs_error_px": (
                rear_proxy_joint_pixel_max_abs_error_px
            ),
            "rigid_translation_max_abs_error_mm": rigid_translation_max_abs_error_mm,
            "bone_length_max_abs_error_mm": bone_length_max_abs_error_mm,
            "all_passed": bool(
                exact_mapping_matches == 24
                and decoded_source_frame_assertions == 24
                and rear_proxy_joint_mutation_max_abs_error_mm <= 1e-12
                and rear_proxy_joint_pixel_max_abs_error_px <= 1e-12
                and rigid_translation_max_abs_error_mm <= 1e-12
                and bone_length_max_abs_error_mm <= 1e-12
            ),
        },
        "rear_proxy_from_raw_wrist_pixel_shift_summary": rear_proxy_shifts,
        "z37_root_pixel_shift_summary": z37_root_shifts,
        "frame_provenance": {
            "path": provenance_path.name,
            "sha256": _sha256(provenance_path),
            "row_count": len(provenance_rows),
        },
        "takes": take_summaries,
    }
    if not summary["assertions"]["all_passed"]:
        raise AssertionError(f"Automated audit assertion failed: {summary['assertions']}")
    _write_json_atomic(summary_path, summary)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate held-out raw/rear-proxy/Z37 wrist-semantics audit artifacts."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_ALIGNMENT_DIR)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    calibration = (
        args.calibration.expanduser().resolve()
        if args.calibration is not None
        else viz._default_calibration(dataset_root)
    )
    summary_path = run_audit(
        dataset_root,
        args.alignment_dir.expanduser().resolve(),
        calibration,
        args.output_dir.expanduser().resolve(),
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
