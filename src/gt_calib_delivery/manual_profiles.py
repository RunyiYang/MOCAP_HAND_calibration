from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np


LEGACY_TAKE007_PROFILE_SCHEMA = "gt_calib.manual_xyz_profile.v1"
FINAL_NINE_PROFILE_SCHEMA = "gt_calib.final_nine_manual_xyz.v1"
FINAL_NINE_VIDEO_IDS = (
    "take01-mocap",
    "take01-solved",
    "take02-mocap",
    "take02-solved",
    "take03-mocap",
    "take03-solved",
    "take007-mocap-markers",
    "take007-solved",
    "no-glove-calibration",
)
HAND_OVERLAY_VIDEO_IDS = frozenset(FINAL_NINE_VIDEO_IDS[:8])
SIDE_RESIDUAL_VIDEO_IDS = frozenset(FINAL_NINE_VIDEO_IDS[6:8])
NO_GLOVE_VIDEO_ID = FINAL_NINE_VIDEO_IDS[8]
EXPECTED_FILENAMES = {
    "take01-mocap": "01_take01_mocap.mp4",
    "take01-solved": "02_take01_solved_pose.mp4",
    "take02-mocap": "03_take02_mocap.mp4",
    "take02-solved": "04_take02_solved_pose.mp4",
    "take03-mocap": "05_take03_mocap.mp4",
    "take03-solved": "06_take03_solved_pose.mp4",
    "take007-mocap-markers": "07_take007_labeled_mocap_markers.mp4",
    "take007-solved": "08_take007_aligned_hand_pose.mp4",
    "no-glove-calibration": "09_no_glove_world_calibration.mp4",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _vector(value: Any, *, field: str) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(type(component) not in (int, float) for component in value)
    ):
        raise ValueError(f"{field} must contain exactly three finite millimetres")
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain exactly three finite millimetres") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{field} must contain exactly three finite millimetres")
    return vector


def _zero() -> np.ndarray:
    return np.zeros(3, dtype=np.float64)


def _is_zero(vector: np.ndarray) -> bool:
    return bool(np.allclose(vector, 0.0, rtol=0.0, atol=0.0))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class VideoXYZAnnotation:
    order: int
    video_id: str
    filename: str
    source_video_sha256: str | None
    apply_translation: bool
    per_video_world_xyz_mm: np.ndarray
    side_residual_world_xyz_mm: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class FinalNineManualProfile:
    schema: str
    source_path: Path | None
    source_sha256: str | None
    raw_payload: Mapping[str, Any] | None
    global_world_xyz_mm: np.ndarray
    annotations: Mapping[str, VideoXYZAnnotation]

    def effective_world_xyz_mm(
        self,
        video_id: str,
        *,
        side: str | None = None,
    ) -> np.ndarray | None:
        if video_id not in self.annotations:
            raise KeyError(video_id)
        annotation = self.annotations[video_id]
        if not annotation.apply_translation:
            return None
        result = self.global_world_xyz_mm + annotation.per_video_world_xyz_mm
        if side is not None:
            if side not in {"left", "right"}:
                raise ValueError(f"Unknown hand side: {side}")
            result = result + annotation.side_residual_world_xyz_mm[side]
        return np.asarray(result, dtype=np.float64)

    def provenance(self) -> dict[str, Any] | None:
        if self.source_path is None:
            return None
        return {
            "schema": self.schema,
            "path": "calibration-workbench/applied_manual_profile.json",
            "sha256": self.source_sha256,
            "coordinate_contract": (
                "operator display translations are applied separately in each "
                "video's own MOCAP world; they are not a shared cross-session extrinsic"
            ),
            "applied_video_ids": [
                video_id
                for video_id in FINAL_NINE_VIDEO_IDS
                if self.annotations[video_id].apply_translation
            ],
            "excluded_video_ids": [
                video_id
                for video_id in FINAL_NINE_VIDEO_IDS
                if not self.annotations[video_id].apply_translation
            ],
        }


def _default_annotations() -> dict[str, VideoXYZAnnotation]:
    return {
        video_id: VideoXYZAnnotation(
            order=order,
            video_id=video_id,
            filename=EXPECTED_FILENAMES[video_id],
            source_video_sha256=None,
            apply_translation=video_id in HAND_OVERLAY_VIDEO_IDS,
            per_video_world_xyz_mm=_zero(),
            side_residual_world_xyz_mm={"left": _zero(), "right": _zero()},
        )
        for order, video_id in enumerate(FINAL_NINE_VIDEO_IDS, start=1)
    }


def default_final_nine_profile() -> FinalNineManualProfile:
    return FinalNineManualProfile(
        schema=FINAL_NINE_PROFILE_SCHEMA,
        source_path=None,
        source_sha256=None,
        raw_payload=None,
        global_world_xyz_mm=_zero(),
        annotations=_default_annotations(),
    )


def _load_legacy_take007(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
) -> FinalNineManualProfile:
    required = {
        "source_recording",
        "source_take",
        "coordinate_system",
        "units",
        "global_world_xyz_mm",
        "left_world_xyz_mm",
        "right_world_xyz_mm",
        "rear_offset_mm",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Legacy Take_007 profile is missing required fields: {missing}")
    if payload["source_recording"] != "camera_glove_recording_20260831_161912":
        raise ValueError("Legacy manual profile belongs to a different recording")
    if payload["source_take"] != "Take_007":
        raise ValueError("Legacy manual profile belongs to a different take")
    if payload["coordinate_system"] != "mocap_world_mm" or payload["units"] != "mm":
        raise ValueError("Legacy manual profile must use mocap_world_mm coordinates in mm")
    rear_value = payload["rear_offset_mm"]
    if type(rear_value) not in (int, float):
        raise ValueError("Legacy Take_007 rear_offset_mm must be a finite JSON number")
    rear = float(rear_value)
    if not math.isfinite(rear) or not np.isclose(rear, 20.0):
        raise ValueError("Legacy Take_007 rear_offset_mm must remain fixed at 20 mm")
    global_xyz = _vector(payload["global_world_xyz_mm"], field="global_world_xyz_mm")
    left = _vector(payload["left_world_xyz_mm"], field="left_world_xyz_mm")
    right = _vector(payload["right_world_xyz_mm"], field="right_world_xyz_mm")
    annotations = _default_annotations()
    for video_id in FINAL_NINE_VIDEO_IDS:
        base = annotations[video_id]
        apply = video_id in SIDE_RESIDUAL_VIDEO_IDS
        annotations[video_id] = VideoXYZAnnotation(
            order=base.order,
            video_id=video_id,
            filename=base.filename,
            source_video_sha256=None,
            apply_translation=apply,
            per_video_world_xyz_mm=_zero(),
            side_residual_world_xyz_mm={
                "left": left.copy() if apply else _zero(),
                "right": right.copy() if apply else _zero(),
            },
        )
    return FinalNineManualProfile(
        schema=LEGACY_TAKE007_PROFILE_SCHEMA,
        source_path=source_path,
        source_sha256=_sha256_file(source_path),
        raw_payload=payload,
        global_world_xyz_mm=global_xyz,
        annotations=annotations,
    )


def _load_final_nine(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
) -> FinalNineManualProfile:
    if payload.get("axis_order") != ["x", "y", "z"]:
        raise ValueError("Final-nine profile axis_order must be exactly ['x', 'y', 'z']")
    if payload.get("units") != "mm":
        raise ValueError("Final-nine profile units must be mm")
    if payload.get("coordinate_frame") != "per_video_mocap_world":
        raise ValueError(
            "Final-nine profile coordinate_frame must be per_video_mocap_world"
        )
    global_xyz = _vector(
        payload.get("global_world_xyz_mm"), field="global_world_xyz_mm"
    )
    rows = payload.get("video_annotations")
    if not isinstance(rows, list) or len(rows) != len(FINAL_NINE_VIDEO_IDS):
        raise ValueError("Final-nine profile must contain exactly nine video_annotations")
    annotations: dict[str, VideoXYZAnnotation] = {}
    for expected_order, expected_id in enumerate(FINAL_NINE_VIDEO_IDS, start=1):
        row = rows[expected_order - 1]
        if not isinstance(row, Mapping):
            raise ValueError(f"video_annotations[{expected_order - 1}] must be an object")
        row_order = row.get("order")
        if (
            type(row_order) is not int
            or row_order != expected_order
            or row.get("video_id") != expected_id
        ):
            raise ValueError(
                "Final-nine profile video_annotations must match canonical order and IDs"
            )
        filename = row.get("filename")
        if filename != EXPECTED_FILENAMES[expected_id]:
            raise ValueError(f"Unexpected filename for {expected_id}: {filename!r}")
        source_sha = row.get("video_sha256")
        if not isinstance(source_sha, str) or _SHA256_PATTERN.fullmatch(source_sha) is None:
            raise ValueError(f"video_sha256 for {expected_id} must be lowercase SHA-256")
        apply_translation = row.get("apply_translation")
        if not isinstance(apply_translation, bool):
            raise ValueError(f"apply_translation for {expected_id} must be boolean")
        expected_apply = expected_id in HAND_OVERLAY_VIDEO_IDS
        if apply_translation != expected_apply:
            raise ValueError(
                f"apply_translation for {expected_id} must be {str(expected_apply).lower()}"
            )
        per_video = _vector(
            row.get("per_video_world_xyz_mm"),
            field=f"{expected_id}.per_video_world_xyz_mm",
        )
        left = _vector(
            row.get("left_residual_world_xyz_mm"),
            field=f"{expected_id}.left_residual_world_xyz_mm",
        )
        right = _vector(
            row.get("right_residual_world_xyz_mm"),
            field=f"{expected_id}.right_residual_world_xyz_mm",
        )
        if expected_id not in SIDE_RESIDUAL_VIDEO_IDS and (
            not _is_zero(left) or not _is_zero(right)
        ):
            raise ValueError(
                f"Side-specific residuals are only supported for Take_007, not {expected_id}"
            )
        if expected_id == NO_GLOVE_VIDEO_ID and not _is_zero(per_video):
            raise ValueError("No-glove calibration evidence cannot receive a manual translation")
        annotations[expected_id] = VideoXYZAnnotation(
            order=expected_order,
            video_id=expected_id,
            filename=filename,
            source_video_sha256=source_sha,
            apply_translation=apply_translation,
            per_video_world_xyz_mm=per_video,
            side_residual_world_xyz_mm={"left": left, "right": right},
        )
    return FinalNineManualProfile(
        schema=FINAL_NINE_PROFILE_SCHEMA,
        source_path=source_path,
        source_sha256=_sha256_file(source_path),
        raw_payload=payload,
        global_world_xyz_mm=global_xyz,
        annotations=annotations,
    )


def load_final_nine_profile(path: Path | None) -> FinalNineManualProfile:
    if path is None:
        return default_final_nine_profile()
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Manual XYZ profile must be a JSON object")
    schema = payload.get("schema")
    if schema == LEGACY_TAKE007_PROFILE_SCHEMA:
        return _load_legacy_take007(payload, source_path=source)
    if schema == FINAL_NINE_PROFILE_SCHEMA:
        return _load_final_nine(payload, source_path=source)
    raise ValueError(f"Unsupported manual XYZ profile schema: {schema!r}")
