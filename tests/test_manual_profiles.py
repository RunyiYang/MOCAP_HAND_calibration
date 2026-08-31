from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gt_calib_delivery.manual_profiles import (
    EXPECTED_FILENAMES,
    FINAL_NINE_PROFILE_SCHEMA,
    FINAL_NINE_VIDEO_IDS,
    load_final_nine_profile,
)


def final_profile_payload() -> dict[str, object]:
    return {
        "schema": FINAL_NINE_PROFILE_SCHEMA,
        "created_at_utc": "2026-08-31T00:00:00Z",
        "axis_order": ["x", "y", "z"],
        "units": "mm",
        "coordinate_frame": "per_video_mocap_world",
        "global_world_xyz_mm": [0.0, -44.0, 0.0],
        "video_annotations": [
            {
                "order": order,
                "video_id": video_id,
                "filename": EXPECTED_FILENAMES[video_id],
                "video_sha256": f"{order:064x}",
                "apply_translation": order <= 8,
                "per_video_world_xyz_mm": [0.0, 0.0, 0.0],
                "left_residual_world_xyz_mm": [0.0, 0.0, 0.0],
                "right_residual_world_xyz_mm": [0.0, 0.0, 0.0],
            }
            for order, video_id in enumerate(FINAL_NINE_VIDEO_IDS, start=1)
        ],
    }


def legacy_profile_payload() -> dict[str, object]:
    return {
        "schema": "gt_calib.manual_xyz_profile.v1",
        "source_recording": "camera_glove_recording_20260831_161912",
        "source_take": "Take_007",
        "coordinate_system": "mocap_world_mm",
        "units": "mm",
        "global_world_xyz_mm": [0.0, -44.0, 0.0],
        "left_world_xyz_mm": [1.0, 0.0, 0.0],
        "right_world_xyz_mm": [-1.0, 0.0, 0.0],
        "rear_offset_mm": 20,
    }


def write_payload(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_final_profile_applies_minus_44_to_all_hand_overlays(tmp_path: Path) -> None:
    profile = load_final_nine_profile(write_payload(tmp_path, final_profile_payload()))
    for video_id in FINAL_NINE_VIDEO_IDS[:8]:
        np.testing.assert_array_equal(
            profile.effective_world_xyz_mm(video_id), [0.0, -44.0, 0.0]
        )
    assert profile.effective_world_xyz_mm("no-glove-calibration") is None
    provenance = profile.provenance()
    assert provenance is not None
    assert provenance["applied_video_ids"] == list(FINAL_NINE_VIDEO_IDS[:8])
    assert provenance["excluded_video_ids"] == ["no-glove-calibration"]


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda value: value.update({"axis_order": ["y", "x", "z"]}), "axis_order"),
        (lambda value: value.update({"units": "m"}), "units"),
        (
            lambda value: value.update({"coordinate_frame": "mocap_world"}),
            "coordinate_frame",
        ),
        (lambda value: value["video_annotations"].pop(), "exactly nine"),
        (
            lambda value: value["video_annotations"][0].update(
                {"video_id": "take02-mocap"}
            ),
            "canonical order",
        ),
        (
            lambda value: value["video_annotations"][8].update(
                {"apply_translation": True}
            ),
            "must be false",
        ),
        (
            lambda value: value["video_annotations"][0].update(
                {"left_residual_world_xyz_mm": [1, 0, 0]}
            ),
            "only supported for Take_007",
        ),
    ],
)
def test_final_profile_rejects_incompatible_payloads(
    tmp_path: Path,
    mutator: object,
    match: str,
) -> None:
    payload = final_profile_payload()
    mutator(payload)
    with pytest.raises(ValueError, match=match):
        load_final_nine_profile(write_payload(tmp_path, payload))


@pytest.mark.parametrize("invalid_order", [True, 1.0, "1"])
def test_final_profile_requires_order_to_be_json_integer(
    tmp_path: Path,
    invalid_order: object,
) -> None:
    payload = final_profile_payload()
    payload["video_annotations"][0]["order"] = invalid_order
    with pytest.raises(ValueError, match="canonical order"):
        load_final_nine_profile(write_payload(tmp_path, payload))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.update({"global_world_xyz_mm": [0.0, "-44", 0.0]}),
        lambda value: value["video_annotations"][0].update(
            {"per_video_world_xyz_mm": [False, 0.0, 0.0]}
        ),
        lambda value: value["video_annotations"][6].update(
            {"left_residual_world_xyz_mm": ["1", 0.0, 0.0]}
        ),
        lambda value: value["video_annotations"][7].update(
            {"right_residual_world_xyz_mm": [0.0, True, 0.0]}
        ),
        lambda value: value.update({"global_world_xyz_mm": [0.0, float("nan"), 0.0]}),
    ],
)
def test_final_profile_rejects_non_numeric_or_non_finite_vector_components(
    tmp_path: Path,
    mutator: object,
) -> None:
    payload = final_profile_payload()
    mutator(payload)
    with pytest.raises(ValueError, match="exactly three finite millimetres"):
        load_final_nine_profile(write_payload(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("global_world_xyz_mm", [0.0, "-44", 0.0]),
        ("left_world_xyz_mm", [True, 0.0, 0.0]),
        ("right_world_xyz_mm", [0.0, "0", 0.0]),
    ],
)
def test_legacy_profile_rejects_non_numeric_vector_components(
    tmp_path: Path,
    field: str,
    invalid_value: list[object],
) -> None:
    payload = legacy_profile_payload()
    payload[field] = invalid_value
    with pytest.raises(ValueError, match="exactly three finite millimetres"):
        load_final_nine_profile(write_payload(tmp_path, payload))


@pytest.mark.parametrize("invalid_rear_offset", [True, "20"])
def test_legacy_profile_requires_numeric_rear_offset(
    tmp_path: Path,
    invalid_rear_offset: object,
) -> None:
    payload = legacy_profile_payload()
    payload["rear_offset_mm"] = invalid_rear_offset
    with pytest.raises(ValueError, match="finite JSON number"):
        load_final_nine_profile(write_payload(tmp_path, payload))


def test_legacy_profile_stays_scoped_to_take007(tmp_path: Path) -> None:
    payload = legacy_profile_payload()
    profile = load_final_nine_profile(write_payload(tmp_path, payload))
    assert profile.effective_world_xyz_mm("take01-mocap") is None
    np.testing.assert_array_equal(
        profile.effective_world_xyz_mm("take007-mocap-markers", side="left"),
        [1.0, -44.0, 0.0],
    )
