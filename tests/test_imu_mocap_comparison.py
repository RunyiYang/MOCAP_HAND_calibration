from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import gt_calib_viz as viz
from gt_calib_delivery import cli
from gt_calib_delivery.imu_mocap_comparison import (
    DELIVERY_SCHEMA,
    EXPECTED_COMPARISONS,
    _assert_tracked_bvh_registration,
    _fit_fixed_similarity,
    nearest_solver_samples,
    validate_comparison_delivery,
)


ROOT = Path(__file__).resolve().parents[1]


class ImuMocapComparisonTests(unittest.TestCase):
    def test_nearest_display_sampler_neither_interpolates_nor_gates(self) -> None:
        source_times = np.asarray((0.0, 0.01, 0.10), dtype=np.float64)
        points = np.asarray(((0.0,), (10.0,), (100.0,)), dtype=np.float64)
        target_times = np.asarray((0.006, 0.055, 0.099), dtype=np.float64)

        sampled = nearest_solver_samples(source_times, points, target_times)

        np.testing.assert_array_equal(sampled.indices, [1, 1, 2])
        np.testing.assert_array_equal(sampled.values[:, 0], [10.0, 10.0, 100.0])
        np.testing.assert_allclose(sampled.age_ms, [4.0, 45.0, 1.0], atol=1e-12)

    def test_nearest_display_sampler_refuses_unsupported_video_tail(self) -> None:
        with self.assertRaisesRegex(ValueError, "beyond solver support"):
            nearest_solver_samples(
                np.asarray((1.0, 2.0)),
                np.zeros((2, 20, 3)),
                np.asarray((0.9, 1.5)),
            )

    def test_comparison_cli_defaults_follow_explicit_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selected_root = Path(directory).resolve()
            expected_destination = selected_root / "imu_mocap_comparison_delivery"
            expected_profile = (
                selected_root
                / "calibration_profiles"
                / "operator_y_minus_44_all_hand_overlays.v1.json"
            )
            with mock.patch.object(
                cli,
                "build_comparison_delivery",
                return_value=expected_destination,
            ) as builder:
                result = cli.main(
                    [
                        "--project-root",
                        str(selected_root),
                        "build-imu-comparison",
                    ]
                )

        self.assertEqual(result, 0)
        builder.assert_called_once_with(
            selected_root,
            expected_destination,
            manual_profile=expected_profile,
        )

    def test_fixed_similarity_recovers_rotation_and_scale(self) -> None:
        rng = np.random.default_rng(9)
        source = rng.normal(size=(80, 4, 3)) * 40.0
        angle = np.deg2rad(31.0)
        rotation = np.asarray(
            (
                (np.cos(angle), -np.sin(angle), 0.0),
                (np.sin(angle), np.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        target = source @ rotation.T * 0.83
        target[7] += 300.0

        fitted = _fit_fixed_similarity(source, target)

        self.assertAlmostEqual(fitted.scale, 0.83, places=12)
        np.testing.assert_allclose(fitted.rotation, rotation, atol=1e-12)
        self.assertLess(fitted.retained_frame_count, fitted.input_frame_count)

    def test_tracked_old_registration_preserves_solver_wrist_orientation(self) -> None:
        profile = (
            ROOT
            / "calibration_profiles"
            / "imu_solver_vs_bvh_take01_registration.v1.json"
        )
        source, sides, payload = _assert_tracked_bvh_registration(ROOT, profile)

        self.assertEqual(source, "01_210814_Take_000")
        self.assertEqual(payload["pose_mode"], viz.POSE_MODE_GLOVE_WRIST)
        self.assertEqual(set(sides), {"left", "right"})
        self.assertIn("do not copy MOCAP", payload["wrist_orientation_policy"])
        self.assertEqual(
            payload["provenance"]["fit_position_source"],
            "Skeleton_0/1 BVH forward kinematics",
        )
        self.assertEqual(
            payload["provenance"]["strict_25ms_candidate_frames"],
            {"left": 1275, "right": 1275},
        )

    def test_validator_fails_closed_on_wrong_comparison_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": DELIVERY_SCHEMA,
                        "status": "pass",
                        "comparison_count": EXPECTED_COMPARISONS - 1,
                        "videos": [],
                    }
                ),
                encoding="utf-8",
            )
            (root / "SHA256SUMS.txt").write_text("", encoding="utf-8")

            result = validate_comparison_delivery(root)

        self.assertEqual(result["status"], "fail")
        self.assertTrue(any("exactly four" in item for item in result["failures"]))

    def test_reviewed_bundle_is_four_video_full_decode_pass(self) -> None:
        delivery = ROOT / "imu_mocap_comparison_delivery"
        manifest = json.loads((delivery / "manifest.json").read_text())
        validation = json.loads((delivery / "validation.json").read_text())

        self.assertEqual(manifest["schema"], DELIVERY_SCHEMA)
        self.assertEqual(manifest["comparison_count"], EXPECTED_COMPARISONS)
        self.assertEqual(len(manifest["videos"]), EXPECTED_COMPARISONS)
        self.assertTrue(manifest["canonical_final_nine_unchanged"])
        self.assertFalse(manifest["display_policy"]["pose_interpolation"])
        self.assertFalse(manifest["display_policy"]["pose_smoothing"])
        self.assertFalse(manifest["display_policy"]["validity_gate_hides_pose"])
        self.assertTrue(manifest["display_policy"]["stale_pose_is_drawn"])
        self.assertEqual(validation["status"], "pass")
        self.assertTrue(validation["full_decode"])
        self.assertEqual(validation["failures"], [])
        self.assertEqual(
            [entry["frame_count"] for entry in manifest["videos"]],
            [1747, 1805, 1799, 1981],
        )


if __name__ == "__main__":
    unittest.main()
