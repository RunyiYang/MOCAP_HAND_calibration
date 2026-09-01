from __future__ import annotations

import copy
from pathlib import Path
import json
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from gt_calib_delivery import cli
from gt_calib_delivery.imu_visualization_lab import (
    EXPECTED_CAMERA_INTRINSICS,
    EXPECTED_METHODS,
    EXPECTED_SEGMENTS,
    EXPECTED_TAKES,
    EXPECTED_VIDEOS,
    METHODS,
    IK_DIP_TO_PIP_RATIO,
    IK_MAX_DIP_FLEXION_DEG,
    IK_MAX_PIP_FLEXION_DEG,
    IK_MAX_THUMB_FLEXION_DEG,
    _high_frequency_residual,
    _ik_chain,
    _joint_flexion_degrees,
    _s2_log,
    _slerp_unit,
    _validate_anatomical_ik_metrics_contract,
    _write_motion_asset,
    _write_webpage,
    build_visualization_lab,
    cmm_guided_fixed_bone_ik,
    manifold_resample_pose,
    mocap_guided_posterior,
    rbf_same_sequence_fit,
    smooth_pose_s2,
)
from gt_calib_delivery.new_capture import GLOVE_CHAINS, MARKER_TO_GLOVE_INDICES


def _pose_sequence(count: int = 80) -> np.ndarray:
    pose = np.zeros((count, 20, 3), dtype=np.float64)
    pose[:, 0, 2] = 600.0
    for frame in range(count):
        for chain_index, chain in enumerate(GLOVE_CHAINS):
            for segment, (parent, child) in enumerate(zip(chain, chain[1:])):
                angle = 0.12 * np.sin(frame / 9.0 + chain_index + segment * 0.4)
                length = 20.0 + segment * 4.0
                direction = np.asarray((np.sin(angle), chain_index * 0.015, np.cos(angle)))
                direction /= np.linalg.norm(direction)
                pose[frame, child] = pose[frame, parent] + length * direction
    return pose


class ImuVisualizationLabTests(unittest.TestCase):
    def test_contract_is_four_takes_by_seven_methods(self) -> None:
        self.assertEqual(EXPECTED_TAKES, 4)
        self.assertEqual(EXPECTED_METHODS, 7)
        self.assertEqual(EXPECTED_VIDEOS, 28)
        self.assertEqual(len(METHODS), 7)
        self.assertEqual(len({method.id for method in METHODS}), 7)
        self.assertEqual(
            EXPECTED_SEGMENTS,
            (
                ("take005", "camera_glove_recording_20260831_161402/Take_005", 0, 1867),
                ("take006a", "camera_glove_recording_20260831_161610/Take_006", 0, 1309),
                ("take006b", "camera_glove_recording_20260831_161610/Take_006", 1309, 1310),
                ("take007", "camera_glove_recording_20260831_161912/Take_007", 0, 1981),
            ),
        )
        self.assertEqual(set(EXPECTED_CAMERA_INTRINSICS), set(EXPECTED_SEGMENTS[index][0] for index in range(4)))

    def test_builder_rejects_legacy_take007_only_manual_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch(
                "gt_calib_delivery.imu_visualization_lab.load_final_nine_profile",
                return_value=SimpleNamespace(schema="gt_calib.manual_calibration_profile.v1"),
            ):
                with self.assertRaisesRegex(ValueError, "legacy Take_007-only"):
                    build_visualization_lab(root, root / "delivery")

    def test_manifold_resampling_is_finite_and_preserves_bone_lengths(self) -> None:
        pose = _pose_sequence()
        source_times = np.arange(len(pose), dtype=np.float64) / 100.0
        target_times = np.arange(5, 75, 3, dtype=np.float64) / 100.0

        sampled, diagnostics = manifold_resample_pose(source_times, pose, target_times)

        self.assertEqual(sampled.shape, (len(target_times), 20, 3))
        self.assertTrue(np.all(np.isfinite(sampled)))
        self.assertEqual(diagnostics["finite_output_frames"], len(target_times))
        for chain in GLOVE_CHAINS:
            lengths = [
                np.linalg.norm(sampled[:, child] - sampled[:, parent], axis=-1)
                for parent, child in zip(chain, chain[1:])
            ]
            self.assertTrue(all(np.max(np.abs(values - np.median(values))) < 1e-9 for values in lengths))

    def test_zero_phase_s2_smoothing_reduces_high_frequency_motion(self) -> None:
        pose = _pose_sequence(120)
        pose[::2, 7:, 0] += 2.5

        smoothed = smooth_pose_s2(pose)

        raw_hf = _high_frequency_residual(pose)[:, 1:]
        smooth_hf = _high_frequency_residual(smoothed)[:, 1:]
        self.assertLess(float(np.median(smooth_hf)), float(np.median(raw_hf)))
        self.assertTrue(np.all(np.isfinite(smoothed)))

    def test_antipodal_s2_operations_have_a_deterministic_finite_geodesic(self) -> None:
        first = np.asarray([[[1.0, 0.0, 0.0]]])
        second = -first

        midpoint = _slerp_unit(first, second, np.asarray([0.5]))
        tangent = _s2_log(first, second)

        self.assertTrue(np.all(np.isfinite(midpoint)))
        np.testing.assert_allclose(np.linalg.norm(midpoint, axis=-1), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.sum(midpoint * first, axis=-1), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(tangent, axis=-1), np.pi, atol=1e-12)

    def test_zero_gain_posterior_is_target_and_two_percent_is_closer(self) -> None:
        raw = _pose_sequence(90)
        target = raw.copy()
        target[:, 1:, 0] += np.linspace(0.0, 18.0, 19)[None, :]

        clone = mocap_guided_posterior(raw, target, imu_residual_gain=0.0)
        two_percent = mocap_guided_posterior(raw, target, imu_residual_gain=0.02)
        raw_error = np.linalg.norm(raw - target, axis=-1)
        fused_error = np.linalg.norm(two_percent - target, axis=-1)

        np.testing.assert_allclose(clone, target, atol=1e-10)
        self.assertLess(float(np.median(fused_error)), float(np.median(raw_error)))

    def test_same_sequence_rbf_has_no_holdout_and_memorizes_training_pose(self) -> None:
        raw = _pose_sequence(45)
        target = raw.copy()
        target[:, 1:, 0] += 4.0 * np.sin(np.arange(len(raw))[:, None] / 7.0)

        predicted, diagnostics = rbf_same_sequence_fit(raw, target, fps=30.0)

        self.assertEqual(diagnostics["training_frames"], len(raw))
        self.assertEqual(diagnostics["rbf_centers"], len(raw))
        self.assertEqual(diagnostics["test_frames"], 0)
        self.assertTrue(diagnostics["training_and_render_sequence_are_identical"])
        self.assertLess(diagnostics["training_epe_mm"]["p95"], 1e-3)
        np.testing.assert_allclose(predicted[:, 0], target[:, 0], atol=0.0)

    def test_two_segment_ik_hits_endpoints_and_lengths(self) -> None:
        base = np.asarray((0.0, 0.0, 0.0))
        tip = np.asarray((50.0, 0.0, 0.0))
        guide = np.asarray((base, (25.0, 18.0, 0.0), tip))

        solved, _ = _ik_chain(base, tip, guide, (30.0, 30.0), [None])

        np.testing.assert_allclose(solved[0], base, atol=1e-12)
        np.testing.assert_allclose(solved[-1], tip, atol=1e-12)
        np.testing.assert_allclose(
            np.linalg.norm(np.diff(solved, axis=0), axis=-1),
            (30.0, 30.0),
            atol=1e-10,
        )

    def test_three_segment_ik_is_constant_curvature_anatomical_and_exact(self) -> None:
        base = np.asarray((0.0, 0.0, 0.0))
        tip = np.asarray((55.0, 0.0, 0.0))
        # Deliberately S-shaped guidance used to select opposite nested-sphere
        # branches in the old solver.
        guide = np.asarray((base, (20.0, 18.0, 2.0), (40.0, -14.0, -1.0), tip))
        lengths = (30.0, 25.0, 20.0)

        solved, returned_normals = _ik_chain(
            base, tip, guide, lengths, [None, None]
        )

        np.testing.assert_allclose(solved[0], base, atol=1e-12)
        np.testing.assert_allclose(solved[-1], tip, atol=1e-12)
        np.testing.assert_allclose(
            np.linalg.norm(np.diff(solved, axis=0), axis=-1),
            lengths,
            atol=1e-10,
        )
        segments = np.diff(solved, axis=0)
        pip_angle = float(_joint_flexion_degrees(segments[0], segments[1]))
        dip_angle = float(_joint_flexion_degrees(segments[1], segments[2]))
        first_bend = np.cross(segments[0], segments[1])
        second_bend = np.cross(segments[1], segments[2])
        self.assertGreater(float(np.dot(first_bend, second_bend)), 0.0)
        self.assertGreater(float(np.dot(returned_normals[0], returned_normals[1])), 0.999999)
        self.assertAlmostEqual(dip_angle / pip_angle, IK_DIP_TO_PIP_RATIO, places=10)
        self.assertLessEqual(pip_angle, IK_MAX_PIP_FLEXION_DEG)
        self.assertLessEqual(dip_angle, IK_MAX_DIP_FLEXION_DEG)

    def test_three_segment_ik_does_not_flip_temporal_branch(self) -> None:
        base = np.asarray((0.0, 0.0, 0.0))
        tip = np.asarray((60.0, 0.0, 0.0))
        lengths = (30.0, 25.0, 20.0)
        previous: list[np.ndarray | None] = [None, None]
        observed: list[np.ndarray] = []
        # Guidance crosses the chord and alternates its old preferred mirror
        # branch. The anatomical branch must remain in one temporal hemisphere.
        for guide_y in (15.0, 5.0, -1.0, -5.0, -15.0, 15.0, -15.0):
            guide = np.asarray(
                (
                    base,
                    (22.0, guide_y, 1.0),
                    (43.0, -0.7 * guide_y, -1.0),
                    tip,
                )
            )
            solved, previous = _ik_chain(base, tip, guide, lengths, previous)
            segments = np.diff(solved, axis=0)
            bend = np.cross(segments[0], segments[1])
            bend /= np.linalg.norm(bend)
            observed.append(bend)
            np.testing.assert_allclose(
                np.linalg.norm(segments, axis=-1), lengths, atol=1e-10
            )
            np.testing.assert_allclose(solved[-1], tip, atol=1e-12)
        consecutive_alignment = np.sum(
            np.asarray(observed[1:]) * np.asarray(observed[:-1]), axis=-1
        )
        self.assertTrue(np.all(consecutive_alignment > 0.0))

    def test_two_segment_thumb_ik_has_stable_branch_and_angle_bound(self) -> None:
        base = np.asarray((0.0, 0.0, 0.0))
        tip = np.asarray((50.0, 0.0, 0.0))
        lengths = (30.0, 30.0)
        previous: list[np.ndarray | None] = [None]
        observed: list[np.ndarray] = []
        for guide_y in (15.0, 2.0, -2.0, -15.0, 15.0):
            guide = np.asarray((base, (25.0, guide_y, 0.0), tip))
            solved, previous = _ik_chain(base, tip, guide, lengths, previous)
            segments = np.diff(solved, axis=0)
            bend = np.cross(segments[0], segments[1])
            bend /= np.linalg.norm(bend)
            observed.append(bend)
            flexion = float(_joint_flexion_degrees(segments[0], segments[1]))
            self.assertLessEqual(flexion, IK_MAX_THUMB_FLEXION_DEG)
            np.testing.assert_allclose(
                np.linalg.norm(segments, axis=-1), lengths, atol=1e-10
            )
        consecutive_alignment = np.sum(
            np.asarray(observed[1:]) * np.asarray(observed[:-1]), axis=-1
        )
        self.assertTrue(np.all(consecutive_alignment > 0.999999))

    def test_ik_rejects_flexion_beyond_anatomical_limits(self) -> None:
        base = np.asarray((0.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "constant-curvature anatomical limits"):
            _ik_chain(
                base,
                np.asarray((10.0, 0.0, 0.0)),
                np.asarray((base, (3.0, 4.0, 0.0), (7.0, 4.0, 0.0), (10.0, 0.0, 0.0))),
                (30.0, 25.0, 20.0),
                [None, None],
            )
        with self.assertRaisesRegex(ValueError, "Thumb IK exceeds"):
            _ik_chain(
                base,
                np.asarray((25.0, 0.0, 0.0)),
                np.asarray((base, (12.5, 10.0, 0.0), (25.0, 0.0, 0.0))),
                (30.0, 30.0),
                [None],
            )

    def test_cmm_ik_records_anatomical_acceptance_diagnostics(self) -> None:
        frame_count = 12
        raw = np.zeros((frame_count, 20, 3), dtype=np.float64)
        raw[:, 0] = (0.0, 0.0, 600.0)
        for frame in range(frame_count):
            translation = np.asarray((frame * 0.1, 0.0, 0.0))
            for finger_index, chain in enumerate(GLOVE_CHAINS):
                base = np.asarray((finger_index * 8.0, finger_index * 12.0, 600.0)) + translation
                if finger_index == 0:
                    lengths = (24.0, 20.0)
                    headings = (0.35, -0.25)
                else:
                    lengths = (30.0, 25.0, 20.0)
                    headings = (0.25, 0.65, 0.91)
                raw[frame, chain[1]] = base
                for segment_index, (parent, child) in enumerate(
                    zip(chain[1:], chain[2:])
                ):
                    angle = headings[segment_index]
                    raw[frame, child] = raw[frame, parent] + lengths[segment_index] * np.asarray(
                        (np.cos(angle), np.sin(angle), 0.0)
                    )
        markers = np.zeros((frame_count, 11, 3), dtype=np.float64)
        markers[:, :10] = raw[:, MARKER_TO_GLOVE_INDICES]
        markers[:, 10] = raw[:, 0]

        _, diagnostics = cmm_guided_fixed_bone_ik(
            raw, raw[:, 0], markers, side="left"
        )

        validation = diagnostics["anatomical_validation"]
        self.assertEqual(validation["opposite_bend_fraction_active_gt_5deg"], 0.0)
        self.assertTrue(all(validation["acceptance"].values()))
        self.assertLessEqual(validation["pip_flexion_deg"]["max"], IK_MAX_PIP_FLEXION_DEG)
        self.assertLessEqual(validation["dip_flexion_deg"]["max"], IK_MAX_DIP_FLEXION_DEG)
        self.assertLessEqual(validation["thumb_flexion_deg"]["max"], IK_MAX_THUMB_FLEXION_DEG)
        self.assertLess(validation["fixed_bone_length_drift_mm"]["max"], 1e-8)
        self.assertLess(
            validation["smoothed_cmm_base_tip_endpoint_epe_mm"]["p95"], 1e-8
        )

        metrics_payload = {
            "take_preparation": {
                "cmm_guided_ik": {
                    side: {"anatomical_validation": copy.deepcopy(validation)}
                    for side in ("left", "right")
                }
            }
        }
        _validate_anatomical_ik_metrics_contract(metrics_payload, label="synthetic")

        missing = copy.deepcopy(metrics_payload)
        del missing["take_preparation"]["cmm_guided_ik"]["left"]
        with self.assertRaisesRegex(ValueError, "anatomical IK contract missing"):
            _validate_anatomical_ik_metrics_contract(missing, label="missing")

        tampered = copy.deepcopy(metrics_payload)
        tampered["take_preparation"]["cmm_guided_ik"]["right"][
            "anatomical_validation"
        ]["acceptance"]["no_opposite_active_bends"] = False
        with self.assertRaisesRegex(ValueError, "anatomical IK acceptance contract"):
            _validate_anatomical_ik_metrics_contract(tampered, label="tampered")

    def test_ik_rejects_an_infeasible_endpoint_without_silent_tip_motion(self) -> None:
        base = np.asarray((0.0, 0.0, 0.0))
        tip = np.asarray((100.0, 0.0, 0.0))
        guide = np.asarray((base, (50.0, 10.0, 0.0), tip))

        with self.assertRaisesRegex(ValueError, "endpoint is infeasible"):
            _ik_chain(base, tip, guide, (30.0, 30.0), [None])

    def test_motion_asset_is_quantized_and_dual_layer(self) -> None:
        pose = _pose_sequence(3)
        mocap = np.concatenate((pose, pose[:, -1:, :]), axis=1)
        take = SimpleNamespace(
            id="take-test",
            frame_count=3,
            fps=30.0,
            mocap_draw={"left": mocap, "right": mocap + 2.0},
            mocap_chains=GLOVE_CHAINS,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.json"
            info = _write_motion_asset(
                path,
                take,
                METHODS[0],
                {"left": pose, "right": pose + 2.0},
                reference_extent_mm=321.0,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            written_bytes = path.stat().st_size

        self.assertEqual(payload["frame_count"], 3)
        self.assertEqual(
            [layer["id"] for layer in payload["layers"]],
            ["mocap-left", "mocap-right", "imu-left", "imu-right"],
        )
        self.assertEqual(payload["view"]["reference_extent_mm"], 321.0)
        self.assertEqual(len(payload["frames"]), 3)
        self.assertEqual(
            len(payload["frames"][0]),
            payload["encoding"]["values_per_frame"],
        )
        self.assertLessEqual(
            payload["validation"]["maximum_quantization_error_mm"],
            payload["encoding"]["quantum_mm"] / 2.0 + 1e-9,
        )
        self.assertEqual(info["bytes"], written_bytes)

    def test_3d_legend_colors_match_the_bgr_video_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            _write_webpage(destination, (), ())
            html = (destination / "index.html").read_text(encoding="utf-8")
            styles = (destination / "styles.css").read_text(encoding="utf-8")
            app = (destination / "app.js").read_text(encoding="utf-8")

        self.assertIn(".mocap-left{background:var(--cyan)}", styles)
        self.assertIn(".mocap-right{background:var(--magenta)}", styles)
        self.assertIn("'mocap-left': '#42deef'", app)
        self.assertIn("'mocap-right': '#f06be6'", app)
        self.assertIn("m.units!=='mm'", app)
        self.assertIn("e.joint_count!==62", app)
        self.assertIn("JSON.stringify(m.layers)!==JSON.stringify(MOTION_LAYERS)", app)
        self.assertIn('<option value="guided_ik" selected>', html)

    def test_cli_defaults_follow_selected_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            destination = root / "imu_mocap_visualization_lab"
            profile = root / "calibration_profiles/operator_y_minus_44_all_hand_overlays.v1.json"
            with mock.patch.object(
                cli, "build_visualization_lab", return_value=destination
            ) as builder:
                result = cli.main(
                    ["--project-root", str(root), "build-imu-visual-lab"]
                )

        self.assertEqual(result, 0)
        builder.assert_called_once_with(root, destination, manual_profile=profile)


if __name__ == "__main__":
    unittest.main()
