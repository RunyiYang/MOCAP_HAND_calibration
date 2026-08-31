from __future__ import annotations

import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

import gt_calib_viz as viz


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "同步整理_20260829_三段"
CALIBRATION = (
    DATASET
    / "movementcap_worldcalib_pointcloud_package_20260830"
    / "movementcap_ruler_worldcalib"
    / "results"
    / "manual_final"
    / "camera_to_world.json"
)


class DatasetContractTests(unittest.TestCase):
    def test_clock_mapping_snaps_only_tolerance_close_boundaries(self) -> None:
        first_device_s = 1_788_009_000.0
        last_device_s = first_device_s + 1.0
        mapping = viz.ClockMapping(
            device_s=np.asarray([first_device_s, last_device_s]),
            monotonic_s=np.asarray([10.0, 11.0]),
            cmavatar_s=np.asarray([20.0, 21.0]),
            capture_to_poll_s=np.asarray([0.25, 0.25]),
            poll_monotonic_s=np.asarray([10.25, 11.25]),
            poll_cmavatar_s=np.asarray([20.25, 21.25]),
        )
        targets = np.asarray(
            [
                first_device_s - 0.25e-6,
                first_device_s,
                last_device_s,
                last_device_s + 0.25e-6,
                first_device_s - 1.0e-6,
                last_device_s + 1.0e-6,
            ]
        )

        monotonic, cmavatar, inside = mapping.map(targets)
        np.testing.assert_array_equal(
            inside, [True, True, True, True, False, False]
        )
        np.testing.assert_allclose(monotonic[:4], [10.0, 10.0, 11.0, 11.0])
        np.testing.assert_allclose(cmavatar[:4], [20.0, 20.0, 21.0, 21.0])
        self.assertTrue(np.all(np.isnan(monotonic[~inside])))
        self.assertTrue(np.all(np.isnan(cmavatar[~inside])))

        poll_monotonic, poll_cmavatar, poll_inside = mapping.map_poll_time(targets)
        np.testing.assert_array_equal(poll_inside, inside)
        np.testing.assert_allclose(
            poll_monotonic[:4], [10.25, 10.25, 11.25, 11.25]
        )
        np.testing.assert_allclose(
            poll_cmavatar[:4], [20.25, 20.25, 21.25, 21.25]
        )
        self.assertTrue(np.all(np.isnan(poll_monotonic[~poll_inside])))
        self.assertTrue(np.all(np.isnan(poll_cmavatar[~poll_inside])))

        scalar_monotonic, scalar_cmavatar, scalar_inside = mapping.map(
            np.asarray(first_device_s)
        )
        self.assertTrue(bool(scalar_inside))
        self.assertEqual(float(scalar_monotonic), 10.0)
        self.assertEqual(float(scalar_cmavatar), 20.0)

    def test_rgb_bag_indices_match_mp4_frames(self) -> None:
        expected = {"01": 1815, "02": 1812, "03": 1810}
        for key, count in expected.items():
            segment = viz._discover_segment(DATASET, key)
            timestamps = viz.ros1_topic_index_timestamps(
                segment / "原始BAG与内参" / "camera_1_rgb_depth.bag"
            )
            self.assertEqual(len(timestamps), count)
            self.assertTrue(np.all(np.diff(timestamps) > 0.0))
            self.assertEqual(viz.video_info(segment / "视频" / "RGB.mp4").frame_count, count)

    def test_delivered_calibration_reproduces_diagnostic_pixels(self) -> None:
        calibration = viz.load_preview_calibration(CALIBRATION)
        diagnostics = calibration.payload["diagnostics"]["formal_take_mocap_reprojection"]
        for item in diagnostics:
            points = np.asarray(
                [
                    item["mocap_world_points_mm"]["LeftHand"],
                    item["mocap_world_points_mm"]["RightHand"],
                ],
                dtype=np.float64,
            )
            expected = np.asarray(
                [
                    item["projected_rgb_pixels"]["LeftHand"],
                    item["projected_rgb_pixels"]["RightHand"],
                ],
                dtype=np.float64,
            )
            actual, positive = viz.project_world_to_rgb(points, calibration)
            self.assertTrue(np.all(positive))
            np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0.0)

    def test_calibration_is_declared_for_all_formal_takes(self) -> None:
        payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
        applies = set(payload["applies_to"])
        for key in ("01", "02", "03"):
            self.assertIn(viz._discover_segment(DATASET, key).name, applies)

    def test_take_000_end_to_end_pose_preparation(self) -> None:
        segment = viz._discover_segment(DATASET, "01")
        correction_xyz_mm = (3.5, -44.0, 2.0)
        prepared = viz.prepare_take(
            segment,
            CALIBRATION,
            operator_world_translation_xyz_mm=correction_xyz_mm,
        )
        self.assertEqual(prepared.mocap_mm.shape, (1815, 2, 21, 3))
        self.assertEqual(
            prepared.operator_world_translation_xyz_mm,
            correction_xyz_mm,
        )
        self.assertEqual(int(np.count_nonzero(prepared.mocap_valid)), 1748)
        for side in ("left", "right"):
            self.assertEqual(prepared.glove_in_world_mm[side].shape, (1815, 20, 3))
            # The stricter v2 protocol rejects interpolation across glove
            # source gaps greater than the default 25 ms.
            self.assertEqual(int(np.count_nonzero(prepared.glove_valid[side])), 1275)
            registration = prepared.registrations[side]
            self.assertGreater(registration.scale, 0.5)
            self.assertLess(registration.scale, 1.5)
            self.assertGreater(registration.fit_frame_count, 1000)

        first, stop = 100, 104
        with tempfile.TemporaryDirectory() as directory:
            video_path, metrics_path, contact_path = viz.render_take(
                prepared,
                Path(directory) / "partial.mp4",
                output_width=320,
                start_frame=first,
                max_frames=stop - first,
                snapshot_count=0,
            )
            partial_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(viz.video_info(video_path).frame_count, stop - first)
            self.assertIsNone(contact_path)

        scope = partial_metrics["evaluation_scope"]
        self.assertEqual(scope["statistics_scope"], "selected_source_frame_interval")
        self.assertEqual(scope["source_frame_first"], first)
        self.assertEqual(scope["source_frame_stop_exclusive"], stop)
        self.assertEqual(scope["evaluated_rgb_frames"], stop - first)
        self.assertFalse(scope["covers_full_source_take"])
        self.assertEqual(
            partial_metrics["synchronization"]["synchronized_rgb_frames"],
            int(np.count_nonzero(prepared.mocap_valid[first:stop])),
        )
        display_correction = partial_metrics["display_correction"]
        self.assertEqual(
            display_correction["type"],
            viz.OPERATOR_DISPLAY_CORRECTION_TYPE,
        )
        self.assertEqual(
            display_correction["operator_world_translation_xyz_mm"],
            list(correction_xyz_mm),
        )
        self.assertEqual(display_correction["coordinate_system"], "mocap_world_mm")
        self.assertEqual(display_correction["units"], "mm")
        self.assertTrue(display_correction["active"])
        self.assertTrue(display_correction["display_only"])
        self.assertFalse(display_correction["is_ground_truth"])
        self.assertFalse(display_correction["is_independent_accuracy_evidence"])
        self.assertIn("do not validate", display_correction["metric_effect"])
        correction_header = viz._operator_display_correction_header(prepared)
        self.assertIn("-44.0", correction_header)
        self.assertIn("NOT GT", correction_header)
        # Display correction is deliberately not embedded into the calibration
        # registration artifact, preserving the existing profile contract.
        registration_profile = viz.registration_profile_payload(prepared)
        self.assertEqual(
            registration_profile["schema"],
            viz.GLOVE_REGISTRATION_PROFILE_SCHEMA,
        )
        self.assertNotIn("display_correction", registration_profile)
        self.assertNotIn(
            "operator_world_translation_xyz_mm", registration_profile
        )
        for side in ("left", "right"):
            self.assertEqual(
                partial_metrics["sides"][side]["valid_rgb_frames"],
                int(np.count_nonzero(prepared.glove_valid[side][first:stop])),
            )

        # Solved-pose contact sheets must sample from available pose frames.
        # Uniform source-frame sampling can alias with the intermittent glove
        # stream and produce an apparently empty six-cell review poster.
        with tempfile.TemporaryDirectory() as directory:
            _, _, solved_contact_path = viz.render_take(
                prepared,
                Path(directory) / "solved.mp4",
                output_width=320,
                start_frame=60,
                max_frames=12,
                snapshot_count=3,
                render_layer=viz.RENDER_LAYER_SOLVED_POSE,
            )
            self.assertIsNotNone(solved_contact_path)
            solved_contact = cv2.imread(str(solved_contact_path))
            self.assertIsNotNone(solved_contact)
            # At least one cell must contain the saturated green pose layer;
            # the header alone is too small to reach this conservative count.
            green = (
                (solved_contact[:, :, 1] > 150)
                & (solved_contact[:, :, 1] > solved_contact[:, :, 0] * 1.3)
                & (solved_contact[:, :, 1] > solved_contact[:, :, 2] * 1.3)
            )
            self.assertGreater(int(np.count_nonzero(green)), 300)

    def test_operator_world_translation_moves_anchor_and_fails_closed(self) -> None:
        mocap = np.arange(2 * 2 * 21 * 3, dtype=np.float64).reshape(2, 2, 21, 3)
        original = mocap.copy()
        translation = (12.5, -44.0, 3.0)
        shifted = viz._apply_operator_world_translation(mocap, translation)
        np.testing.assert_array_equal(mocap, original)
        np.testing.assert_allclose(
            shifted - mocap,
            np.broadcast_to(np.asarray(translation), mocap.shape),
            atol=0.0,
            rtol=0.0,
        )

        glove = np.arange(2 * 20 * 3, dtype=np.float64).reshape(2, 20, 3)
        registration = viz.SimilarityRegistration(1.0, np.eye(3), 2, 0.0, 0.0)
        anchored = viz.apply_registration(glove, mocap[:, 0], registration)
        shifted_anchored = viz.apply_registration(
            glove, shifted[:, 0], registration
        )
        np.testing.assert_allclose(
            shifted_anchored - anchored,
            np.broadcast_to(np.asarray(translation), anchored.shape),
            atol=0.0,
            rtol=0.0,
        )

        invalid_values = (
            (),
            (1.0, 2.0),
            (1.0, 2.0, 3.0, 4.0),
            (0.0, np.nan, 0.0),
            (0.0, np.inf, 0.0),
            "1,2,3",
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "exactly three finite XYZ"):
                    viz.prepare_take(
                        Path("must-not-be-read"),
                        CALIBRATION,
                        operator_world_translation_xyz_mm=value,
                    )
        with self.assertRaisesRegex(ValueError, "end in XYZ"):
            viz._apply_operator_world_translation(
                np.zeros((2, 21, 2), dtype=np.float64),
                (0.0, 0.0, 0.0),
            )

    def test_operator_world_translation_cli_axes_and_defaults(self) -> None:
        defaults = viz.parse_args([])
        self.assertEqual(defaults.operator_world_x_mm, 0.0)
        self.assertEqual(defaults.operator_world_y_mm, 0.0)
        self.assertEqual(defaults.operator_world_z_mm, 0.0)

        args = viz.parse_args(
            [
                "--operator-world-x-mm",
                "1.25",
                "--operator-world-y-mm",
                "-44",
                "--operator-world-z-mm",
                "3.5",
            ]
        )
        self.assertEqual(
            (args.operator_world_x_mm, args.operator_world_y_mm, args.operator_world_z_mm),
            (1.25, -44.0, 3.5),
        )
        for nonfinite in ("nan", "inf", "-inf"):
            with self.subTest(nonfinite=nonfinite):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        viz.parse_args(["--operator-world-y-mm", nonfinite])

    def test_interpolation_rejects_large_source_gaps(self) -> None:
        source_times = np.asarray([0.0, 0.01, 0.10, 0.11])
        values = source_times[:, None]
        targets = np.asarray([0.005, 0.05, 0.105])
        interpolated, valid = viz.interpolate_series(
            source_times,
            values,
            targets,
            max_gap_s=0.025,
        )
        np.testing.assert_array_equal(valid, [True, False, True])
        self.assertTrue(np.isnan(interpolated[1, 0]))

    def test_interpolation_exact_samples_include_both_endpoints(self) -> None:
        source_times = np.asarray([0.0, 1.0, 10.0])
        values = np.asarray([[0.0], [1.0], [10.0]])
        targets = np.asarray([0.0, 0.5, 1.0, 5.0, 10.0])
        interpolated, valid = viz.interpolate_series(
            source_times,
            values,
            targets,
            max_gap_s=0.25,
        )
        np.testing.assert_array_equal(valid, [True, False, True, False, True])
        np.testing.assert_allclose(interpolated[valid, 0], [0.0, 1.0, 10.0])
        self.assertTrue(np.all(np.isnan(interpolated[~valid])))

        quaternions = np.asarray(
            [
                [[1.0, 0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0, 0.0]],
            ]
        )
        interpolated_q, quaternion_valid = viz.interpolate_quaternions(
            source_times,
            quaternions,
            targets,
            max_gap_s=0.25,
        )
        np.testing.assert_array_equal(quaternion_valid, valid)
        np.testing.assert_allclose(
            interpolated_q[quaternion_valid],
            quaternions,
            atol=0.0,
            rtol=0.0,
        )

        low, high, alpha, span, bracket_valid = viz._interpolation_brackets(
            source_times,
            source_times,
            max_gap_s=0.25,
        )
        np.testing.assert_array_equal(low, [0, 1, 2])
        np.testing.assert_array_equal(high, [0, 1, 2])
        np.testing.assert_array_equal(bracket_valid, [True, True, True])
        np.testing.assert_allclose(alpha, 0.0, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(span, 0.0, atol=0.0, rtol=0.0)
        bracket_ms, nearest_ms = viz._interpolation_time_diagnostics_ms(
            source_times,
            source_times,
            np.ones(len(source_times), dtype=bool),
        )
        self.assertEqual(bracket_ms["count"], 3)
        self.assertEqual(bracket_ms["max"], 0.0)
        self.assertEqual(nearest_ms["count"], 3)
        self.assertEqual(nearest_ms["max"], 0.0)

    def test_wxyz_slerp_and_active_rotation_direction(self) -> None:
        half_angle = np.deg2rad(45.0)
        q0 = np.asarray([1.0, 0.0, 0.0, 0.0])
        q1 = np.asarray([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)])
        samples = np.stack(
            (
                np.stack((q0, q0)),
                np.stack((q1, -q1)),
            )
        )
        interpolated, valid = viz.interpolate_quaternions(
            np.asarray([0.0, 1.0]),
            samples,
            np.asarray([0.5]),
            max_gap_s=2.0,
        )
        np.testing.assert_array_equal(valid, [True])
        rotations = viz.quaternion_wxyz_to_matrix(interpolated)
        local_x = np.asarray([1.0, 0.0, 0.0])
        world = local_x @ rotations[0, 0].T
        np.testing.assert_allclose(
            world,
            [np.sqrt(0.5), np.sqrt(0.5), 0.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(rotations[0, 0], rotations[0, 1], atol=1e-12)

    def test_fixed_registration_recovers_known_rotation_and_scale(self) -> None:
        frames = 20
        glove = np.zeros((frames, 20, 3), dtype=np.float64)
        base = np.asarray(
            [[25.0, 20.0, 2.0], [8.0, 45.0, -1.0], [-12.0, 43.0, 3.0], [-28.0, 30.0, -2.0]]
        )
        glove[:, viz.GLOVE_PALM_INDICES] = base[None]
        angle = np.deg2rad(23.0)
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        scale = 0.91
        mocap = np.zeros((frames, 21, 3), dtype=np.float64)
        mocap[:, viz.MOCAP_PALM_INDICES] = base @ rotation.T * scale
        actual = viz.fit_fixed_palm_registration(
            glove,
            mocap,
            np.ones(frames, dtype=bool),
        )
        self.assertAlmostEqual(actual.scale, scale, places=12)
        np.testing.assert_allclose(actual.rotation, rotation, atol=1e-12)

    def test_mocap_root_fusion_recovers_canonical_registration(self) -> None:
        frames = 20
        glove = np.zeros((frames, 20, 3), dtype=np.float64)
        base = np.asarray(
            [[25.0, 20.0, 2.0], [8.0, 45.0, -1.0], [-12.0, 43.0, 3.0], [-28.0, 30.0, -2.0]]
        )
        glove[:, viz.GLOVE_PALM_INDICES] = base
        canonical_angle = np.deg2rad(23.0)
        canonical = np.asarray(
            [
                [np.cos(canonical_angle), -np.sin(canonical_angle), 0.0],
                [np.sin(canonical_angle), np.cos(canonical_angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        scale = 0.91
        local_target = base @ canonical.T * scale
        root_angles = np.linspace(-1.0, 1.0, frames)
        root_rotations = np.asarray(
            [
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
                for angle in root_angles
            ]
        )
        translations = np.stack(
            (np.arange(frames), np.arange(frames) * -0.5, np.full(frames, 1000.0)),
            axis=1,
        )
        mocap = np.zeros((frames, 21, 3), dtype=np.float64)
        mocap[:, 0] = translations
        mocap[:, viz.MOCAP_PALM_INDICES] = (
            np.einsum("ji,fki->fjk", local_target, root_rotations)
            + translations[:, None]
        )
        actual = viz.fit_fixed_palm_registration(
            glove,
            mocap,
            np.ones(frames, dtype=bool),
            mocap_wrist_rotations=root_rotations,
        )
        self.assertAlmostEqual(actual.scale, scale, places=12)
        np.testing.assert_allclose(actual.rotation, canonical, atol=1e-12)
        fused = viz.apply_registration(
            glove,
            mocap,
            actual,
            mocap_wrist_rotations=root_rotations,
        )
        np.testing.assert_allclose(
            fused[:, viz.GLOVE_PALM_INDICES],
            mocap[:, viz.MOCAP_PALM_INDICES],
            atol=1e-10,
        )

    def test_robust_registration_rejects_frame_outlier_without_shape_error(self) -> None:
        frames = 20
        glove = np.zeros((frames, 20, 3), dtype=np.float64)
        base = np.asarray(
            [[25.0, 20.0, 2.0], [8.0, 45.0, -1.0], [-12.0, 43.0, 3.0], [-28.0, 30.0, -2.0]]
        )
        glove[:, viz.GLOVE_PALM_INDICES] = base
        mocap = np.zeros((frames, 21, 3), dtype=np.float64)
        mocap[:, viz.MOCAP_PALM_INDICES] = base
        mocap[0, viz.MOCAP_PALM_INDICES] += 1000.0
        actual = viz.fit_fixed_palm_registration(
            glove,
            mocap,
            np.ones(frames, dtype=bool),
        )
        self.assertEqual(actual.fit_frame_count, frames - 1)
        self.assertAlmostEqual(actual.scale, 1.0, places=12)
        np.testing.assert_allclose(actual.rotation, np.eye(3), atol=1e-12)

    def test_fixed_registration_rejects_degenerate_geometry(self) -> None:
        glove = np.zeros((10, 20, 3), dtype=np.float64)
        glove[:, viz.GLOVE_PALM_INDICES, 0] = np.asarray([1.0, 2.0, 3.0, 4.0])
        mocap = np.zeros((10, 21, 3), dtype=np.float64)
        mocap[:, viz.MOCAP_PALM_INDICES, 0] = np.asarray([2.0, 4.0, 6.0, 8.0])
        with self.assertRaisesRegex(ValueError, "degenerate"):
            viz.fit_fixed_palm_registration(
                glove,
                mocap,
                np.ones(10, dtype=bool),
            )

    def test_take_000_registration_is_frozen_for_take_001(self) -> None:
        source = viz.prepare_take(viz._discover_segment(DATASET, "01"), CALIBRATION)
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "registration.json"
            viz.write_registration_profile(profile_path, source)
            source_name, registrations, profile = viz.load_registration_profile(profile_path)

            # Profiles predating pose_mode remain portable and continue to imply
            # the legacy glove-wrist-preserved behavior from their schema.
            profile.pop("pose_mode")
            profile.pop("target_finger_joints_used_in_fit", None)
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            legacy_source_name, legacy_registrations, _ = (
                viz.load_registration_profile(profile_path)
            )

        self.assertEqual(profile["schema"], viz.GLOVE_REGISTRATION_PROFILE_SCHEMA)
        self.assertEqual(legacy_source_name, source_name)
        for side in ("left", "right"):
            np.testing.assert_allclose(
                legacy_registrations[side].rotation,
                registrations[side].rotation,
                atol=0.0,
                rtol=0.0,
            )

        for side in ("left", "right"):
            self.assertAlmostEqual(
                registrations[side].scale,
                source.registrations[side].scale,
            )
            self.assertEqual(
                registrations[side].anchor_residual_p95_mm,
                source.registrations[side].anchor_residual_p95_mm,
            )

        target = viz.prepare_take(
            viz._discover_segment(DATASET, "02"),
            CALIBRATION,
            registrations=registrations,
            registration_source_segment=source_name,
        )
        self.assertFalse(target.registration_fitted_on_target)
        self.assertEqual(target.registration_source_segment, source.segment_dir.name)
        for side in ("left", "right"):
            self.assertEqual(
                target.registrations[side].fit_frame_count,
                source.registrations[side].fit_frame_count,
            )
            np.testing.assert_allclose(
                target.registrations[side].rotation,
                source.registrations[side].rotation,
            )
        metrics = viz.comparison_metrics(target)
        self.assertTrue(metrics["evaluation_protocol"]["is_cross_take_holdout"])
        self.assertIsNone(metrics["evaluation_protocol"]["absolute_6dof_metrics"])
        with self.assertRaisesRegex(ValueError, "holdout"):
            viz.registration_profile_payload(target)

    def test_mocap_root_fusion_is_frozen_and_explicitly_conditioned(self) -> None:
        source = viz.prepare_take(
            viz._discover_segment(DATASET, "01"),
            CALIBRATION,
            pose_mode=viz.POSE_MODE_MOCAP_ROOT,
        )
        self.assertAlmostEqual(source.registrations["left"].scale, 0.9276215059, places=8)
        self.assertAlmostEqual(source.registrations["right"].scale, 0.9244824133, places=8)
        profile = viz.registration_profile_payload(source)
        self.assertEqual(profile["schema"], viz.MOCAP_ROOT_PROFILE_SCHEMA)
        self.assertEqual(
            profile["artifact_type"], viz.REGISTRATION_PROFILE_ARTIFACT_TYPE
        )
        self.assertFalse(profile["target_finger_joints_used_in_fit"])
        self.assertTrue(
            profile["timestamp_alignment"][
                "capture_to_poll_latency_removed_row_by_row"
            ]
        )
        self.assertEqual(
            profile["quaternion_semantics"],
            "empirically verified active root-local-to-mocap-world",
        )
        self.assertFalse(
            profile["root_conditioning"]["target_finger_joints_used_in_fit"]
        )
        self.assertEqual(
            profile["root_conditioning"]["quaternion_semantics"],
            "empirically verified active root-local-to-mocap-world",
        )
        for side in ("left", "right"):
            side_payload = profile["sides"][side]
            np.testing.assert_allclose(
                side_payload["rotation_mocap_root_local_from_glove_local"],
                side_payload["rotation_mocap_from_glove"],
                atol=0.0,
                rtol=0.0,
            )
            canonical_only = dict(side_payload)
            canonical_only.pop("rotation_mocap_from_glove")
            loaded = viz._registration_from_payload(canonical_only)
            np.testing.assert_allclose(
                loaded.rotation,
                source.registrations[side].rotation,
                atol=0.0,
                rtol=0.0,
            )
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "mocap-root-registration.json"
            viz.write_registration_profile(profile_path, source)
            source_name, registrations, loaded_profile = (
                viz.load_registration_profile(profile_path)
            )
        self.assertEqual(loaded_profile["schema"], viz.MOCAP_ROOT_PROFILE_SCHEMA)
        self.assertEqual(
            loaded_profile["artifact_type"],
            viz.REGISTRATION_PROFILE_ARTIFACT_TYPE,
        )
        target = viz.prepare_take(
            viz._discover_segment(DATASET, "02"),
            CALIBRATION,
            registrations=registrations,
            registration_source_segment=source_name,
            pose_mode=viz.POSE_MODE_MOCAP_ROOT,
        )
        metrics = viz.comparison_metrics(target)
        protocol = metrics["evaluation_protocol"]
        self.assertEqual(metrics["schema"], viz.MOCAP_ROOT_METRICS_SCHEMA)
        self.assertEqual(
            metrics["artifact_type"], viz.EVALUATION_METRICS_ARTIFACT_TYPE
        )
        self.assertTrue(protocol["is_cross_take_holdout"])
        self.assertTrue(protocol["conditional_on_mocap_wrist_translation"])
        self.assertTrue(protocol["conditional_on_mocap_wrist_orientation"])
        self.assertFalse(protocol["target_finger_joints_used_in_fit"])
        scope = metrics["evaluation_scope"]
        self.assertEqual(scope["statistics_scope"], "full_source_take")
        self.assertTrue(scope["covers_full_source_take"])
        self.assertEqual(scope["evaluated_rgb_frames"], target.video.frame_count)
        synchronization = metrics["synchronization"]
        canonical_delta = synchronization["nearest_mocap_sample_delta_ms"]
        for alias in (
            "exposure_to_nearest_mocap_sample_absolute_error_ms",
            "camera_to_mocap_absolute_error_ms",
        ):
            self.assertEqual(synchronization[alias], canonical_delta)
            deprecation = synchronization["deprecated_metric_aliases"][alias]
            self.assertEqual(
                deprecation["replacement"], "nearest_mocap_sample_delta_ms"
            )
            self.assertIn(
                "not synchronization error or GT accuracy",
                deprecation["semantics"],
            )
        self.assertEqual(
            protocol["quaternion_semantics"],
            "empirically verified active root-local-to-mocap-world",
        )
        self.assertIsNone(protocol["absolute_6dof_metrics"])
        for side in ("left", "right"):
            registration = metrics["sides"][side]["registration"]
            self.assertFalse(registration["target_finger_joints_used_in_fit"])
            np.testing.assert_allclose(
                registration["rotation_mocap_root_local_from_glove_local"],
                registration["rotation_mocap_from_glove"],
                atol=0.0,
                rtol=0.0,
            )
        self.assertAlmostEqual(
            metrics["sides"]["left"]["root_normalized_non_thumb_joint_epe_mm"]["median"],
            11.485225,
            places=5,
        )
        self.assertAlmostEqual(
            metrics["sides"]["right"]["root_normalized_non_thumb_fingertip_epe_mm"]["median"],
            18.875880,
            places=5,
        )

    def test_mocap_root_profile_loader_is_portable_and_accepts_legacy_schema(self) -> None:
        source = viz.prepare_take(
            viz._discover_segment(DATASET, "01"),
            CALIBRATION,
            pose_mode=viz.POSE_MODE_MOCAP_ROOT,
        )
        profile = viz.registration_profile_payload(source)
        profile["schema"] = viz.LEGACY_MOCAP_ROOT_PROFILE_SCHEMA
        profile.pop("artifact_type")
        profile["camera_world_calibration"]["path"] = "/not-mounted/calibration.json"
        for item in profile["input_files"].values():
            item["path"] = "/not-mounted/input.dat"
        for item in profile["glove_solver"].values():
            item["meta_path"] = "/not-mounted/solver.meta.json"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-root-profile.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            source_name, registrations, loaded = viz.load_registration_profile(path)

        self.assertEqual(source_name, source.segment_dir.name)
        self.assertEqual(loaded["schema"], viz.LEGACY_MOCAP_ROOT_PROFILE_SCHEMA)
        for side in ("left", "right"):
            np.testing.assert_allclose(
                registrations[side].rotation,
                source.registrations[side].rotation,
                atol=0.0,
                rtol=0.0,
            )

    def test_mocap_root_profile_loader_rejects_semantic_tampering(self) -> None:
        source = viz.prepare_take(
            viz._discover_segment(DATASET, "01"),
            CALIBRATION,
            pose_mode=viz.POSE_MODE_MOCAP_ROOT,
        )
        profile = viz.registration_profile_payload(source)

        def set_bad_alias(payload: dict[str, object]) -> None:
            sides = payload["sides"]
            assert isinstance(sides, dict)
            left = sides["left"]
            assert isinstance(left, dict)
            alias = copy.deepcopy(left["rotation_mocap_from_glove"])
            alias[0][0] += 0.01
            left["rotation_mocap_from_glove"] = alias

        def remove_right(payload: dict[str, object]) -> None:
            sides = payload["sides"]
            assert isinstance(sides, dict)
            sides.pop("right")

        cases = {
            "metrics schema used as profile": lambda item: item.__setitem__(
                "schema", viz.MOCAP_ROOT_METRICS_SCHEMA
            ),
            "wrong artifact type": lambda item: item.__setitem__(
                "artifact_type", viz.EVALUATION_METRICS_ARTIFACT_TYPE
            ),
            "wrong pose mode": lambda item: item.__setitem__(
                "pose_mode", viz.POSE_MODE_GLOVE_WRIST
            ),
            "wrong fit joints": lambda item: item.__setitem__(
                "fit_joint_names", ["index_mcp", "middle_mcp", "ring_mcp"]
            ),
            "target fingers used": lambda item: item.__setitem__(
                "target_finger_joints_used_in_fit", True
            ),
            "tips used": lambda item: item.__setitem__(
                "fingertips_used_in_fit", True
            ),
            "wrong top quaternion semantics": lambda item: item.__setitem__(
                "quaternion_semantics", "passive world-to-local"
            ),
            "wrong root conditioning": lambda item: item["root_conditioning"].__setitem__(
                "conditional_on_mocap_wrist_orientation", False
            ),
            "callback time reused as exposure": lambda item: item[
                "timestamp_alignment"
            ].__setitem__("capture_to_poll_latency_removed_row_by_row", False),
            "wrong nested quaternion semantics": lambda item: item[
                "root_conditioning"
            ].__setitem__("quaternion_semantics", "passive world-to-local"),
            "rotation aliases disagree": set_bad_alias,
            "right side missing": remove_right,
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered-root-profile.json"
            for label, mutate in cases.items():
                with self.subTest(label=label):
                    tampered = copy.deepcopy(profile)
                    mutate(tampered)
                    path.write_text(json.dumps(tampered), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        viz.load_registration_profile(path)

            legacy = copy.deepcopy(profile)
            legacy["schema"] = viz.LEGACY_MOCAP_ROOT_PROFILE_SCHEMA
            legacy.pop("artifact_type")
            legacy["pose_mode"] = viz.POSE_MODE_GLOVE_WRIST
            path.write_text(json.dumps(legacy), encoding="utf-8")
            with self.assertRaises(ValueError):
                viz.load_registration_profile(path)

            old_metrics = viz.comparison_metrics(source)
            old_metrics["schema"] = viz.LEGACY_MOCAP_ROOT_PROFILE_SCHEMA
            old_metrics.pop("artifact_type")
            path.write_text(json.dumps(old_metrics), encoding="utf-8")
            with self.assertRaises(ValueError):
                viz.load_registration_profile(path)

    def test_external_registration_requires_source_segment(self) -> None:
        dummy = viz.SimilarityRegistration(1.0, np.eye(3), 10, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "registration_source_segment"):
            viz.prepare_take(
                Path("not-read"),
                CALIBRATION,
                registrations={"left": dummy, "right": dummy},
            )

    def test_solved_pose_layer_has_explicit_non_gt_semantics(self) -> None:
        metrics = viz.add_visualization_metadata(
            {}, viz.RENDER_LAYER_SOLVED_POSE
        )
        layer = metrics["visualization"]
        self.assertEqual(layer["render_layer"], viz.RENDER_LAYER_SOLVED_POSE)
        self.assertFalse(layer["draws_mocap_21_joint_skeleton"])
        self.assertTrue(layer["draws_glove_solved_20_joint_pose"])
        self.assertFalse(layer["draws_fingertip_error_connectors"])
        self.assertIn("not independent", layer["solved_pose_semantics"])

    def test_visualization_layer_rejects_unknown_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported render layer"):
            viz.add_visualization_metadata({}, "unknown")


if __name__ == "__main__":
    unittest.main()
