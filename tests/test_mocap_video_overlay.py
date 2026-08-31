from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

import gt_calib_viz as viz
import mocap_video_overlay as overlay


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "同步整理_20260829_三段"
CALIBRATION = viz._default_calibration(DATASET)


def _mp4_sample(frame_index: int, frame: np.ndarray) -> overlay.Mp4FrameSample:
    luma = overlay._analysis_luma(frame)
    return overlay.Mp4FrameSample(
        frame_index=frame_index,
        width=frame.shape[1],
        height=frame.shape[0],
        decoded_bgr_sha256=overlay._array_sha256(frame),
        analysis_luma=luma,
        analysis_luma_sha256=overlay._array_sha256(luma),
        dhash64=overlay._dhash64(frame),
    )


def _bag_sample(
    message_index: int,
    frame: np.ndarray,
) -> overlay.BagColorSample:
    luma = overlay._analysis_luma(frame)
    device_timestamp_us = 1_000_000 + message_index * 33_333
    return overlay.BagColorSample(
        message_index=message_index,
        message_timestamp_ns=device_timestamp_us * 1000,
        device_timestamp_us=device_timestamp_us,
        frame_number=10_000 + message_index,
        encoding="mjpg",
        width=frame.shape[1],
        height=frame.shape[0],
        jpeg_sha256=overlay._bytes_sha256(frame.tobytes()),
        decoded_bgr_sha256=overlay._array_sha256(frame),
        analysis_luma=luma,
        analysis_luma_sha256=overlay._array_sha256(luma),
        dhash64=overlay._dhash64(frame),
    )


class VisualizationRearWristMountTests(unittest.TestCase):
    def test_anchor_extends_opposite_wrist_to_middle_mcp_direction(self) -> None:
        joints = np.zeros((2, 21, 3), dtype=np.float64)
        joints[0, 0] = [10.0, 20.0, 30.0]
        joints[0, 9] = [20.0, 20.0, 30.0]
        joints[1, 0] = [-4.0, 5.0, 6.0]
        joints[1, 9] = [-4.0, 15.0, 6.0]

        anchors = overlay.visualization_rear_wrist_mount_anchor(joints)

        np.testing.assert_allclose(anchors[0], [2.0, 20.0, 30.0])
        np.testing.assert_allclose(anchors[1], [-4.0, -3.0, 6.0])
        wrist_to_middle = joints[:, 9] - joints[:, 0]
        wrist_to_anchor = anchors - joints[:, 0]
        np.testing.assert_allclose(
            wrist_to_anchor,
            -overlay.VISUALIZATION_REAR_WRIST_MOUNT_SCALE * wrist_to_middle,
        )

    def test_anchor_does_not_modify_or_alias_input_joints(self) -> None:
        joints = np.arange(2 * 21 * 3, dtype=np.float32).reshape(2, 21, 3)
        original = joints.copy()

        anchors = overlay.visualization_rear_wrist_mount_anchor(joints)
        anchors[0, 0] = -12345.0

        np.testing.assert_array_equal(joints, original)
        self.assertFalse(np.shares_memory(anchors, joints))


class ExposureClockTests(unittest.TestCase):
    def test_callback_latency_is_removed_before_clock_interpolation(self) -> None:
        fields = (
            "color_device_timestamp_us",
            "thor_realtime_ns",
            "thor_monotonic_ns",
            "cmavatar_windows_estimated_ns",
        )
        rows = (
            {
                "color_device_timestamp_us": 100_000_000,
                "thor_realtime_ns": 100_250_000_000,
                "thor_monotonic_ns": 10_250_000_000,
                "cmavatar_windows_estimated_ns": 100_750_000_000,
            },
            {
                "color_device_timestamp_us": 101_000_000,
                "thor_realtime_ns": 101_260_000_000,
                "thor_monotonic_ns": 11_260_000_000,
                "cmavatar_windows_estimated_ns": 101_760_000_000,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clock.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            mapping = viz.load_clock_mapping(path)

        monotonic, cmavatar, inside = mapping.map(np.asarray([100.5]))
        poll_monotonic, poll_cmavatar, _ = mapping.map_poll_time(
            np.asarray([100.5])
        )
        np.testing.assert_allclose(monotonic, [10.5], atol=1e-9)
        np.testing.assert_allclose(cmavatar, [101.0], atol=1e-9)
        np.testing.assert_allclose(poll_monotonic, [10.755], atol=1e-9)
        np.testing.assert_allclose(poll_cmavatar, [101.255], atol=1e-9)
        np.testing.assert_allclose(mapping.capture_to_poll_s, [0.25, 0.26])
        np.testing.assert_array_equal(inside, [True])

    def test_interpolation_audit_accepts_exact_endpoints_and_propagates_nan(self) -> None:
        source = np.asarray([1.0, 2.0, 4.0])
        target = np.asarray([1.0, 4.0, np.nan])
        result = overlay.interpolation_audit(source, target, max_gap_s=0.5)
        np.testing.assert_array_equal(result.valid, [True, True, False])
        np.testing.assert_array_equal(result.low_indices[:2], [0, 2])
        np.testing.assert_array_equal(result.high_indices[:2], [0, 2])
        np.testing.assert_allclose(result.alpha[:2], [0.0, 0.0])
        np.testing.assert_allclose(result.bracket_span_s[:2], [0.0, 0.0])
        np.testing.assert_allclose(result.nearest_delta_s[:2], [0.0, 0.0])
        self.assertTrue(np.isnan(result.alpha[2]))


class FrameIdentityUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        generator = np.random.default_rng(20260831)
        cls.frames = [
            generator.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
            for _ in range(40)
        ]
        cls.indices = overlay._fixed_span_sample_indices(40, 31)
        cls.mp4 = {
            index: _mp4_sample(index, frame)
            for index, frame in enumerate(cls.frames)
        }

    def test_full_span_samples_and_gap_stem_are_parameterized(self) -> None:
        self.assertEqual(len(self.indices), 31)
        self.assertEqual((self.indices[0], self.indices[-1]), (0, 39))
        self.assertTrue(np.all(np.diff(self.indices) > 0))
        self.assertEqual(overlay._fixed_span_sample_indices(8, 0), list(range(8)))
        self.assertEqual(overlay._gap_stem_token(25.0), "strict25")
        self.assertEqual(overlay._gap_stem_token(50.0), "gap50")
        self.assertEqual(overlay._gap_stem_token(12.5), "gap12p5")
        self.assertTrue(
            overlay.alignment_output_stem("Take", 50.0).endswith("_gap50")
        )
        self.assertTrue(
            overlay.alignment_output_stem(
                "Take",
                25.0,
                overlay.MOCAP_POSITION_SOURCE_SKELETON_BVH,
            ).endswith("_strict25_skeleton_bvh")
        )
        with self.assertRaisesRegex(ValueError, "Unsupported MOCAP position source"):
            overlay.alignment_output_stem("Take", 25.0, "automatic")

    def test_bvh_axis_and_unit_inverse_is_explicit_and_fail_closed(self) -> None:
        bvh = np.asarray(
            [
                [-1.2, 3.4, 5.6],
                [7.8, -9.0, 1.1],
            ],
            dtype=np.float64,
        )
        world = overlay.bvh_positions_to_mocap_world_mm(bvh)
        np.testing.assert_allclose(
            world,
            [[12.0, 56.0, 34.0], [-78.0, 11.0, -90.0]],
        )
        with self.assertRaisesRegex(ValueError, r"shape \(\.\.\., 3\)"):
            overlay.bvh_positions_to_mocap_world_mm(np.zeros((4, 2)))
        with self.assertRaisesRegex(ValueError, "must be finite"):
            overlay.bvh_positions_to_mocap_world_mm(
                np.asarray([[0.0, np.nan, 0.0]])
            )

    def test_candidate_origin_and_rear_proxy_cli_defaults_are_explicit(self) -> None:
        with mock.patch("sys.argv", ["mocap_video_overlay.py"]):
            defaults = overlay.parse_args()
        self.assertEqual(
            defaults.mocap_position_source,
            overlay.MOCAP_POSITION_SOURCE_HUMAN_CMA,
        )
        self.assertEqual(defaults.mocap_world_x_offset_mm, 0.0)
        self.assertEqual(defaults.mocap_world_y_offset_mm, 0.0)
        self.assertEqual(defaults.mocap_world_z_offset_mm, 0.0)
        self.assertFalse(defaults.draw_rear_mount_proxy)

        with mock.patch(
            "sys.argv",
            [
                "mocap_video_overlay.py",
                "--mocap-position-source",
                "skeleton-bvh",
                "--mocap-world-x-offset-mm",
                "12",
                "--mocap-world-y-offset-mm",
                "-4.5",
                "--mocap-world-z-offset-mm",
                "37",
                "--draw-rear-mount-proxy",
            ],
        ):
            candidate = overlay.parse_args()
        self.assertEqual(
            candidate.mocap_position_source,
            overlay.MOCAP_POSITION_SOURCE_SKELETON_BVH,
        )
        self.assertEqual(candidate.mocap_world_x_offset_mm, 12.0)
        self.assertEqual(candidate.mocap_world_y_offset_mm, -4.5)
        self.assertEqual(candidate.mocap_world_z_offset_mm, 37.0)
        self.assertTrue(candidate.draw_rear_mount_proxy)
        self.assertEqual(
            overlay._mocap_world_translation_header((12.0, -4.5, 37.0)),
            "CANDIDATE origin +X12 -Y4.5 +Z37 mm / not GT",
        )

        with mock.patch(
            "sys.argv",
            [
                "mocap_video_overlay.py",
                "--mocap-position-source",
                "automatic",
            ],
        ), self.assertRaises(SystemExit):
            overlay.parse_args()

    def test_analyze_only_discovers_existing_render_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stem = "take_mocap_video_aligned_strict25"
            mapping = root / f"{stem}.alignment.csv"
            mapping.write_bytes(b"mapping")
            (root / f"{stem}.mp4").write_bytes(b"mp4v")
            (root / f"{stem}.contact.jpg").write_bytes(b"jpg")
            (root / f"{stem}_h264.mp4").write_bytes(b"h264")
            artifacts = overlay._existing_alignment_artifacts(
                root,
                stem,
                mapping,
            )
            self.assertEqual(
                set(artifacts),
                {
                    "frame_mapping_csv",
                    "mp4v_preview",
                    "contact_sheet",
                    "h264_delivery_video",
                },
            )

    def test_main_analyze_only_calls_existing_artifacts_with_three_arguments(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "output"
            segment = root / "01_210814_Take_000"
            prepared = mock.Mock()
            args = mock.Mock(
                dataset_root=root,
                calibration=root / "calibration.json",
                output_dir=output_dir,
                segment=["01"],
                max_interpolation_gap_ms=25.0,
                mocap_position_source="human-cma",
                mocap_world_x_offset_mm=0.0,
                mocap_world_y_offset_mm=0.0,
                mocap_world_z_offset_mm=0.0,
                analyze_only=True,
                frame_content_samples=31,
                draw_rear_mount_proxy=False,
            )
            mapping_path = output_dir / "mapping.csv"
            with (
                mock.patch.object(overlay, "parse_args", return_value=args),
                mock.patch.object(
                    overlay.viz,
                    "_discover_segment",
                    return_value=segment,
                ),
                mock.patch.object(
                    overlay,
                    "prepare_mocap_video",
                    return_value=prepared,
                ),
                mock.patch.object(
                    overlay,
                    "synchronized_interval",
                    return_value=(0, 1),
                ),
                mock.patch.object(
                    overlay,
                    "write_frame_mapping",
                    return_value=mapping_path,
                ),
                mock.patch.object(
                    overlay,
                    "_existing_alignment_artifacts",
                    autospec=True,
                    return_value={},
                ) as existing,
                mock.patch.object(
                    overlay,
                    "alignment_metrics",
                    return_value={},
                ),
            ):
                overlay.main()

            stem = overlay.alignment_output_stem(
                segment.name,
                25.0,
                "human-cma",
            )
            existing.assert_called_once_with(output_dir.resolve(), stem, mapping_path)

    def test_content_comparator_accepts_same_index_and_rejects_off_by_one(self) -> None:
        correct = {
            index: _bag_sample(index, self.frames[index]) for index in self.indices
        }
        result = overlay._compare_sampled_frame_content(
            correct,
            self.mp4,
            self.indices,
            total_frame_count=len(self.frames),
        )
        self.assertTrue(result["acceptance"]["pass"])
        self.assertEqual(result["best_global_offset_frames"], 0)

        full_bag = {
            index: _bag_sample(index, frame)
            for index, frame in enumerate(self.frames)
        }
        full_result = overlay._compare_sampled_frame_content(
            full_bag,
            self.mp4,
            list(range(len(self.frames))),
            total_frame_count=len(self.frames),
            requested_sample_count=0,
        )
        self.assertTrue(full_result["acceptance"]["pass"])
        self.assertTrue(full_result["full_frame_content_comparison"])
        self.assertEqual(full_result["sample_count"], len(self.frames))

        shifted = {
            index: _bag_sample(index, self.frames[min(index + 1, 39)])
            for index in self.indices
        }
        shifted_result = overlay._compare_sampled_frame_content(
            shifted,
            self.mp4,
            self.indices,
            total_frame_count=len(self.frames),
        )
        self.assertFalse(shifted_result["acceptance"]["pass"])
        self.assertEqual(shifted_result["best_global_offset_frames"], 1)

    def test_content_comparator_rejects_reordering_and_unordered_indices(self) -> None:
        reversed_frames = list(reversed(self.frames))
        reordered = {
            index: _bag_sample(index, reversed_frames[index])
            for index in self.indices
        }
        result = overlay._compare_sampled_frame_content(
            reordered,
            self.mp4,
            self.indices,
            total_frame_count=len(self.frames),
        )
        self.assertFalse(result["acceptance"]["pass"])
        with self.assertRaisesRegex(ValueError, "unique, and ordered"):
            overlay._compare_sampled_frame_content(
                reordered,
                self.mp4,
                list(reversed(self.indices)),
                total_frame_count=len(self.frames),
            )

    def test_timecode_match_rejects_duplicates_order_and_non_unique_indices(self) -> None:
        bag = [100, 200, 300, 400]
        valid = overlay._timecode_bag_index_match([100, 300, 400], bag)
        self.assertTrue(valid["pass"])
        self.assertEqual(valid["matched_bag_message_indices"], [0, 2, 3])
        self.assertFalse(
            overlay._timecode_bag_index_match([100, 300, 300], bag)["pass"]
        )
        self.assertFalse(
            overlay._timecode_bag_index_match([100, 400, 300], bag)["pass"]
        )
        duplicate_bag = overlay._timecode_bag_index_match(
            [100, 300], [100, 200, 300, 300]
        )
        self.assertFalse(duplicate_bag["pass"])
        self.assertFalse(
            duplicate_bag[
                "each_timecode_row_matches_exactly_one_unique_bag_index"
            ]
        )


class MocapVideoDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prepared = {
            key: overlay.prepare_mocap_video(
                viz._discover_segment(DATASET, key), CALIBRATION
            )
            for key in ("01", "02", "03")
        }
        segment_02 = viz._discover_segment(DATASET, "02")
        cls.prepared_02_explicit_zero = overlay.prepare_mocap_video(
            segment_02,
            CALIBRATION,
            mocap_world_translation_mm=(0.0, 0.0, 0.0),
        )
        cls.prepared_02_z37 = overlay.prepare_mocap_video(
            segment_02,
            CALIBRATION,
            mocap_world_translation_mm=(0.0, 0.0, 37.0),
        )
        cls.prepared_02_xyz = overlay.prepare_mocap_video(
            segment_02,
            CALIBRATION,
            mocap_world_translation_mm=(12.0, -4.5, 37.0),
        )
        cls.prepared_bvh = {
            key: overlay.prepare_mocap_video(
                viz._discover_segment(DATASET, key),
                CALIBRATION,
                mocap_position_source=(
                    overlay.MOCAP_POSITION_SOURCE_SKELETON_BVH
                ),
            )
            for key in ("01", "02", "03")
        }
        cls.prepared_02_bvh = cls.prepared_bvh["02"]

    def test_zero_world_translation_is_backward_compatible(self) -> None:
        implicit = self.prepared["02"]
        explicit = self.prepared_02_explicit_zero
        self.assertEqual(implicit.mocap_world_translation_mm, (0.0, 0.0, 0.0))
        self.assertEqual(explicit.mocap_world_translation_mm, (0.0, 0.0, 0.0))
        np.testing.assert_array_equal(implicit.mocap_mm, explicit.mocap_mm)
        np.testing.assert_array_equal(implicit.mocap_valid, explicit.mocap_valid)
        np.testing.assert_array_equal(implicit.cmavatar_s, explicit.cmavatar_s)
        self.assertIsNone(
            overlay._mocap_world_translation_header(
                implicit.mocap_world_translation_mm
            )
        )

    def test_skeleton_bvh_fk_uses_cma_ordinals_and_cma_timebase(self) -> None:
        human = self.prepared["02"]
        bvh = self.prepared_02_bvh
        self.assertEqual(
            bvh.mocap_position_source,
            overlay.MOCAP_POSITION_SOURCE_SKELETON_BVH,
        )
        self.assertEqual(set(bvh.mocap_position_paths), {"left", "right"})
        self.assertTrue(
            bvh.mocap_position_paths["left"].name.endswith("Skeleton_0.bvh")
        )
        self.assertTrue(
            bvh.mocap_position_paths["right"].name.endswith("Skeleton_1.bvh")
        )
        np.testing.assert_array_equal(bvh.mocap.times_s, human.mocap.times_s)
        np.testing.assert_array_equal(
            bvh.mocap.frame_counters,
            human.mocap.frame_counters,
        )
        np.testing.assert_array_equal(
            bvh.mocap.display_times_s,
            human.mocap.display_times_s,
        )
        self.assertTrue(np.all(np.isnan(bvh.mocap.points_mm[0])))
        self.assertTrue(np.all(np.isfinite(bvh.mocap.points_mm[1:])))

        contract = bvh.mocap_position_contract
        self.assertEqual(contract["status"], "pass")
        self.assertEqual(
            contract["bvh_forward_kinematics"],
            "bvh_web_export.evaluate_world_positions",
        )
        self.assertEqual(
            contract["cma_world_mm_to_bvh_units"],
            "[-X, Z, Y] / 10",
        )
        self.assertEqual(
            contract["bvh_units_to_mocap_world_mm"],
            overlay.BVH_TO_MOCAP_WORLD_CONTRACT,
        )
        self.assertTrue(
            contract["human_cma_timestamps_preserved_without_resampling"]
        )
        self.assertFalse(
            contract["human_cma_joint_positions_used_as_render_source"]
        )
        self.assertTrue(
            contract["human_cma_root_positions_used_for_ordinal_validation_only"]
        )
        self.assertLessEqual(
            contract["all_usable_root_ordinals_max_abs_error_mm"],
            overlay.BVH_CMA_ROOT_TOLERANCE_MM,
        )
        self.assertEqual(
            contract["cma_row_count"],
            len(human.mocap.frame_counters),
        )
        self.assertEqual(
            contract["bvh_source_frame_count"],
            len(human.mocap.frame_counters) + 1,
        )
        # FK is intentionally the detailed position source, not an alias of
        # the Human.cma joint-position columns.
        finger_delta_mm = np.linalg.norm(
            bvh.mocap.points_mm[1:, :, 1:]
            - human.mocap.points_mm[1:, :, 1:],
            axis=-1,
        )
        self.assertGreater(float(np.median(finger_delta_mm)), 0.05)
        self.assertEqual(
            overlay.synchronized_interval(bvh.mocap_valid),
            overlay.synchronized_interval(human.mocap_valid),
        )

    def test_every_formal_take_has_a_strict_skeleton_bvh_interval(self) -> None:
        expected = {
            # The first otherwise-synchronized RGB frame in Take_000/002
            # brackets CMA ordinal zero.  That ordinal is the BVH vendor seed,
            # so skeleton-bvh correctly trims one additional RGB frame.
            "01": (61, 1808, 1747),
            "02": (0, 1805, 1805),
            "03": (4, 1803, 1799),
        }
        for key, (first, stop, count) in expected.items():
            with self.subTest(take=key):
                prepared = self.prepared_bvh[key]
                actual = overlay.synchronized_interval(prepared.mocap_valid)
                self.assertEqual(actual, (first, stop))
                self.assertEqual(stop - first, count)
                self.assertEqual(
                    prepared.mocap_position_contract["status"],
                    "pass",
                )
                self.assertTrue(
                    prepared.mocap_position_contract[
                        "human_cma_timestamps_preserved_without_resampling"
                    ]
                )

    def test_skeleton_bvh_metrics_include_both_hashed_sources_and_contract(
        self,
    ) -> None:
        prepared = self.prepared_02_bvh
        first, stop = overlay.synchronized_interval(prepared.mocap_valid)
        result = overlay.alignment_metrics(
            prepared,
            first,
            stop,
            frame_content_validation={
                "full_frame_content_comparison": False,
                "acceptance": {"pass": True},
            },
            motion_validation={"acceptance": {"pass": True}},
        )
        source = result["mocap_position_source"]
        self.assertEqual(source["selected"], "skeleton-bvh")
        self.assertTrue(source["human_cma_supplies_timestamps_and_frame_counters"])
        self.assertEqual(source["contract"], prepared.mocap_position_contract)
        for key, side in (
            ("mocap_skeleton_0_bvh", "left"),
            ("mocap_skeleton_1_bvh", "right"),
        ):
            item = result["inputs"][key]
            self.assertEqual(
                Path(item["path"]),
                prepared.mocap_position_paths[side].resolve(),
            )
            self.assertEqual(
                item["sha256"],
                overlay._sha256(prepared.mocap_position_paths[side]),
            )

    def test_skeleton_bvh_parser_and_ordinal_failures_propagate(self) -> None:
        segment = viz._discover_segment(DATASET, "02")
        with mock.patch.object(
            overlay.bvh_export,
            "parse_bvh",
            side_effect=overlay.bvh_export.BvhValidationError("malformed BVH"),
        ), self.assertRaisesRegex(
            overlay.bvh_export.BvhValidationError,
            "malformed BVH",
        ):
            overlay.prepare_mocap_video(
                segment,
                CALIBRATION,
                mocap_position_source="skeleton-bvh",
            )

        with mock.patch.object(
            overlay.bvh_export,
            "validate_bvh_cma_root_ordinal_contract",
            side_effect=overlay.bvh_export.BvhValidationError(
                "ordinal mismatch"
            ),
        ), self.assertRaisesRegex(
            overlay.bvh_export.BvhValidationError,
            "ordinal mismatch",
        ):
            overlay.prepare_mocap_video(
                segment,
                CALIBRATION,
                mocap_position_source="skeleton-bvh",
            )

        with self.assertRaisesRegex(ValueError, "Unsupported MOCAP position source"):
            overlay.prepare_mocap_video(
                segment,
                CALIBRATION,
                mocap_position_source="automatic",
            )

    def test_world_translation_preserves_time_and_all_bone_lengths(self) -> None:
        baseline = self.prepared["02"]
        translated = self.prepared_02_xyz
        self.assertEqual(
            translated.mocap_world_translation_mm,
            (12.0, -4.5, 37.0),
        )
        np.testing.assert_array_equal(baseline.rgb_device_s, translated.rgb_device_s)
        np.testing.assert_array_equal(baseline.cmavatar_s, translated.cmavatar_s)
        np.testing.assert_array_equal(baseline.mocap_valid, translated.mocap_valid)
        np.testing.assert_array_equal(
            baseline.interpolation.low_indices,
            translated.interpolation.low_indices,
        )
        np.testing.assert_array_equal(
            baseline.interpolation.high_indices,
            translated.interpolation.high_indices,
        )
        np.testing.assert_array_equal(
            baseline.interpolation.alpha,
            translated.interpolation.alpha,
        )

        valid = baseline.mocap_valid
        expected_translation = np.asarray([12.0, -4.5, 37.0])
        observed_translation = (
            translated.mocap_mm[valid] - baseline.mocap_mm[valid]
        )
        np.testing.assert_allclose(
            observed_translation,
            np.broadcast_to(expected_translation, observed_translation.shape),
            atol=1e-12,
        )
        edges = [
            (start, end)
            for chain in viz.MOCAP_CHAINS
            for start, end in zip(chain[:-1], chain[1:])
        ]
        starts, ends = np.asarray(edges).T
        baseline_valid = baseline.mocap_mm[valid]
        translated_valid = translated.mocap_mm[valid]
        baseline_lengths = np.linalg.norm(
            baseline_valid[:, :, ends, :] - baseline_valid[:, :, starts, :],
            axis=-1,
        )
        translated_lengths = np.linalg.norm(
            translated_valid[:, :, ends, :] - translated_valid[:, :, starts, :],
            axis=-1,
        )
        np.testing.assert_allclose(translated_lengths, baseline_lengths, atol=1e-10)

    def test_take02_output661_z37_wrist_projection_golden(self) -> None:
        prepared = self.prepared_02_z37
        first, _ = overlay.synchronized_interval(prepared.mocap_valid)
        source_frame = first + 661
        projected, positive = viz.project_world_to_rgb(
            prepared.mocap_mm[source_frame], prepared.calibration
        )
        np.testing.assert_array_equal(positive[:, 0], [True, True])
        np.testing.assert_allclose(
            projected[:, 0],
            [[1388.82, 494.82], [695.51, 515.19]],
            atol=0.01,
        )
        self.assertEqual(
            overlay._mocap_world_translation_header(
                prepared.mocap_world_translation_mm
            ),
            "CANDIDATE origin +Z37 mm / not GT",
        )

    def test_exposure_latency_and_frame_complete_intervals(self) -> None:
        expected = {
            "01": (60, 1808, 1748, 255.103),
            "02": (0, 1805, 1805, 255.250),
            "03": (3, 1803, 1800, 254.008),
        }
        for key, (first, stop, count, median_ms) in expected.items():
            with self.subTest(take=key):
                prepared = self.prepared[key]
                actual_first, actual_stop = overlay.synchronized_interval(
                    prepared.mocap_valid
                )
                self.assertEqual((actual_first, actual_stop), (first, stop))
                self.assertEqual(actual_stop - actual_first, count)
                self.assertTrue(np.all(prepared.mocap_valid[first:stop]))
                self.assertAlmostEqual(
                    float(np.median(prepared.clock_mapping.capture_to_poll_s) * 1000.0),
                    median_ms,
                    places=3,
                )
                latency_ms = prepared.clock_mapping.capture_to_poll_s * 1000.0
                self.assertGreater(float(np.min(latency_ms)), 230.0)
                self.assertLess(float(np.max(latency_ms)), 310.0)

    def test_timecode_rows_exactly_match_bag_index_timestamps(self) -> None:
        expected_rows = {"01": 875, "02": 878, "03": 876}
        for key, count in expected_rows.items():
            with self.subTest(take=key):
                result = overlay._camera_timecode_bag_match(self.prepared[key])
                self.assertTrue(result["pass"])
                self.assertEqual(result["timecode_rows_inside_bag_interval"], count)
                self.assertEqual(result["exact_bag_index_timestamp_matches"], count)
                self.assertEqual(result["exact_match_fraction"], 1.0)
                self.assertEqual(result["timecode_duplicate_timestamp_count"], 0)
                self.assertEqual(result["bag_duplicate_timestamp_count"], 0)
                self.assertTrue(result["timecode_timestamps_strictly_increasing"])
                self.assertTrue(result["bag_timestamps_strictly_increasing"])
                self.assertTrue(
                    result[
                        "each_timecode_row_matches_exactly_one_unique_bag_index"
                    ]
                )

    def test_real_bag_mp4_sampled_content_contract(self) -> None:
        expected_connections = {"01": 4, "02": 2, "03": 4}
        for key, prepared in self.prepared.items():
            with self.subTest(take=key):
                result = overlay.sampled_bag_mp4_content_validation(prepared)
                self.assertTrue(result["acceptance"]["pass"])
                self.assertEqual(result["sample_count"], 31)
                self.assertEqual(
                    (result["sample_indices"][0], result["sample_indices"][-1]),
                    (0, prepared.video.frame_count - 1),
                )
                self.assertEqual(result["best_global_offset_frames"], 0)
                self.assertEqual(
                    result["ros_connection_id"], expected_connections[key]
                )
                self.assertLessEqual(result["same_index_luma_mse"]["max"], 6.0)
                self.assertGreaterEqual(
                    result["distinguishable_sample_count"], 24
                )

    def test_metrics_hashes_sync_report_and_states_claim_boundaries(self) -> None:
        prepared = self.prepared["02"]
        first, stop = overlay.synchronized_interval(prepared.mocap_valid)
        content_qa = {
            "full_frame_content_comparison": False,
            "acceptance": {"pass": True},
        }
        motion_qa = {"acceptance": {"pass": True}}
        result = overlay.alignment_metrics(
            prepared,
            first,
            stop,
            frame_content_validation=content_qa,
            motion_validation=motion_qa,
        )
        report = (
            prepared.segment_dir
            / "同步校验"
            / "common_interval_sync_report.json"
        )
        self.assertEqual(result["schema"], overlay.ALIGNMENT_SCHEMA)
        self.assertEqual(
            result["inputs"]["common_interval_sync_report"]["sha256"],
            overlay._sha256(report),
        )
        contract = result["video_frame_contract"]
        self.assertNotIn(
            "bag_message_i_equals_mp4_frame_i_delivery_contract", contract
        )
        self.assertTrue(
            contract["bag_message_i_equals_mp4_frame_i_content_validated"]
        )
        acceptance = result["temporal_alignment"]["acceptance"]
        self.assertTrue(
            acceptance[
                "frame_complete_is_not_claimed_as_measured_sub_millisecond_gt"
            ]
        )
        rear_mount = result["spatial_alignment"][
            "rear_wrist_mount_visualization"
        ]
        self.assertFalse(rear_mount["enabled"])
        self.assertEqual(
            rear_mount["scale"],
            overlay.VISUALIZATION_REAR_WRIST_MOUNT_SCALE,
        )
        self.assertFalse(rear_mount["original_21_joint_positions_modified"])
        self.assertFalse(rear_mount["additional_mocap_gt_joint"])
        translation = result["spatial_alignment"]["mocap_world_translation"]
        self.assertFalse(translation["enabled"])
        self.assertEqual(translation["profile"], "identity_default")
        self.assertEqual(translation["translation_xyz_mm"], [0.0, 0.0, 0.0])
        self.assertFalse(translation["ground_truth_claimed"])

        candidate = self.prepared_02_xyz
        candidate_first, candidate_stop = overlay.synchronized_interval(
            candidate.mocap_valid
        )
        candidate_result = overlay.alignment_metrics(
            candidate,
            candidate_first,
            candidate_stop,
            frame_content_validation=content_qa,
            motion_validation=motion_qa,
            draw_rear_mount_proxy=True,
        )
        candidate_spatial = candidate_result["spatial_alignment"]
        self.assertTrue(
            candidate_spatial["rear_wrist_mount_visualization"]["enabled"]
        )
        candidate_translation = candidate_spatial["mocap_world_translation"]
        self.assertTrue(candidate_translation["enabled"])
        self.assertEqual(
            candidate_translation["profile"],
            overlay.MOCAP_WORLD_TRANSLATION_CANDIDATE_PROFILE,
        )
        self.assertEqual(
            candidate_translation["translation_xyz_mm"], [12.0, -4.5, 37.0]
        )
        self.assertFalse(candidate_translation["changes_joint_relative_geometry"])
        self.assertFalse(candidate_translation["ground_truth_claimed"])

    def test_every_mapped_frame_has_explicit_cmavatar_bracket(self) -> None:
        for key, prepared in self.prepared.items():
            with self.subTest(take=key):
                first, stop = overlay.synchronized_interval(prepared.mocap_valid)
                rows = overlay.frame_mapping_rows(prepared, first, min(first + 3, stop))
                self.assertEqual(len(rows), 3)
                for output_frame, row in enumerate(rows):
                    self.assertEqual(row["output_frame"], output_frame)
                    self.assertLess(
                        row["mocap_low_frame_counter"],
                        row["mocap_high_frame_counter"],
                    )
                    self.assertGreaterEqual(row["interpolation_alpha"], 0.0)
                    self.assertLessEqual(row["interpolation_alpha"], 1.0)
                    self.assertLess(row["bracket_span_ms"], 25.0)
                    self.assertEqual(row["left_wrist_inside_frame"], 1)
                    self.assertEqual(row["right_wrist_inside_frame"], 1)

    def test_corrected_mocap_frame_renders_at_source_resolution_contract(self) -> None:
        prepared = self.prepared["02"]
        first, _ = overlay.synchronized_interval(prepared.mocap_valid)
        capture = cv2.VideoCapture(str(prepared.video.path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        ok, frame = capture.read()
        capture.release()
        self.assertTrue(ok)
        with mock.patch.object(
            overlay,
            "visualization_rear_wrist_mount_anchor",
            wraps=overlay.visualization_rear_wrist_mount_anchor,
        ) as rear_mount:
            rendered = overlay.render_mocap_frame(
                prepared,
                first,
                0,
                frame,
                output_width=960,
                first_source_frame=first,
            )
            rear_mount.assert_not_called()
            rendered_with_proxy = overlay.render_mocap_frame(
                prepared,
                first,
                0,
                frame,
                output_width=960,
                first_source_frame=first,
                draw_rear_mount_proxy=True,
            )
            rear_mount.assert_called_once()
        self.assertEqual(rendered.shape, (540, 960, 3))
        self.assertFalse(np.array_equal(rendered, cv2.resize(frame, (960, 540))))
        self.assertFalse(np.array_equal(rendered, rendered_with_proxy))

    def test_visual_motion_qa_rejects_poll_time_and_accepts_exposure_time(self) -> None:
        for key, prepared in self.prepared.items():
            with self.subTest(take=key):
                first, stop = overlay.synchronized_interval(prepared.mocap_valid)
                result = overlay.visual_motion_validation(prepared, first, stop)
                self.assertTrue(result["acceptance"]["pass"])
                self.assertGreaterEqual(
                    result["acquisition_time_corrected"][
                        "zero_adjustment_correlation"
                    ],
                    0.80,
                )
                self.assertGreaterEqual(result["zero_adjustment_correlation_gain"], 0.20)
                self.assertLessEqual(
                    abs(
                        result["acquisition_time_corrected"][
                            "best_timestamp_adjustment_ms"
                        ]
                    ),
                    result["one_video_frame_ms"],
                )


if __name__ == "__main__":
    unittest.main()
