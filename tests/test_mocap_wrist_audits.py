from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

import mocap_world_origin_audit as world_audit
import mocap_wrist_depth_audit as depth_audit


ROOT = Path(__file__).resolve().parents[1]
WORLD_OUTPUT = ROOT / "outputs" / "mocap_wrist_semantics_audit"
DEPTH_OUTPUT = ROOT / "outputs" / "mocap_wrist_depth_audit" / "depth_audit.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class WristAuditHelperTests(unittest.TestCase):
    def test_joint_groups_use_left_then_right_side_major_indices(self) -> None:
        groups = depth_audit._joint_group_masks()
        np.testing.assert_array_equal(
            np.flatnonzero(groups["wrists"]),
            [0, 21],
        )
        np.testing.assert_array_equal(
            np.flatnonzero(groups["wrist_plus_four_non_thumb_mcps"]),
            [0, 5, 9, 13, 17, 21, 26, 30, 34, 38],
        )
        np.testing.assert_array_equal(
            np.flatnonzero(groups["fingertips"]),
            [4, 8, 12, 16, 20, 25, 29, 33, 37, 41],
        )
        self.assertEqual(int(np.count_nonzero(groups["all_42_joints"])), 42)
        self.assertEqual(
            int(np.count_nonzero(groups["internal_excluding_fingertips"])),
            32,
        )

    def test_nearest_nonzero_maps_propagate_depth_of_nearest_valid_pixel(self) -> None:
        depth = np.zeros((5, 7), dtype=np.uint16)
        depth[1, 1] = 700
        depth[3, 5] = 1300

        distance, nearest = depth_audit._nearest_nonzero_maps(depth)

        self.assertEqual(float(distance[1, 1]), 0.0)
        self.assertEqual(int(nearest[1, 1]), 700)
        self.assertEqual(int(nearest[1, 2]), 700)
        self.assertEqual(int(nearest[3, 4]), 1300)
        self.assertEqual(int(nearest[3, 5]), 1300)

        empty_distance, empty_nearest = depth_audit._nearest_nonzero_maps(
            np.zeros((3, 4), dtype=np.uint16)
        )
        self.assertTrue(np.all(empty_distance > 5.0))
        np.testing.assert_array_equal(empty_nearest, np.zeros((3, 4), np.uint16))

    def test_point_diagnostic_only_accepts_nonzero_depth_inside_radius(self) -> None:
        depth = np.zeros((5, 5), dtype=np.uint16)
        depth[1, 1] = 100
        distance, nearest = depth_audit._nearest_nonzero_maps(depth)
        rows = depth_audit._point_rows(
            np.asarray([[1.0, 1.0], [2.0, 2.0]]),
            np.asarray([110.0, 220.0]),
            depth,
            distance,
            nearest,
            local_radius_px=0.5,
        )

        self.assertEqual([row["side"] for row in rows], ["left", "right"])
        self.assertTrue(rows[0]["nearest_nonzero_accepted_within_local_radius"])
        self.assertEqual(rows[0]["signed_residual_mm"], 10.0)
        self.assertEqual(rows[1]["nearest_nonzero_depth_mm"], 100)
        self.assertFalse(rows[1]["nearest_nonzero_accepted_within_local_radius"])
        self.assertIsNone(rows[1]["signed_residual_mm"])

        no_depth = np.zeros((5, 5), dtype=np.uint16)
        no_distance, no_nearest = depth_audit._nearest_nonzero_maps(no_depth)
        no_depth_rows = depth_audit._point_rows(
            np.asarray([[1.0, 1.0], [2.0, 2.0]]),
            np.asarray([110.0, 220.0]),
            no_depth,
            no_distance,
            no_nearest,
            local_radius_px=5.0,
        )
        for row in no_depth_rows:
            self.assertIsNone(row["nearest_nonzero_distance_px"])
            self.assertIsNone(row["nearest_nonzero_depth_mm"])
            self.assertFalse(row["nearest_nonzero_accepted_within_local_radius"])
            self.assertIsNone(row["signed_residual_mm"])

    def test_metric_denominators_are_explicit_and_do_not_drop_misses(self) -> None:
        metric = depth_audit.MetricAccumulator()
        metric.add(
            np.ones(4, dtype=bool),
            np.asarray([True, True, True, False]),
            np.asarray([True, True, False, False]),
            np.asarray([True, False, False, False]),
            np.asarray([True, True, False, False]),
            np.asarray([10.0, 60.0, np.nan, np.nan]),
        )

        summary = metric.summary()
        self.assertEqual(summary["requested_projected_point_count"], 4)
        self.assertEqual(summary["positive_camera_depth_count"], 3)
        self.assertEqual(summary["in_frame_count"], 2)
        self.assertEqual(summary["exact_nonzero_depth_count"], 1)
        self.assertEqual(summary["local_nonzero_depth_count"], 2)
        threshold = summary["absolute_residual_thresholds_mm"]["50"]
        self.assertEqual(threshold["count"], 1)
        self.assertEqual(threshold["fraction_of_requested"], 0.25)
        self.assertEqual(threshold["fraction_of_in_frame"], 0.5)
        self.assertEqual(threshold["fraction_of_local_nonzero"], 0.5)

    def test_rear_proxy_is_derived_without_mutating_or_aliasing_joints(self) -> None:
        joints = np.arange(2 * 21 * 3, dtype=np.float64).reshape(2, 21, 3)
        original = joints.copy()

        proxy = world_audit._rear_proxy_points(joints)

        np.testing.assert_array_equal(joints, original)
        np.testing.assert_allclose(
            proxy,
            joints[:, 0]
            - world_audit.REAR_PROXY_MIDDLE_MCP_SCALE
            * (joints[:, world_audit.MIDDLE_MCP_INDEX] - joints[:, 0]),
        )
        self.assertFalse(np.shares_memory(proxy, joints))

    def test_heldout_selector_skips_hypothesis_frame_661_deterministically(self) -> None:
        first = world_audit._selected_output_frames(1575, "02")
        second = world_audit._selected_output_frames(1575, "02")
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(world_audit.SAMPLE_QUANTILES))
        self.assertNotIn(661, first)
        self.assertIn(662, first)
        self.assertEqual(first, sorted(first))

    def test_atomic_json_writers_preserve_previous_file_on_serialization_error(self) -> None:
        for writer in (world_audit._write_json_atomic, depth_audit._write_json_atomic):
            with self.subTest(writer=writer.__module__):
                with tempfile.TemporaryDirectory() as directory:
                    destination = Path(directory) / "audit.json"
                    destination.write_text("previous\n", encoding="utf-8")
                    with self.assertRaises(ValueError):
                        writer(destination, {"invalid": float("nan")})
                    self.assertEqual(
                        destination.read_text(encoding="utf-8"),
                        "previous\n",
                    )
                    self.assertEqual(
                        list(destination.parent.glob(f".{destination.name}.*")),
                        [],
                    )

                    writer(destination, {"ok": 1})
                    self.assertEqual(json.loads(destination.read_text()), {"ok": 1})

    def test_atomic_png_encoding_is_repeatable(self) -> None:
        image = np.arange(9 * 13 * 3, dtype=np.uint8).reshape(9, 13, 3)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.png"
            second = Path(directory) / "second.png"
            world_audit._write_png(first, image)
            world_audit._write_png(second, image)
            self.assertEqual(_sha256(first), _sha256(second))
            np.testing.assert_array_equal(cv2.imread(str(first)), image)
            self.assertFalse(list(Path(directory).glob(".*.png")))


class WristSemanticsArtifactContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary_path = WORLD_OUTPUT / "summary.json"
        cls.provenance_path = WORLD_OUTPUT / "frame_provenance.csv"
        cls.summary = json.loads(cls.summary_path.read_text(encoding="utf-8"))
        with cls.provenance_path.open("r", encoding="utf-8", newline="") as stream:
            cls.provenance = list(csv.DictReader(stream))

    def test_summary_contract_and_geometry_invariants(self) -> None:
        data = self.summary
        self.assertEqual(data["schema"], world_audit.SUMMARY_SCHEMA)
        self.assertEqual(
            data["status"], "automated_render_and_provenance_checks_passed"
        )
        self.assertEqual(data["selection"]["frames_per_take"], 8)
        self.assertEqual(data["selection"]["total_frames"], 24)
        self.assertTrue(data["selection"]["take02_output_661_excluded"])
        self.assertTrue(data["assertions"]["all_passed"])
        self.assertEqual(data["assertions"]["exact_alignment_mapping_matches"], 24)
        self.assertEqual(data["assertions"]["decoded_exact_source_frame_assertions"], 24)
        self.assertLessEqual(
            data["assertions"]["rear_proxy_joint_mutation_max_abs_error_mm"],
            1e-12,
        )
        self.assertLessEqual(
            data["assertions"]["rear_proxy_joint_pixel_max_abs_error_px"],
            1e-12,
        )
        self.assertLessEqual(
            data["assertions"]["rigid_translation_max_abs_error_mm"],
            1e-12,
        )
        self.assertLessEqual(
            data["assertions"]["bone_length_max_abs_error_mm"],
            1e-12,
        )
        proxy = data["views"]["rear_mount_proxy"]
        self.assertEqual(proxy["middle_mcp_index"], 9)
        self.assertEqual(proxy["additional_points"], 2)
        self.assertFalse(proxy["modifies_delivered_21_joint_hands"])
        self.assertIn("not_physical_ground_truth", proxy["status"])
        self.assertIn(
            "not_recommended_global_fix",
            data["views"]["global_z37_counterexample"]["status"],
        )

    def test_all_24_provenance_rows_match_frozen_alignment_rows(self) -> None:
        self.assertEqual(len(self.provenance), 24)
        self.assertEqual(
            {key: sum(row["take_key"] == key for row in self.provenance)
             for key in world_audit.TAKES},
            {"01": 8, "02": 8, "03": 8},
        )
        self.assertFalse(
            any(
                row["take_key"] == "02" and int(row["output_frame"]) == 661
                for row in self.provenance
            )
        )

        take_by_key = {item["take_key"]: item for item in self.summary["takes"]}
        alignment_cache: dict[str, list[dict[str, str]]] = {}
        for take_key, take in take_by_key.items():
            alignment_path = ROOT / take["alignment_csv"]["path"]
            with alignment_path.open("r", encoding="utf-8", newline="") as stream:
                alignment_cache[take_key] = list(csv.DictReader(stream))

        for row in self.provenance:
            original = alignment_cache[row["take_key"]][int(row["output_frame"])]
            for field in world_audit.ALIGNMENT_FIELDS:
                self.assertEqual(row[field], original[field])
            self.assertEqual(
                row["alignment_row_sha256"],
                world_audit._canonical_row_sha256(original),
            )
            self.assertEqual(row["rear_proxy_modifies_delivered_joints"], "false")
            self.assertEqual(
                row["z37_is_2d_counterexample_not_recommended_global_fix"],
                "true",
            )

    def test_png_contact_sheets_and_frame_artifacts_are_present(self) -> None:
        for take in self.summary["takes"]:
            self.assertEqual(len(take["selected_output_frames"]), 8)
            self.assertEqual(len(take["frames"]), 8)
            self.assertNotIn(661, take["selected_output_frames"] if take["take_key"] == "02" else [])
            contact = WORLD_OUTPUT / take["contact_sheet"]
            self.assertEqual(_sha256(contact), take["contact_sheet_sha256"])
            contact_image = cv2.imread(str(contact))
            self.assertIsNotNone(contact_image)
            self.assertEqual(contact_image.shape, (1312, 2880, 3))

            for frame in take["frames"]:
                pair = WORLD_OUTPUT / frame["pair_image"]
                self.assertTrue(pair.is_file(), pair)
            sample_pair = cv2.imread(str(WORLD_OUTPUT / take["frames"][0]["pair_image"]))
            self.assertIsNotNone(sample_pair)
            self.assertEqual(sample_pair.shape, (598, 3660, 3))


class WristDepthArtifactContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = json.loads(DEPTH_OUTPUT.read_text(encoding="utf-8"))

    def test_scan_shape_group_indices_and_requested_denominators(self) -> None:
        data = self.data
        self.assertEqual(data["schema"], depth_audit.SCHEMA)
        self.assertEqual(data["status"], "completed_candidate_scan_not_ground_truth")
        self.assertEqual(data["scan"]["segments"], ["01", "02", "03"])
        self.assertEqual(data["scan"]["uniform_samples_per_take"], 301)
        self.assertEqual(data["scan"]["total_uniform_sampled_depth_frames"], 903)
        self.assertEqual(data["scan"]["flattening"], "left 21 joints followed by right 21 joints")
        self.assertEqual(data["scan"]["groups"]["wrists"], [0, 21])
        self.assertEqual(
            data["scan"]["groups"]["wrist_plus_four_non_thumb_mcps"],
            [0, 5, 9, 13, 17, 21, 26, 30, 34, 38],
        )

        expected_requested = {
            "all_42_joints": 42 * 903,
            "wrists": 2 * 903,
            "wrist_plus_four_non_thumb_mcps": 10 * 903,
            "internal_excluding_fingertips": 32 * 903,
            "fingertips": 10 * 903,
        }
        for offset, payload in data["aggregate"]["offsets"].items():
            self.assertEqual(
                payload["translation_world_xyz_mm"],
                [0.0, 0.0, float(offset)],
            )
            for group, expected in expected_requested.items():
                self.assertEqual(
                    payload["groups"][group]["requested_projected_point_count"],
                    expected,
                )

    def test_aggregate_counts_equal_sums_of_three_take_counts(self) -> None:
        count_fields = (
            "requested_projected_point_count",
            "positive_camera_depth_count",
            "in_frame_count",
            "exact_nonzero_depth_count",
            "local_nonzero_depth_count",
        )
        offsets = self.data["aggregate"]["offsets"]
        for offset, aggregate in offsets.items():
            for group, summary in aggregate["groups"].items():
                for field in count_fields:
                    self.assertEqual(
                        summary[field],
                        sum(
                            take["offsets"][offset]["groups"][group][field]
                            for take in self.data["takes"].values()
                        ),
                    )

        for take in self.data["takes"].values():
            self.assertEqual(take["sample_count"], 301)
            self.assertEqual(len(take["sample_source_frame_indices"]), 301)
            self.assertEqual(
                take["offsets"]["0"]["groups"]["wrists"]
                ["requested_projected_point_count"],
                602,
            )

    def test_key_aggregate_values_and_candidate_decision_are_locked(self) -> None:
        offsets = self.data["aggregate"]["offsets"]
        wrist_zero = offsets["0"]["groups"]["wrists"]
        wrist_37 = offsets["37"]["groups"]["wrists"]
        palm_zero = offsets["0"]["groups"]["wrist_plus_four_non_thumb_mcps"]
        palm_37 = offsets["37"]["groups"]["wrist_plus_four_non_thumb_mcps"]

        self.assertEqual(wrist_zero["local_nonzero_depth_count"], 1718)
        self.assertAlmostEqual(wrist_zero["absolute_residual_mm"]["median"], 12.944725442569847)
        self.assertEqual(
            wrist_zero["absolute_residual_thresholds_mm"]["50"]["count"],
            1563,
        )
        self.assertEqual(wrist_37["local_nonzero_depth_count"], 1790)
        self.assertAlmostEqual(wrist_37["absolute_residual_mm"]["median"], 66.99632811361778)
        self.assertEqual(
            wrist_37["absolute_residual_thresholds_mm"]["50"]["count"],
            29,
        )
        self.assertAlmostEqual(palm_zero["absolute_residual_mm"]["median"], 24.584741107992727)
        self.assertAlmostEqual(palm_37["absolute_residual_mm"]["median"], 64.03612549849163)

        wrist_rank = self.data["aggregate"]["candidate_rankings"]["wrists"]
        self.assertEqual(
            wrist_rank["minimum_local_valid_median_absolute_residual"]
            ["offset_world_z_mm"],
            5.0,
        )
        self.assertEqual(
            wrist_rank["maximum_fraction_of_requested_with_absolute_residual_at_most_50mm"]
            ["offset_world_z_mm"],
            -10.0,
        )

        decision = self.data["decision"]
        self.assertFalse(decision["activate_global_world_z_plus_37_as_ground_truth"])
        self.assertFalse(decision["activate_global_world_z_plus_37_as_default_overlay"])
        self.assertEqual(
            decision["recommended_original_joint_translation_world_xyz_mm"],
            [0.0, 0.0, 0.0],
        )
        self.assertEqual(decision["rear_mount_proxy_must_be_labeled"], "VIZ-only, not GT")
        proxy = self.data["aggregate"]["rear_mount_proxy"]
        self.assertFalse(proxy["original_21_joint_positions_modified"])
        self.assertFalse(proxy["additional_measured_mocap_joint"])

    def test_output661_is_explicit_diagnostic_not_heldout_provenance(self) -> None:
        diagnostic = self.data["take02_output661"]
        self.assertEqual(diagnostic["segment"], "02_210955_Take_001")
        self.assertEqual(diagnostic["output_frame"], 661)
        self.assertEqual(diagnostic["source_rgb_frame"], 661)
        self.assertIn("same-index pairing", diagnostic["note"])
        self.assertIn("not an anatomical joint centre", self.data["limits"][0])
        self.assertTrue(
            any("not an independent" in limit for limit in self.data["limits"])
        )


if __name__ == "__main__":
    unittest.main()
