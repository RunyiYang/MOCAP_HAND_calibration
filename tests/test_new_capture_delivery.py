from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gt_calib_delivery.delivery import (
    _delivery_video_specs,
    inspect_inputs,
    normalize_delivery_provenance,
    publish_web_delivery,
)
from gt_calib_delivery.new_capture import (
    ACTION_RECORDING,
    ACTION_TAKE,
    DEFAULT_REAR_OFFSET_MM,
    MARKER_TRACKS,
    condition_solved_pose,
    derive_frame_clock,
    load_manual_calibration,
    raw_mocap_nodes,
    load_camera_model,
    virtual_wrist,
    verify_calibration_reference_rgb,
)


ROOT = Path(__file__).resolve().parents[1]
TAKE007 = (
    ROOT
    / "thor_new4_20260831_processed"
    / ACTION_RECORDING
)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class NewCaptureContractTests(unittest.TestCase):
    def test_no_glove_calibration_reference_is_the_exact_155410_frame(self) -> None:
        archive = ROOT / "movementcap_20260831_worldcalib_tabletop_final.tar.gz"
        no_glove = (
            ROOT
            / "thor_new4_20260831_processed"
            / "camera_glove_recording_20260831_155410"
            / "rgbd_unpack"
        )
        camera = load_camera_model(
            archive,
            no_glove / "camera_1_intrinsics.json",
            expected_recording="camera_glove_recording_20260831_155410",
        )
        result = verify_calibration_reference_rgb(
            archive, no_glove / "RGB.mp4", camera
        )
        self.assertEqual(result["status"], "pixel_identical")
        self.assertEqual(result["reference_rgb_frame_index"], 30)

    def test_take007_sparse_clock_excludes_unanchored_tail(self) -> None:
        alignment = rows(
            TAKE007
            / "mocap"
            / ACTION_TAKE
            / "alignment"
            / "camera_cmavatar_alignment.csv"
        )
        summary = rows(
            TAKE007
            / "glove_processing"
            / "aligned"
            / "primary"
            / "aligned_frame_summary.csv"
        )
        clock = derive_frame_clock(
            alignment,
            summary,
            rgb_frame_count=1982,
            depth_frame_count=1982,
        )
        self.assertEqual(clock.anchor_indices[0], 0)
        self.assertEqual(clock.anchor_indices[-1], 1980)
        self.assertEqual(len(clock.output_indices), 1981)
        self.assertEqual(clock.step_counts, {1: 40, 2: 878, 3: 60, 4: 1})
        self.assertAlmostEqual(clock.cadence_us, 33427.0, places=6)
        self.assertTrue(np.all(np.diff(clock.target_cmm_counter) > 0.0))

    def test_take007_photograph_marker_mapping_and_thumb_reid(self) -> None:
        self.assertEqual(len(MARKER_TRACKS["left"]), 11)
        self.assertEqual(len(MARKER_TRACKS["right"]), 11)
        self.assertEqual(MARKER_TRACKS["left"][0], ("11781", "12503"))
        self.assertEqual(MARKER_TRACKS["left"][10], ("12064",))
        self.assertEqual(MARKER_TRACKS["right"][10], ("12433",))
        all_ids = [
            marker
            for side in ("left", "right")
            for track in MARKER_TRACKS[side]
            for marker in track
        ]
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_virtual_wrist_is_twenty_mm_behind_dorsum_marker(self) -> None:
        markers = np.zeros((11, 3), dtype=np.float64)
        markers[[3, 5, 7, 9]] = np.asarray(
            [[120, 30, 0], [120, 10, 5], [120, -10, 5], [120, -30, 0]],
            dtype=np.float64,
        )
        markers[10] = np.asarray([20.0, 0.0, 2.5])
        root = virtual_wrist(markers)
        np.testing.assert_allclose(root, [0.0, 0.0, 2.5], atol=1e-12)

        nodes = raw_mocap_nodes(markers)
        self.assertEqual(nodes.shape, (11, 3))
        np.testing.assert_allclose(nodes[0], root)
        np.testing.assert_allclose(nodes[1:], markers[:10])

    def test_conditioned_pose_copies_corrected_virtual_root(self) -> None:
        markers = np.zeros((11, 3), dtype=np.float64)
        local = np.zeros((20, 3), dtype=np.float64)
        local[[4, 8, 12, 16]] = np.asarray(
            [[120, 30, 0], [120, 10, 5], [120, -10, 5], [120, -30, 0]],
            dtype=np.float64,
        )
        markers[[3, 5, 7, 9]] = local[[4, 8, 12, 16]]
        markers[10] = np.asarray([20.0, 0.0, 2.5])
        root = virtual_wrist(markers, rear_offset_mm=DEFAULT_REAR_OFFSET_MM)
        world = condition_solved_pose(local, markers, "left", scale=1.0)
        np.testing.assert_allclose(world[0], root, atol=1e-12)
        self.assertTrue(np.all(np.isfinite(world)))

    def test_browser_manual_xyz_profile_is_directly_consumable(self) -> None:
        payload = {
            "schema": "gt_calib.manual_xyz_profile.v1",
            "source_recording": ACTION_RECORDING,
            "source_take": ACTION_TAKE,
            "coordinate_system": "mocap_world_mm",
            "units": "mm",
            "global_world_xyz_mm": [12.5, -3.0, 4.0],
            "left_world_xyz_mm": [1.0, 2.0, 3.0],
            "right_world_xyz_mm": [-1.0, -2.0, -3.0],
            "rear_offset_mm": 20.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "take007_manual_xyz_calibration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            profile = load_manual_calibration(path)
        np.testing.assert_allclose(profile.global_world_xyz_mm, [12.5, -3.0, 4.0])
        np.testing.assert_allclose(profile.side_world_xyz_mm["left"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(profile.side_world_xyz_mm["right"], [-1.0, -2.0, -3.0])
        self.assertEqual(profile.rear_offset_mm, 20.0)

    def test_manual_xyz_profile_fails_closed_on_missing_or_wrong_frame(self) -> None:
        base = {
            "schema": "gt_calib.manual_xyz_profile.v1",
            "source_recording": ACTION_RECORDING,
            "source_take": ACTION_TAKE,
            "coordinate_system": "mocap_world_mm",
            "units": "mm",
            "global_world_xyz_mm": [0.0, 0.0, 0.0],
            "left_world_xyz_mm": [0.0, 0.0, 0.0],
            "right_world_xyz_mm": [0.0, 0.0, 0.0],
            "rear_offset_mm": 20.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            missing = dict(base)
            del missing["left_world_xyz_mm"]
            path.write_text(json.dumps(missing), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required fields"):
                load_manual_calibration(path)

            wrong_frame = dict(base, coordinate_system="camera_mm")
            path.write_text(json.dumps(wrong_frame), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mocap_world_mm"):
                load_manual_calibration(path)

            wrong_rear_offset = dict(base, rear_offset_mm=10.0)
            path.write_text(json.dumps(wrong_rear_offset), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fixed 20 mm"):
                load_manual_calibration(path)

    def test_final_inventory_is_exactly_nine_with_four_pairs(self) -> None:
        specs = _delivery_video_specs()
        self.assertEqual(len(specs), 9)
        self.assertEqual([item["order"] for item in specs], list(range(1, 10)))
        self.assertEqual(len({item["filename"] for item in specs}), 9)
        for first, second in ((0, 1), (2, 3), (4, 5), (6, 7)):
            self.assertEqual(specs[first]["group"], specs[second]["group"])
            self.assertEqual(
                specs[first]["expected_frames"], specs[second]["expected_frames"]
            )

    def test_delivery_preflight_passes(self) -> None:
        result = inspect_inputs(ROOT)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["missing"], [])

    def test_web_publication_is_atomic_and_refuses_build_scratch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "delivery"
            source.mkdir()
            (source / "manifest.json").write_text("{}\n", encoding="utf-8")
            destination = root / "public" / "downloads" / "final-nine"
            valid = {"status": "pass", "failures": []}
            with patch(
                "gt_calib_delivery.delivery.validate_delivery", return_value=valid
            ):
                published = publish_web_delivery(source, destination)
                self.assertEqual(published, destination.resolve())
                self.assertEqual(
                    (published / "manifest.json").read_text(encoding="utf-8"),
                    "{}\n",
                )
                with self.assertRaises(FileExistsError):
                    publish_web_delivery(source, destination)

            scratch_source = root / "scratch-delivery"
            (scratch_source / "_render_scratch").mkdir(parents=True)
            with patch(
                "gt_calib_delivery.delivery.validate_delivery", return_value=valid
            ):
                with self.assertRaisesRegex(ValueError, "build scratch"):
                    publish_web_delivery(scratch_source, root / "blocked")

    def test_delivery_provenance_paths_are_portable(self) -> None:
        metrics = ROOT / "final_9_video_delivery" / "metrics"
        for path in sorted(metrics.glob("*.json")):
            self.assertNotIn(
                str(ROOT), path.read_text(encoding="utf-8"), path.name
            )
        profile = (
            ROOT
            / "outputs"
            / "mocap_root_fusion_review"
            / "01_210814_Take_000_mocap_root_fusion_registration_profile.json"
        )
        self.assertNotIn(str(ROOT), profile.read_text(encoding="utf-8"))

    def test_normalize_delivery_provenance_only_rewrites_project_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "delivery" / "metrics"
            metrics.mkdir(parents=True)
            path = metrics / "sample.json"
            path.write_text(
                '{"inside":"'
                + str(root / "data" / "take.json")
                + '","uri":"https://example.test/evidence"}\n',
                encoding="utf-8",
            )
            changed = normalize_delivery_provenance(root, root / "delivery")
            self.assertEqual(changed, [path.resolve()])
            payload = __import__("json").loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["inside"], "project://data/take.json")
            self.assertEqual(payload["uri"], "https://example.test/evidence")


if __name__ == "__main__":
    unittest.main()
