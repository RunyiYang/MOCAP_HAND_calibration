from __future__ import annotations

import csv
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
    HAND_ROOT_CONDITIONING,
    PERSISTENT_MARKERS,
    VIEWER_LEFT_MARKERS,
    VIEWER_RIGHT_MARKERS,
    condition_solved_pose,
    derive_frame_clock,
)


ROOT = Path(__file__).resolve().parents[1]
TAKE006 = (
    ROOT
    / "thor_new4_20260831_processed"
    / "camera_glove_recording_20260831_161610"
)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class NewCaptureContractTests(unittest.TestCase):
    def test_take006_sparse_clock_closes_on_depth_endpoint(self) -> None:
        alignment = rows(
            TAKE006
            / "mocap"
            / "Take_006"
            / "alignment"
            / "camera_cmavatar_alignment.csv"
        )
        summary = rows(
            TAKE006
            / "glove_processing"
            / "aligned"
            / "primary"
            / "aligned_frame_summary.csv"
        )
        clock = derive_frame_clock(
            alignment,
            summary,
            rgb_frame_count=2620,
            depth_frame_count=2619,
        )
        self.assertEqual(clock.anchor_indices[0], 0)
        self.assertEqual(clock.anchor_indices[-1], 2618)
        self.assertEqual(len(clock.output_indices), 2619)
        self.assertEqual(clock.step_counts, {1: 53, 2: 1146, 3: 91})
        self.assertAlmostEqual(clock.cadence_us, 33428.5, places=6)
        self.assertTrue(np.all(np.diff(clock.target_cmm_counter) > 0.0))

    def test_persistent_marker_clusters_are_disjoint_and_complete(self) -> None:
        self.assertEqual(len(PERSISTENT_MARKERS), 22)
        self.assertEqual(len(set(PERSISTENT_MARKERS)), 22)
        self.assertEqual(set(VIEWER_LEFT_MARKERS) & set(VIEWER_RIGHT_MARKERS), set())
        for config in HAND_ROOT_CONDITIONING.values():
            self.assertIn(config["root_marker"], PERSISTENT_MARKERS)
            self.assertEqual(len(set(config["palm_markers"])), 4)
            self.assertNotIn(config["root_marker"], config["palm_markers"])

    def test_conditioned_pose_copies_mocap_root_translation(self) -> None:
        markers = np.zeros((22, 3), dtype=np.float64)
        index = {name: value for value, name in enumerate(PERSISTENT_MARKERS)}
        local = np.zeros((20, 3), dtype=np.float64)
        local[[4, 8, 12, 16]] = np.asarray(
            [[100, 25, 0], [100, 8, 0], [100, -8, 0], [90, -25, 0]],
            dtype=np.float64,
        )
        for side, config in HAND_ROOT_CONDITIONING.items():
            root = np.asarray([20.0, -30.0, 700.0])
            markers[index[config["root_marker"]]] = root
            for marker, point in zip(
                config["palm_markers"], local[[4, 8, 12, 16]], strict=True
            ):
                markers[index[marker]] = root + point * float(config["scale"])
            world = condition_solved_pose(local, markers, side)
            np.testing.assert_allclose(world[0], root, atol=1e-12)
            self.assertTrue(np.all(np.isfinite(world)))

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
