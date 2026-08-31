from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from gt_calib_delivery.delivery import (
    OLD_TAKES,
    _delivery_video_specs,
    _render_old_mocap,
    _render_old_solved,
    _write_readme,
    inspect_inputs,
    normalize_delivery_provenance,
    publish_web_delivery,
    refresh_auxiliary_hashes,
)
from gt_calib_delivery.manual_profiles import (
    default_final_nine_profile,
    load_final_nine_profile,
)
from gt_calib_delivery.new_capture import (
    ACTION_RECORDING,
    ACTION_TAKE,
    DEFAULT_REAR_OFFSET_MM,
    MARKER_TRACKS,
    capture_source_assets,
    condition_solved_pose,
    derive_frame_clock,
    load_manual_calibration,
    raw_mocap_nodes,
    load_camera_model,
    sha256_file,
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
    def test_delivery_readme_describes_the_actual_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            _write_readme(destination, default_final_nine_profile())
            self.assertIn(
                "all operator XYZ translations are zero",
                (destination / "README.md").read_text(),
            )
            profile = load_final_nine_profile(
                ROOT
                / "calibration_profiles"
                / "operator_y_minus_44_all_hand_overlays.v1.json"
            )
            _write_readme(destination, profile)
            content = (destination / "README.md").read_text()
        self.assertIn("global XYZ [0.0, -44.0, 0.0] mm", content)
        self.assertIn("no-glove-calibration", content)

    def test_take007_source_assets_bind_cmm_and_alignment_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name in ("rgb", "cmm", "alignment", "summary", "intrinsics"):
                path = root / name
                path.write_text(name + "\n", encoding="utf-8")
                paths[name] = path
            capture = SimpleNamespace(
                rgb_path=paths["rgb"],
                cmm_path=paths["cmm"],
                alignment_path=paths["alignment"],
                frame_summary_path=paths["summary"],
                camera=SimpleNamespace(intrinsics_path=paths["intrinsics"]),
            )
            assets = capture_source_assets(capture)
        self.assertEqual(
            set(assets),
            {
                "rgb_video",
                "cmm_markers",
                "camera_cmavatar_alignment",
                "glove_aligned_frame_summary",
                "camera_intrinsics",
            },
        )
        self.assertEqual(
            assets["cmm_markers"]["sha256"],
            __import__("hashlib").sha256(b"cmm\n").hexdigest(),
        )

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

    def test_final_nine_profile_dispatches_distinct_take007_entries(self) -> None:
        video_ids = [item["id"] for item in _delivery_video_specs()]
        filenames = [item["filename"] for item in _delivery_video_specs()]
        payload = {
            "schema": "gt_calib.final_nine_manual_xyz.v1",
            "axis_order": ["x", "y", "z"],
            "units": "mm",
            "coordinate_frame": "per_video_mocap_world",
            "global_world_xyz_mm": [0.0, -44.0, 0.0],
            "video_annotations": [
                {
                    "order": order,
                    "video_id": video_id,
                    "filename": filename,
                    "video_sha256": f"{order:064x}",
                    "apply_translation": order <= 8,
                    "per_video_world_xyz_mm": (
                        [2.0, 0.0, 0.0] if order == 8 else [0.0, 0.0, 0.0]
                    ),
                    "left_residual_world_xyz_mm": (
                        [1.0, 0.0, 0.0] if order == 7 else [0.0, 0.0, 0.0]
                    ),
                    "right_residual_world_xyz_mm": [0.0, 0.0, 0.0],
                }
                for order, (video_id, filename) in enumerate(
                    zip(video_ids, filenames, strict=True), start=1
                )
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            raw = load_manual_calibration(
                path, video_id="take007-mocap-markers"
            )
            solved = load_manual_calibration(path, video_id="take007-solved")
        np.testing.assert_array_equal(raw.global_world_xyz_mm, [0.0, -44.0, 0.0])
        np.testing.assert_array_equal(raw.side_world_xyz_mm["left"], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(solved.global_world_xyz_mm, [2.0, -44.0, 0.0])
        np.testing.assert_array_equal(solved.side_world_xyz_mm["left"], [0.0, 0.0, 0.0])

    def test_old_delivery_renderers_receive_latest_bvh_and_operator_xyz(self) -> None:
        take = OLD_TAKES[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "delivery.staging"

            def fake_mocap_run(command: list[str]) -> None:
                output = Path(command[command.index("--output-dir") + 1])
                output.mkdir(parents=True, exist_ok=True)
                stem = f"{take.segment_name}_mocap_video_aligned_strict25_skeleton_bvh"
                for suffix in ("_h264.mp4", ".contact.jpg", ".metrics.json", ".alignment.csv"):
                    (output / f"{stem}{suffix}").write_bytes(b"test")

            with patch("gt_calib_delivery.delivery._run", side_effect=fake_mocap_run) as run:
                outputs = _render_old_mocap(
                    root, take, staging, np.asarray([12.0, -44.0, 3.5])
                )
            self.assertTrue(all(path.is_file() for path in outputs))
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--mocap-position-source") + 1], "skeleton-bvh")
            self.assertEqual(command[command.index("--frame-content-samples") + 1], "0")
            self.assertEqual(command[command.index("--mocap-world-x-offset-mm") + 1], "12.0")
            self.assertEqual(command[command.index("--mocap-world-y-offset-mm") + 1], "-44.0")
            self.assertEqual(command[command.index("--mocap-world-z-offset-mm") + 1], "3.5")

            def fake_solved_run(command: list[str]) -> None:
                output = Path(command[command.index("--output-dir") + 1])
                output.mkdir(parents=True, exist_ok=True)
                stem = f"{take.segment_name}_solved_hand_pose_{take.solved_suffix}"
                for suffix in (".mp4", ".contact.jpg", ".metrics.json"):
                    (output / f"{stem}{suffix}").write_bytes(b"test")

            with patch("gt_calib_delivery.delivery._run", side_effect=fake_solved_run) as run:
                outputs = _render_old_solved(
                    root, take, staging, np.asarray([12.0, -44.0, 3.5])
                )
            self.assertTrue(all(path.is_file() for path in outputs))
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--operator-world-x-mm") + 1], "12.0")
            self.assertEqual(command[command.index("--operator-world-y-mm") + 1], "-44.0")
            self.assertEqual(command[command.index("--operator-world-z-mm") + 1], "3.5")

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
        self.assertEqual(
            [specs[index]["variant"] for index in (0, 2, 4)],
            ["skeleton_0_1_bvh_fk_21_joint"] * 3,
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
            workbench = root / "delivery/calibration-workbench"
            workbench.mkdir(parents=True)
            metadata = workbench / "take007_alignment.json"
            metadata.write_text(
                json.dumps({"cmm": str(root / "data/Take_007.cmm")}) + "\n",
                encoding="utf-8",
            )
            applied = workbench / "applied_manual_profile.json"
            applied.write_text(
                json.dumps({"operator_note": str(root / "keep-exact")}) + "\n",
                encoding="utf-8",
            )
            changed = normalize_delivery_provenance(root, root / "delivery")
            self.assertEqual(changed, sorted([path.resolve(), metadata.resolve()]))
            payload = __import__("json").loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["inside"], "project://data/take.json")
            self.assertEqual(payload["uri"], "https://example.test/evidence")
            self.assertEqual(
                json.loads(metadata.read_text())["cmm"],
                "project://data/Take_007.cmm",
            )
            self.assertEqual(
                json.loads(applied.read_text())["operator_note"],
                str(root / "keep-exact"),
            )

    def test_refresh_auxiliary_hashes_includes_calibration_workbench(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            delivery = Path(directory)
            poster = delivery / "posters/01.jpg"
            metrics = delivery / "metrics/01.json"
            frame_map = delivery / "frame_maps/01.csv"
            metadata = delivery / "calibration-workbench/take007_alignment.json"
            clean = delivery / "calibration-workbench/take007_clean_rgb.mp4"
            mocap = delivery / "calibration-workbench/take007_mocap.f32"
            solved = delivery / "calibration-workbench/take007_solved.f32"
            profile = delivery / "calibration-workbench/applied_manual_profile.json"
            for index, artifact in enumerate(
                (poster, metrics, frame_map, clean, mocap, solved, profile)
            ):
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(f"artifact-{index}".encode())
            metadata.write_text(
                json.dumps(
                    {
                        "video": {
                            "path": "take007_clean_rgb.mp4",
                            "sha256": "stale-inner-clean",
                        },
                        "layers": {
                            "mocap": {
                                "path": "take007_mocap.f32",
                                "sha256": "stale-inner-mocap",
                            },
                            "solved": {
                                "path": "take007_solved.f32",
                                "sha256": "stale-inner-solved",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            profile_hash = sha256_file(profile)
            manifest = {
                "videos": [{
                    "poster": "posters/01.jpg",
                    "poster_sha256": "stale",
                    "metrics": "metrics/01.json",
                    "metrics_sha256": "stale",
                    "frame_map": "frame_maps/01.csv",
                    "frame_map_sha256": "stale",
                }],
                "calibration_workbench": {
                    "path": "calibration-workbench/take007_alignment.json",
                    "sha256": "stale",
                    "clean_rgb": {
                        "path": "calibration-workbench/take007_clean_rgb.mp4",
                        "sha256": "stale",
                    },
                    "mocap_trajectory": {
                        "path": "calibration-workbench/take007_mocap.f32",
                        "sha256": "stale",
                    },
                    "solved_trajectory": {
                        "path": "calibration-workbench/take007_solved.f32",
                        "sha256": "stale",
                    },
                    "applied_manual_profile": {
                        "path": "calibration-workbench/applied_manual_profile.json",
                        "sha256": profile_hash,
                    },
                },
            }
            (delivery / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            refresh_auxiliary_hashes(delivery)
            refreshed = json.loads((delivery / "manifest.json").read_text())
            refreshed_metadata = json.loads(metadata.read_text())
            manifest_before_profile_mutation = (delivery / "manifest.json").read_text()
            profile.write_text("changed without rerender", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rebuild all rendered videos"):
                refresh_auxiliary_hashes(delivery)
            self.assertEqual(
                (delivery / "manifest.json").read_text(),
                manifest_before_profile_mutation,
            )
        self.assertNotEqual(refreshed["videos"][0]["metrics_sha256"], "stale")
        self.assertNotEqual(refreshed["calibration_workbench"]["sha256"], "stale")
        self.assertEqual(
            refreshed["calibration_workbench"]["applied_manual_profile"]["sha256"],
            profile_hash,
        )
        self.assertEqual(
            refreshed_metadata["video"]["sha256"],
            refreshed["calibration_workbench"]["clean_rgb"]["sha256"],
        )
        self.assertEqual(
            refreshed_metadata["layers"]["mocap"]["sha256"],
            refreshed["calibration_workbench"]["mocap_trajectory"]["sha256"],
        )
        self.assertEqual(
            refreshed_metadata["layers"]["solved"]["sha256"],
            refreshed["calibration_workbench"]["solved_trajectory"]["sha256"],
        )

    def test_refresh_auxiliary_hashes_rejects_package_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            delivery = root / "delivery"
            delivery.mkdir()
            (root / "outside.jpg").write_bytes(b"outside")
            (delivery / "manifest.json").write_text(
                json.dumps(
                    {
                        "videos": [
                            {
                                "poster": "../outside.jpg",
                                "poster_sha256": "stale",
                                "metrics": None,
                                "metrics_sha256": None,
                                "frame_map": None,
                                "frame_map_sha256": None,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "escapes the delivery"):
                refresh_auxiliary_hashes(delivery)


if __name__ == "__main__":
    unittest.main()
