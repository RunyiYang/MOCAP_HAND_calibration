from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import bvh_web_export as bvh


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "同步整理_20260829_三段"


SYNTHETIC_BVH = """HIERARCHY
ROOT Root
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Child
  {
    OFFSET 1 0 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 1 0 0
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.5
0 0 0 0 0 0 0 0 0
1 2 3 90 0 0 0 0 0
"""


def _write_sync_fixture(
    directory: Path,
    *,
    rows: list[dict[str, object]],
) -> tuple[Path, Path]:
    human = directory / "Human.cma"
    human.write_text(
        "FrameCounter\tValue\n"
        "100\t0\n"
        "101\t0\n"
        "102\t0\n"
        "103\t0\n",
        encoding="utf-8",
    )
    alignment = directory / "take.alignment.csv"
    fields = [
        "output_frame",
        "source_video_frame",
        "source_video_nominal_pts_s",
        "mocap_low_frame_counter",
        "mocap_high_frame_counter",
        "interpolation_alpha",
    ]
    with alignment.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return alignment, human


class GenericBvhParserTests(unittest.TestCase):
    def test_standard_hierarchy_and_forward_kinematics(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "synthetic.bvh"
            path.write_text(SYNTHETIC_BVH, encoding="utf-8")
            clip = bvh.parse_bvh(path)

        self.assertEqual(clip.frame_count, 2)
        self.assertEqual(clip.channel_count, 9)
        self.assertEqual([node.channel_start for node in clip.nodes], [0, 6, 9])
        self.assertEqual([node.parent for node in clip.nodes], [-1, 0, 1])
        self.assertTrue(clip.nodes[-1].is_end_site)

        positions = bvh.evaluate_world_positions(clip)
        np.testing.assert_allclose(positions[0, 0], [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(positions[0, 1], [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(positions[0, 2], [2.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(positions[1, 0], [1.0, 2.0, 3.0], atol=1e-12)
        np.testing.assert_allclose(positions[1, 1], [1.0, 3.0, 3.0], atol=1e-12)
        np.testing.assert_allclose(positions[1, 2], [1.0, 4.0, 3.0], atol=1e-12)

    def test_motion_row_count_channel_count_and_finiteness_fail_closed(self) -> None:
        broken_variants = (
            SYNTHETIC_BVH.replace("Frames: 2", "Frames: 3"),
            SYNTHETIC_BVH.replace(
                "1 2 3 90 0 0 0 0 0", "1 2 3 90 0 0 0 0"
            ),
            SYNTHETIC_BVH.replace(
                "1 2 3 90 0 0 0 0 0", "nan 2 3 90 0 0 0 0 0"
            ),
        )
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            for index, content in enumerate(broken_variants):
                with self.subTest(index=index):
                    path = Path(directory) / f"broken-{index}.bvh"
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(bvh.BvhValidationError):
                        bvh.parse_bvh(path)

    def test_sampling_omits_seed_and_preserves_strict_endpoint(self) -> None:
        # Passing raw N-1 as the exclusive bound models strict source indices
        # 1..N-2 for a raw clip with N frames.
        indices, stride = bvh.sample_source_indices(
            10, 1.0 / 120.0, leading_frames_to_omit=1, target_fps=60.0
        )
        self.assertEqual(stride, 2)
        np.testing.assert_array_equal(indices, [1, 3, 5, 7, 9])


class OverlayBvhMappingTests(unittest.TestCase):
    def test_dummy_seed_is_null_and_following_frame_maps_without_plus_one(self) -> None:
        rows = [
            {
                "output_frame": 0,
                "source_video_frame": 20,
                "source_video_nominal_pts_s": 20 / 30,
                "mocap_low_frame_counter": 100,
                "mocap_high_frame_counter": 101,
                "interpolation_alpha": 0.25,
            },
            {
                "output_frame": 1,
                "source_video_frame": 21,
                "source_video_nominal_pts_s": 21 / 30,
                "mocap_low_frame_counter": 102,
                "mocap_high_frame_counter": 103,
                "interpolation_alpha": 0.75,
            },
        ]
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            alignment, human = _write_sync_fixture(Path(directory), rows=rows)
            sync = bvh.load_overlay_video_sync(
                alignment, human, bvh_frame_count=5, expected_output_frame_count=2
            )

        self.assertEqual(sync["bvh_fractional_source_frame_by_output_frame"], [None, 2.75])
        self.assertEqual(sync["first_valid_output_frame"], 1)
        self.assertEqual(sync["unavailable_output_frames"], [0])
        self.assertEqual(sync["bvh_bracket_bounds"]["strictly_usable_source_frame_last"], 3)

    def test_alignment_row_count_mismatch_fails_closed(self) -> None:
        rows = [
            {
                "output_frame": 0,
                "source_video_frame": 20,
                "source_video_nominal_pts_s": 20 / 30,
                "mocap_low_frame_counter": 101,
                "mocap_high_frame_counter": 102,
                "interpolation_alpha": 0.5,
            },
            {
                "output_frame": 1,
                "source_video_frame": 21,
                "source_video_nominal_pts_s": 21 / 30,
                "mocap_low_frame_counter": 102,
                "mocap_high_frame_counter": 103,
                "interpolation_alpha": 0.5,
            },
        ]
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            alignment, human = _write_sync_fixture(Path(directory), rows=rows)
            with self.assertRaisesRegex(bvh.BvhValidationError, "expected 3"):
                bvh.load_overlay_video_sync(
                    alignment, human, bvh_frame_count=5, expected_output_frame_count=3
                )

    def test_alignment_bvh_bounds_fail_closed(self) -> None:
        rows = [
            {
                "output_frame": 0,
                "source_video_frame": 20,
                "source_video_nominal_pts_s": 20 / 30,
                "mocap_low_frame_counter": 102,
                "mocap_high_frame_counter": 103,
                "interpolation_alpha": 0.5,
            },
            {
                "output_frame": 1,
                "source_video_frame": 21,
                "source_video_nominal_pts_s": 21 / 30,
                "mocap_low_frame_counter": 103,
                "mocap_high_frame_counter": 104,
                "interpolation_alpha": 0.5,
            },
        ]
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            alignment, human = _write_sync_fixture(Path(directory), rows=rows)
            with self.assertRaisesRegex(bvh.BvhValidationError, "out of bounds"):
                bvh.load_overlay_video_sync(
                    alignment, human, bvh_frame_count=5, expected_output_frame_count=2
                )


class FormalDatasetBvhTests(unittest.TestCase):
    def test_only_skeleton_0_and_1_are_selected_for_all_formal_takes(self) -> None:
        expected_frames = {"01": 7216, "02": 7483, "03": 7350}
        for key in bvh.FORMAL_SEGMENTS:
            with self.subTest(segment=key):
                _, left_path, right_path, _ = bvh.discover_formal_take(DATASET, key)
                self.assertTrue(left_path.name.endswith("_Skeleton_0.bvh"))
                self.assertTrue(right_path.name.endswith("_Skeleton_1.bvh"))
                left = bvh.parse_bvh(left_path)
                right = bvh.parse_bvh(right_path)
                bvh._validate_hand_clip(left, side="left", segment_key=key)
                bvh._validate_hand_clip(right, side="right", segment_key=key)
                self.assertEqual(left.frame_count, expected_frames[key])
                self.assertEqual(right.frame_count, expected_frames[key])
                self.assertEqual(bvh.leading_zero_motion_frames(left), 1)
                self.assertEqual(bvh.leading_zero_motion_frames(right), 1)

    def test_take_000_bvh1_matches_second_cma_row_not_first(self) -> None:
        _, left_path, _, take_name = bvh.discover_formal_take(DATASET, "01")
        clip = bvh.parse_bvh(left_path)
        human = left_path.parent / f"{take_name}_Human.cma"
        with human.open("r", encoding="utf-8-sig") as stream:
            header = stream.readline().rstrip("\r\n").split("\t")
            indices = [
                header.index(f"Skeleton_0_LeftHand_Pos_{axis}(mm)")
                for axis in "XYZ"
            ]
            first = stream.readline().rstrip("\r\n").split("\t")
            second = stream.readline().rstrip("\r\n").split("\t")
        first_cma_xyz = np.asarray([float(first[index]) for index in indices])
        second_cma_xyz = np.asarray([float(second[index]) for index in indices])

        # MovementCap's BVH export maps CMA mm to its unnamed BVH units as
        # [-X, Z, Y] / 10.  The exact source values prove the ordinal contract.
        first_as_bvh = np.asarray([-first_cma_xyz[0], first_cma_xyz[2], first_cma_xyz[1]]) / 10.0
        second_as_bvh = np.asarray([-second_cma_xyz[0], second_cma_xyz[2], second_cma_xyz[1]]) / 10.0
        np.testing.assert_allclose(clip.motion[1, :3], second_as_bvh, atol=5e-5, rtol=0.0)
        self.assertGreater(float(np.max(np.abs(clip.motion[1, :3] - first_as_bvh))), 1e-3)
        np.testing.assert_allclose(
            clip.motion[1, :3], [24.88331909, 23.27032166, 19.4625351], atol=0.0, rtol=0.0
        )

    def test_formal_video_sync_mapping_known_endpoints(self) -> None:
        expected = {
            "01": ([None, 4.89818], 7009.78089, 1, [0]),
            "02": ([85.239313, 89.251502], 7322.928612, 0, []),
            "03": ([None, 4.613476], 7218.0701, 1, [0]),
        }
        for key in bvh.FORMAL_SEGMENTS:
            with self.subTest(segment=key):
                segment, left, _, take = bvh.discover_formal_take(DATASET, key)
                alignment = (
                    bvh.DEFAULT_ALIGNMENT_DIR
                    / f"{segment.name}_mocap_video_aligned_strict25.alignment.csv"
                )
                sync = bvh.load_overlay_video_sync(
                    alignment,
                    left.parent / f"{take}_Human.cma",
                    bvh_frame_count=bvh.EXPECTED_SOURCE_FRAME_COUNTS[key],
                    expected_output_frame_count=bvh.EXPECTED_OVERLAY_FRAME_COUNTS[key],
                )
                first_two, last, first_valid, unavailable = expected[key]
                self.assertEqual(sync["bvh_fractional_source_frame_by_output_frame"][:2], first_two)
                self.assertEqual(sync["bvh_fractional_source_frame_by_output_frame"][-1], last)
                self.assertEqual(sync["first_valid_output_frame"], first_valid)
                self.assertEqual(sync["unavailable_output_frames"], unavailable)

    def test_compact_payload_quantization_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            output_dir = Path(directory)
            manifest = bvh.export_segments(
                DATASET,
                output_dir,
                ("01",),
                target_fps=1.0,
                quantum_bvh_units=0.001,
            )
            manifest_on_disk = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest_on_disk, manifest)
            self.assertEqual(manifest["schema"], bvh.MANIFEST_SCHEMA)
            entry = manifest["takes"][0]
            payload = json.loads((output_dir / entry["file"]).read_text(encoding="utf-8"))

        self.assertEqual(payload["schema"], bvh.EXPORT_SCHEMA)
        self.assertEqual(payload["motion"]["source_frame_indices"][0], 1)
        self.assertEqual(payload["motion"]["source_frame_indices"][-1], 7214)
        self.assertEqual(payload["positions"]["total_joint_count"], 42)
        self.assertEqual(payload["positions"]["values_per_frame"], 126)
        self.assertLessEqual(
            payload["positions"]["max_abs_quantization_error_bvh_units"], 0.000500001
        )
        self.assertTrue(payload["source"]["left"]["path"].endswith("_Skeleton_0.bvh"))
        self.assertTrue(payload["source"]["right"]["path"].endswith("_Skeleton_1.bvh"))
        self.assertEqual(
            payload["video_sync"]["bvh_fractional_source_frame_by_output_frame"][0],
            None,
        )


if __name__ == "__main__":
    unittest.main()
