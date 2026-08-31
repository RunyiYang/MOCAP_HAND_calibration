from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

import depth_mocap_overlay as depth_overlay
import gt_calib_viz as viz
import mocap_video_overlay as rgb_overlay


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "同步整理_20260829_三段"
CALIBRATION = viz._default_calibration(DATASET)


def _ros_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def _serialized_depth(
    depth_mm: np.ndarray,
    *,
    timestamp_ns: int,
    frame_number: int = 42,
    bigendian: bool = False,
    row_padding_bytes: int = 0,
) -> bytes:
    height, width = depth_mm.shape
    if row_padding_bytes % 2:
        raise ValueError("test row padding must preserve uint16 alignment")
    dtype = np.dtype(">u2" if bigendian else "<u2")
    rows = []
    for row in depth_mm:
        rows.append(row.astype(dtype).tobytes() + b"\x00" * row_padding_bytes)
    image = b"".join(rows)
    metadata = b"metadata-123"
    packed = metadata + image
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    device_timestamp_us = (timestamp_ns + 500) // 1000
    return b"".join(
        (
            struct.pack("<III", 7, seconds, nanoseconds),
            _ros_string("depth_frame"),
            struct.pack("<II", height, width),
            _ros_string("mono16"),
            struct.pack("<B", int(bigendian)),
            struct.pack("<I", width * 2 + row_padding_bytes),
            struct.pack("<II", len(metadata), len(packed)),
            packed,
            struct.pack(
                "<QQQQ",
                frame_number,
                device_timestamp_us,
                device_timestamp_us + 10,
                device_timestamp_us + 20,
            ),
        )
    )


class DepthWireFormatTests(unittest.TestCase):
    def test_decodes_little_endian_mono16_with_metadata_and_row_padding(self) -> None:
        expected = np.asarray([[0, 123, 65535], [17, 900, 1002]], dtype=np.uint16)
        timestamp_ns = 1_788_008_995_600_794_077
        frame = depth_overlay.decode_orbbec_depth_image(
            _serialized_depth(
                expected,
                timestamp_ns=timestamp_ns,
                row_padding_bytes=4,
            ),
            message_index=3,
            message_timestamp_ns=timestamp_ns,
        )
        self.assertEqual(frame.message_index, 3)
        self.assertEqual(frame.encoding, "mono16")
        self.assertEqual((frame.width, frame.height), (3, 2))
        self.assertEqual(frame.depth_mm.dtype, np.uint16)
        np.testing.assert_array_equal(frame.depth_mm, expected)

    def test_decodes_big_endian_and_rejects_wrong_bag_time(self) -> None:
        expected = np.asarray([[1, 256], [4096, 8192]], dtype=np.uint16)
        timestamp_ns = 2_000_000_123
        payload = _serialized_depth(
            expected,
            timestamp_ns=timestamp_ns,
            bigendian=True,
        )
        frame = depth_overlay.decode_orbbec_depth_image(
            payload,
            message_index=0,
            message_timestamp_ns=timestamp_ns,
        )
        np.testing.assert_array_equal(frame.depth_mm, expected)
        with self.assertRaisesRegex(ValueError, "header time"):
            depth_overlay.decode_orbbec_depth_image(
                payload,
                message_index=0,
                message_timestamp_ns=timestamp_ns + 1,
            )


class DepthGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.calibration = depth_overlay.load_depth_calibration(CALIBRATION)

    def test_world_to_depth_is_a_proper_frozen_se3(self) -> None:
        transform = self.calibration.world_to_depth
        rotation = transform[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)
        delivered = json.loads(CALIBRATION.read_text(encoding="utf-8"))
        np.testing.assert_array_equal(
            transform,
            np.asarray(delivered["transforms"]["world_to_depth_camera"]),
        )

    def test_optical_axis_projects_to_depth_principal_point(self) -> None:
        depth_point = np.asarray([0.0, 0.0, 1000.0, 1.0])
        world_point = np.linalg.inv(self.calibration.world_to_depth) @ depth_point
        pixels, positive, depth_z = depth_overlay.project_world_to_depth(
            world_point[:3], self.calibration
        )
        np.testing.assert_allclose(pixels, self.calibration.intrinsics[2:4], atol=1e-9)
        self.assertTrue(bool(positive))
        self.assertAlmostEqual(float(depth_z), 1000.0, places=9)

    def test_metric_preview_uses_fixed_scale_and_black_for_invalid(self) -> None:
        depth = np.asarray([[0, 350, 1075, 1800, 2500]], dtype=np.uint16)
        preview = depth_overlay.metric_depth_preview(
            depth,
            min_depth_mm=350.0,
            max_depth_mm=1800.0,
        )
        self.assertEqual(preview.shape, (1, 5, 3))
        np.testing.assert_array_equal(preview[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(preview[0, 3], preview[0, 4])
        self.assertFalse(np.array_equal(preview[0, 1], preview[0, 2]))


class DepthDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prepared = {
            key: depth_overlay.prepare_depth_mocap(
                viz._discover_segment(DATASET, key), CALIBRATION
            )
            for key in ("01", "02", "03")
        }

    def test_uses_depth_counts_timestamps_and_common_intervals(self) -> None:
        expected = {
            "01": (1816, 1815, 60, 1807, 1747),
            "02": (1812, 1812, 0, 1804, 1804),
            "03": (1810, 1810, 3, 1802, 1799),
        }
        for key, (depth_count, rgb_count, first, stop, count) in expected.items():
            with self.subTest(take=key):
                prepared = self.prepared[key]
                self.assertEqual(len(prepared.depth_refs), depth_count)
                self.assertEqual(len(prepared.rgb_device_s), rgb_count)
                self.assertEqual(
                    rgb_overlay.synchronized_interval(prepared.mocap_valid),
                    (first, stop),
                )
                self.assertEqual(stop - first, count)
                self.assertTrue(np.all(prepared.mocap_valid[first:stop]))
                paired = min(depth_count, rgb_count)
                delta_ms = (
                    prepared.depth_device_s[:paired]
                    - prepared.rgb_device_s[:paired]
                ) * 1000.0
                self.assertGreater(float(np.median(delta_ms)), 1.2)
                self.assertLess(float(np.median(delta_ms)), 1.4)
                self.assertTrue(prepared.camera_contract["pass"])
                self.assertEqual(
                    prepared.camera_contract["serial_number"], "CL8L563010D"
                )
                self.assertEqual(prepared.camera_contract["resolution"], [640, 576])
                self.assertTrue(prepared.depth_rgb_pairing["pass"])
                self.assertTrue(
                    prepared.depth_rgb_pairing[
                        "same_index_is_nearest_for_all_paired_messages"
                    ]
                )
                self.assertEqual(
                    prepared.depth_rgb_pairing["extra_depth_message_count"],
                    depth_count - rgb_count,
                )

    def test_depth_rgb_pairing_rejects_an_interior_index_shift(self) -> None:
        rgb = np.asarray([0, 10_000_000, 20_000_000, 30_000_000], dtype=np.int64)
        valid_depth = rgb + 1_300_000
        self.assertTrue(
            depth_overlay.validate_depth_rgb_pairing(valid_depth, rgb)["pass"]
        )
        shifted_depth = valid_depth.copy()
        shifted_depth[2] = 28_000_000
        with self.assertRaisesRegex(ValueError, "same-index/tail pairing failed"):
            depth_overlay.validate_depth_rgb_pairing(shifted_depth, rgb)

    def test_decodes_raw_depth_and_renders_one_synchronized_frame(self) -> None:
        prepared = self.prepared["02"]
        first, _ = rgb_overlay.synchronized_interval(prepared.mocap_valid)
        frames = list(
            depth_overlay.iter_bag_depth_frames(
                prepared.bag_path,
                prepared.depth_refs,
                first,
                first + 1,
            )
        )
        self.assertEqual(len(frames), 1)
        frame = frames[0]
        self.assertEqual(frame.encoding, "mono16")
        self.assertEqual(frame.depth_mm.shape, (576, 640))
        self.assertEqual(frame.depth_mm.dtype, np.uint16)
        self.assertGreater(float(np.mean(frame.depth_mm > 0)), 0.5)
        rendered, coverage = depth_overlay.render_depth_mocap_frame(
            prepared,
            first,
            0,
            frame.depth_mm,
            first_source_frame=first,
            min_depth_mm=350.0,
            max_depth_mm=1800.0,
        )
        self.assertEqual(rendered.shape, (576, 640, 3))
        self.assertEqual(coverage.joint_count, 42)
        self.assertEqual(coverage.positive_depth_joint_count, 42)
        self.assertEqual(coverage.in_frame_joint_count, 42)
        self.assertGreater(coverage.raw_depth_valid_at_joint_pixel_count, 20)

    def test_output_stem_records_interpolation_gate(self) -> None:
        self.assertEqual(
            depth_overlay.depth_output_stem("take", 25.0),
            "take_depth_mocap_aligned_strict25",
        )
        self.assertEqual(
            depth_overlay.depth_output_stem("take", 12.5),
            "take_depth_mocap_aligned_gap12p5",
        )
        self.assertEqual(
            depth_overlay.depth_output_stem("take", 25.0, max_frames=6),
            "take_depth_mocap_aligned_strict25_smoke6",
        )

    def test_smoke_publish_is_isolated_and_metrics_are_commit_marker(self) -> None:
        prepared = self.prepared["02"]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / (
                depth_overlay.depth_output_stem(
                    prepared.segment_dir.name, 25.0, max_frames=1
                )
                + ".mp4"
            )
            artifacts = depth_overlay.render_depth_mocap_video(
                prepared,
                output,
                snapshot_count=1,
                h264=False,
                max_frames=1,
            )
            self.assertEqual(output, artifacts["mp4v_preview"])
            self.assertTrue(all(path.is_file() for path in artifacts.values()))
            self.assertFalse(any(".staging-" in str(path) for path in artifacts.values()))
            payload = json.loads(
                artifacts["metrics_json"].read_text(encoding="utf-8")
            )
            self.assertEqual(payload["depth_frame_contract"]["published_clip_frames"], 1)
            self.assertFalse(
                payload["depth_frame_contract"][
                    "covers_full_synchronized_depth_interval"
                ]
            )
            for name, record in payload["artifacts"].items():
                self.assertEqual(Path(record["path"]), artifacts[name].resolve())

    def test_depth_frame_map_records_raw_and_interpolated_time(self) -> None:
        prepared = self.prepared["03"]
        first, _ = rgb_overlay.synchronized_interval(prepared.mocap_valid)
        frame = next(
            depth_overlay.iter_bag_depth_frames(
                prepared.bag_path, prepared.depth_refs, first, first + 1
            )
        )
        _, coverage = depth_overlay.render_depth_mocap_frame(
            prepared,
            first,
            0,
            frame.depth_mm,
            first_source_frame=first,
            min_depth_mm=350.0,
            max_depth_mm=1800.0,
        )
        metadata = depth_overlay.DepthFrameMetadata(
            message_index=frame.message_index,
            message_timestamp_ns=frame.message_timestamp_ns,
            device_timestamp_us=frame.device_timestamp_us,
            frame_number=frame.frame_number,
        )
        rows = depth_overlay.depth_frame_mapping_rows(
            prepared, first, first + 1, [coverage], [metadata]
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["output_frame"], 0)
        self.assertEqual(row["source_depth_frame"], first)
        self.assertEqual(row["bag_message_timestamp_ns"], frame.message_timestamp_ns)
        self.assertEqual(row["depth_device_timestamp_us"], frame.device_timestamp_us)
        self.assertGreater(row["depth_minus_rgb_device_timestamp_ms"], 1.2)
        self.assertLess(row["depth_minus_rgb_device_timestamp_ms"], 1.4)
        self.assertEqual(row["mocap_high_frame_counter"] - row["mocap_low_frame_counter"], 1)
        self.assertEqual(row["left_wrist_inside_frame"], 1)
        self.assertEqual(row["right_wrist_inside_frame"], 1)


if __name__ == "__main__":
    unittest.main()
