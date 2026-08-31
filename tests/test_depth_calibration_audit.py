from __future__ import annotations

import json
from pathlib import Path
import struct
import unittest

import cv2
import numpy as np

import depth_calibration_audit as audit


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = audit.DEFAULT_CALIBRATION
CONFIG = audit.DEFAULT_CONFIG
MEDIAN_DEPTH = (
    audit.CALIBRATION_PROJECT
    / "results"
    / "manual_final"
    / "reference_median_depth_mm.png"
)


def _ros_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


class DepthBagDecoderTests(unittest.TestCase):
    def test_decode_orbbec_mono16_preserves_millimetres_and_metadata(self) -> None:
        seconds = 3
        nanoseconds = 123_000
        timestamp_ns = seconds * 1_000_000_000 + nanoseconds
        timestamp_us = (timestamp_ns + 500) // 1000
        values = np.asarray([[0, 547, 1002], [1200, 65535, 42]], dtype="<u2")
        metadata = b"xy"
        packed = metadata + values.tobytes()
        payload = b"".join(
            [
                struct.pack("<III", 9, seconds, nanoseconds),
                _ros_string("depth"),
                struct.pack("<II", 2, 3),
                _ros_string("mono16"),
                struct.pack("<B", 0),
                struct.pack("<I", 6),
                struct.pack("<II", len(metadata), len(packed)),
                packed,
                struct.pack("<QQQQ", 77, timestamp_us, 999, 0),
            ]
        )

        decoded = audit.decode_orbbec_depth(
            payload,
            message_index=5,
            message_timestamp_ns=timestamp_ns,
        )

        self.assertEqual(decoded.message_index, 5)
        self.assertEqual(decoded.frame_number, 77)
        self.assertEqual(decoded.device_timestamp_us, timestamp_us)
        self.assertEqual(decoded.image_mm.dtype, np.uint16)
        np.testing.assert_array_equal(decoded.image_mm, values)

    def test_temporal_median_ignores_zero_depth_holes(self) -> None:
        stack = np.asarray(
            [
                [[0, 100], [200, 0]],
                [[50, 0], [220, 0]],
                [[70, 120], [0, 0]],
            ],
            dtype=np.uint16,
        )
        np.testing.assert_array_equal(
            audit.temporal_median_mm(stack),
            np.asarray([[60, 110], [210, 0]], dtype=np.uint16),
        )


class DepthPoseContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_delivered_median_reproduces_depth_se3(self) -> None:
        median = cv2.imread(str(MEDIAN_DEPTH), cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(median)
        self.assertEqual(median.dtype, np.uint16)
        fit = audit.fit_depth_pose(median, self.calibration, self.config)
        delivered = np.asarray(
            self.calibration["transforms"]["depth_camera_to_world"],
            dtype=np.float64,
        )
        np.testing.assert_allclose(fit.depth_to_world, delivered, atol=1e-9, rtol=0.0)
        rotation = fit.depth_to_world[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)

    def test_d2c_is_flagged_affine_and_candidate_is_rigid(self) -> None:
        result, candidate = audit.d2c_audit(self.calibration)
        self.assertFalse(result["is_valid_rotation_at_1e-6"])
        self.assertAlmostEqual(result["determinant"], 0.9949142153, places=8)
        self.assertGreater(
            result["affine_vs_rigid_workspace_projection_delta_px"]["p95"],
            2.0,
        )
        rigid = np.asarray(
            candidate["depth_camera_to_color_camera_rigid_candidate"],
            dtype=np.float64,
        )
        np.testing.assert_allclose(rigid[:3, :3].T @ rigid[:3, :3], np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rigid[:3, :3])), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
