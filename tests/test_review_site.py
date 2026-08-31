from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
PUBLIC = WEB / "public"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReviewSiteContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run(
            [sys.executable, str(WEB / "build_site.py")],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        cls.data = json.loads(
            (PUBLIC / "data/site-data.json").read_text(encoding="utf-8")
        )

    def test_site_exposes_four_distinct_review_modes(self) -> None:
        self.assertEqual(self.data["schema"], "gt-calib.review-site.v2")
        self.assertEqual(
            self.data["status"],
            "verified_exposure_time_alignment_with_full_frame_rgb_identity",
        )
        for key in ("mocapTakes", "depthTakes", "takes", "diagnosticTakes"):
            self.assertEqual(len(self.data[key]), 3)
        self.assertEqual(
            [take["validFrames"] for take in self.data["mocapTakes"]],
            [1748, 1805, 1800],
        )
        self.assertTrue(
            all(take["coveragePercent"] == 100.0 for take in self.data["mocapTakes"])
        )
        self.assertTrue(
            all(
                take["contentBestGlobalOffsetFrames"] == 0
                for take in self.data["mocapTakes"]
            )
        )
        self.assertTrue(all(take["contentFullFrame"] for take in self.data["mocapTakes"]))
        self.assertTrue(
            all(
                take["contentSampleCount"] == take["contentSourceFrameCount"]
                for take in self.data["mocapTakes"]
            )
        )
        self.assertTrue(
            all(
                take["contentDistinguishableSameIndexCount"]
                == take["contentDistinguishableCount"]
                for take in self.data["mocapTakes"]
            )
        )
        for take in self.data["mocapTakes"]:
            self.assertTrue(take["visualizationOnly"])
            self.assertTrue(take["rearMountProxyEnabled"])
            self.assertEqual(take["rearMountProxyScale"], 0.8)
            self.assertEqual(take["deliveredJointCountPerHand"], 21)
            self.assertFalse(take["original21JointPositionsModified"])
            self.assertFalse(take["additionalGroundTruthJoint"])
            self.assertTrue(take["alignmentIdenticalToRaw21JointView"])
            self.assertIn("proxy", take["title"])
            self.assertIn("21 joints", take["summary"])
            self.assertIn("不是 GT", take["summary"])
        self.assertEqual(
            [take["validFrames"] for take in self.data["depthTakes"]],
            [1747, 1804, 1799],
        )
        for take in self.data["depthTakes"]:
            self.assertEqual(take["coveragePercent"], 100.0)
            self.assertTrue(take["timestampGatesPassed"])
            self.assertTrue(take["se3GatesPassed"])
            self.assertFalse(take["skeletonToSurfaceAccuracyMeasured"])
            self.assertIn("不是 pose accuracy", take["coverageInterpretation"])
            self.assertAlmostEqual(take["rotationDeterminant"], 1.0, places=9)
            self.assertGreater(take["depthRgbMedianMs"], 1.2)
            self.assertLess(take["depthRgbMedianMs"], 1.4)
            self.assertEqual(take["alignmentRows"], take["validFrames"])

    def test_all_generated_site_links_are_allow_listed_files(self) -> None:
        for collection in ("mocapTakes", "depthTakes", "takes", "diagnosticTakes"):
            for take in self.data[collection]:
                for key in ("video", "poster", "metricsDownload"):
                    path = PUBLIC / urlparse(take[key]).path.lstrip("/")
                    self.assertTrue(path.is_file(), f"missing {collection}.{key}: {path}")
                if take.get("alignmentDownload"):
                    path = PUBLIC / take["alignmentDownload"].lstrip("/")
                    self.assertTrue(path.is_file(), f"missing frame map: {path}")
                for key in ("rawVideoDownload", "rawPosterDownload"):
                    if take.get(key):
                        path = PUBLIC / take[key].lstrip("/")
                        self.assertTrue(path.is_file(), f"missing {collection}.{key}: {path}")

        manifest = json.loads(
            (PUBLIC / "data/asset-manifest.json").read_text(encoding="utf-8")
        )
        self.assertTrue(manifest["assets"])
        self.assertLessEqual(
            max(item["bytes"] for item in manifest["assets"]),
            manifest["cloudflareMaxAssetBytes"],
        )

    def test_authored_page_defaults_to_mocap_and_has_privacy_headers(self) -> None:
        html = (PUBLIC / "index.html").read_text(encoding="utf-8")
        self.assertIn('data-view-mode="mocap" aria-pressed="true"', html)
        self.assertIn('data-view-mode="depth" aria-pressed="false"', html)
        self.assertEqual(html.count("data-mocap-src="), 3)
        self.assertEqual(html.count("data-depth-src="), 3)
        self.assertEqual(html.count("data-depth-poster="), 3)
        self.assertEqual(html.count("data-alignment-link"), 3)
        self.assertEqual(html.count("data-raw-mocap-video-link"), 3)
        self.assertEqual(html.count("data-raw-mocap-poster-link"), 3)
        self.assertIn("rear wrist module proxy", html)
        self.assertIn("VIZ-only", html)
        self.assertIn("MOCAP_WRIST_SEMANTICS_V1.md", html)
        self.assertIn("gt-calib-evidence-2026-08-31.zip", html)
        self.assertIn("/downloads/DEPTH_MOCAP_ALIGNMENT_V1.md", html)
        self.assertTrue(
            (PUBLIC / "downloads/DEPTH_MOCAP_ALIGNMENT_V1.md").is_file()
        )
        self.assertIn("BAG color message i", html)

        headers = (PUBLIC / "_headers").read_text(encoding="utf-8")
        self.assertIn("Content-Security-Policy:", headers)
        self.assertIn("X-Robots-Tag: noindex, noarchive, nosnippet", headers)
        self.assertIn("frame-ancestors 'none'", headers)

    def test_rear_proxy_and_raw_mocap_views_share_the_exact_frame_map(self) -> None:
        audit = self.data["wristSemanticsAudit"]
        self.assertEqual(audit["heldOutFrames"], 24)
        self.assertTrue(audit["hypothesisFrameExcluded"])
        self.assertEqual(audit["deliveredJointCountPerHand"], 21)
        self.assertFalse(audit["originalJointPositionsModified"])
        self.assertEqual(len(audit["contacts"]), 3)
        for contact in audit["contacts"]:
            self.assertTrue((PUBLIC / contact["image"].lstrip("/")).is_file())
            self.assertEqual(len(contact["selectedFrames"]), 8)
        for key in ("summaryDownload", "provenanceDownload", "reportDownload"):
            self.assertTrue((PUBLIC / audit[key].lstrip("/")).is_file())

        segments = (
            "01_210814_Take_000",
            "02_210955_Take_001",
            "03_211139_Take_002",
        )
        for take, segment in zip(self.data["mocapTakes"], segments, strict=True):
            stem = f"{segment}_mocap_video_aligned_strict25"
            raw_csv = ROOT / "outputs/mocap_video_alignment_review" / f"{stem}.alignment.csv"
            rear_csv = ROOT / "outputs/mocap_video_rear_mount_review" / f"{stem}.alignment.csv"
            published_csv = PUBLIC / take["alignmentDownload"].lstrip("/")
            self.assertEqual(sha256(raw_csv), sha256(rear_csv))
            self.assertEqual(sha256(rear_csv), sha256(published_csv))

            metrics = json.loads(
                (PUBLIC / take["metricsDownload"].lstrip("/")).read_text(
                    encoding="utf-8"
                )
            )
            spatial = metrics["spatial_alignment"]
            proxy = spatial["rear_wrist_mount_visualization"]
            self.assertTrue(proxy["enabled"])
            self.assertFalse(proxy["original_21_joint_positions_modified"])
            self.assertFalse(proxy["additional_mocap_gt_joint"])
            self.assertFalse(spatial["mocap_world_translation"]["enabled"])

    def test_depth_media_and_metrics_are_full_interval_h264_evidence(self) -> None:
        for take in self.data["depthTakes"]:
            video = PUBLIC / take["video"].lstrip("/")
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,width,height,nb_frames",
                    "-of",
                    "json",
                    str(video),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            stream = json.loads(probe.stdout)["streams"][0]
            self.assertEqual(stream["codec_name"], "h264")
            self.assertEqual((stream["width"], stream["height"]), (640, 576))
            self.assertEqual(int(stream["nb_frames"]), take["validFrames"])

            metrics_path = PUBLIC / take["metricsDownload"].lstrip("/")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["schema"], "gt_calib.depth_mocap_overlay.v1")
            frames = metrics["depth_frame_contract"]
            temporal = metrics["temporal_alignment"]
            spatial = metrics["spatial_projection_coverage"]
            self.assertTrue(frames["covers_full_synchronized_depth_interval"])
            self.assertTrue(frames["decoded_device_timestamps_match_bag_index"])
            self.assertTrue(frames["decoded_frame_numbers_strictly_increasing"])
            self.assertTrue(frames["rgb_depth_pairing"]["pass"])
            self.assertTrue(
                frames["rgb_depth_pairing"][
                    "same_index_is_nearest_for_all_paired_messages"
                ]
            )
            self.assertTrue(temporal["depth_timestamp_used_directly_not_rgb_timestamp_substituted"])
            self.assertTrue(temporal["all_published_frames_have_valid_interpolation"])
            self.assertTrue(spatial["world_to_depth_camera_frozen_no_refit"])
            self.assertTrue(spatial["recorded_camera_contract"]["pass"])
            self.assertTrue(
                all(spatial["recorded_camera_contract"]["gates"].values())
            )
            self.assertLessEqual(
                spatial["rotation_orthogonality_max_abs_error"], 1e-9
            )
            self.assertFalse(spatial["skeleton_to_surface_error_measured"])

            alignment_path = PUBLIC / take["alignmentDownload"].lstrip("/")
            with alignment_path.open(encoding="utf-8", newline="") as stream_handle:
                rows = list(csv.DictReader(stream_handle))
            self.assertEqual(len(rows), take["validFrames"])
            self.assertEqual(int(rows[0]["output_frame"]), 0)
            self.assertEqual(
                int(rows[-1]["output_frame"]), take["validFrames"] - 1
            )
            self.assertEqual(
                int(rows[0]["source_depth_frame"]),
                take["firstSourceDepthFrame"],
            )
            self.assertEqual(
                int(rows[-1]["source_depth_frame"]),
                take["lastSourceDepthFrame"],
            )

    def test_current_evidence_index_hashes_resolve(self) -> None:
        with (ROOT / "docs/evidence/index.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(rows)
        for row in rows:
            if not row["sha256"]:
                continue
            artifact = ROOT / row["artifact_path"]
            self.assertTrue(artifact.is_file(), artifact)
            self.assertEqual(sha256(artifact), row["sha256"], artifact)

    def test_bvh_viewer_publishes_only_the_two_articulated_hand_skeletons(self) -> None:
        manifest_path = PUBLIC / "bvh/data/manifest.json"
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], "gt-calib-bvh-web-manifest-v1")
        self.assertEqual(manifest["export_schema"], "gt-calib-bvh-web-v1")
        self.assertEqual(manifest["validation"]["status"], "pass")
        self.assertEqual([entry["segment_key"] for entry in manifest["takes"]], ["01", "02", "03"])

        expected = {
            "01": (7216, 7214, 1748, 1, [0], "000"),
            "02": (7483, 7481, 1805, 0, [], "001"),
            "03": (7350, 7348, 1800, 1, [0], "002"),
        }
        for entry in manifest["takes"]:
            key = entry["segment_key"]
            source_count, strict_last, video_count, first_valid, unavailable, number = expected[key]
            payload_path = PUBLIC / "bvh/data" / entry["file"]
            self.assertTrue(payload_path.is_file())
            self.assertEqual(sha256(payload_path), entry["sha256"])
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "gt-calib-bvh-web-v1")
            self.assertEqual(payload["source"]["left"]["skeleton"], "Skeleton_0")
            self.assertEqual(payload["source"]["right"]["skeleton"], "Skeleton_1")
            self.assertEqual([item["joint_count"] for item in payload["skeletons"]], [21, 21])
            self.assertEqual([len(item["edges"]) for item in payload["skeletons"]], [20, 20])
            self.assertEqual(payload["positions"]["total_joint_count"], 42)
            self.assertEqual(payload["motion"]["source_frame_count"], source_count)
            self.assertEqual(payload["motion"]["strict_source_frame_first"], 1)
            self.assertEqual(payload["motion"]["strict_source_frame_last"], strict_last)
            sync = payload["video_sync"]
            self.assertEqual(sync["overlay_output_frame_count"], video_count)
            self.assertEqual(sync["first_valid_output_frame"], first_valid)
            self.assertEqual(sync["unavailable_output_frames"], unavailable)
            self.assertTrue(sync["validation"]["dummy_seed_mappings_explicitly_null"])
            published_video = PUBLIC / f"media/take-{number}-mocap.mp4"
            self.assertEqual(
                entry["h264_overlay_video"]["sha256"], sha256(published_video)
            )
            self.assertIn(
                "outputs/mocap_video_rear_mount_review",
                entry["h264_overlay_video"]["path"],
            )
            for skeleton in ("0", "1"):
                raw = PUBLIC / f"bvh/data/take-{number}-skeleton-{skeleton}.bvh"
                self.assertTrue(raw.is_file(), raw)
                self.assertLess(raw.stat().st_size, 25 * 1024 * 1024)

        published_names = {path.name.lower() for path in (PUBLIC / "bvh/data").glob("*.bvh")}
        self.assertFalse(any("arm" in name or "forearm" in name or "torso" in name for name in published_names))

    def test_bvh_viewer_is_offline_and_uses_video_frame_mapping(self) -> None:
        html = (PUBLIC / "bvh/index.html").read_text(encoding="utf-8")
        javascript = (PUBLIC / "bvh/bvh-viewer.js").read_text(encoding="utf-8")
        self.assertIn('id="motion-canvas"', html)
        self.assertIn("Skeleton_0", html)
        self.assertIn("Skeleton_1", html)
        self.assertIn('/bvh/bvh-viewer.js', html)
        self.assertNotIn("https://", html)
        self.assertNotIn("http://", html)
        self.assertIn("bvh_fractional_source_frame_by_output_frame", javascript)
        self.assertIn("sourceSampleBracket", javascript)
        self.assertIn("requestVideoFrameCallback", javascript)
        self.assertIn("crypto.subtle", javascript)


if __name__ == "__main__":
    unittest.main()
