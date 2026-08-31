from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "web/public/final-nine"
DELIVERY = ROOT / "final_9_video_delivery"


class FinalNineSiteContractTests(unittest.TestCase):
    def test_page_authors_exactly_the_nine_manifest_video_ids(self) -> None:
        html = (PAGE / "index.html").read_text(encoding="utf-8")
        manifest = json.loads((DELIVERY / "manifest.json").read_text(encoding="utf-8"))
        authored_ids = re.findall(r'data-video-id="([^"]+)"', html)

        self.assertEqual(html.count("<video "), 9)
        self.assertEqual(len(authored_ids), 9)
        self.assertEqual(set(authored_ids), {entry["id"] for entry in manifest["videos"]})
        self.assertEqual(manifest["video_count"], 9)

    def test_page_is_local_only_and_states_the_evidence_boundaries(self) -> None:
        html = (PAGE / "index.html").read_text(encoding="utf-8")
        javascript = (PAGE / "app.js").read_text(encoding="utf-8")

        self.assertNotIn("https://", html)
        self.assertNotIn("http://", html)
        self.assertIn("same-recording consistency", html)
        self.assertIn("不是独立动态手部 GT 精度", html)
        self.assertIn("#11 不作为第 21 点", html)
        self.assertIn("69.0%", html)
        self.assertIn("Take_007", html)
        self.assertIn("无手套世界坐标系", html)
        self.assertIn('/downloads/final-nine/manifest.json', html)
        self.assertIn('fetch(`${DELIVERY_ROOT}/manifest.json`', javascript)
        self.assertIn("manifest?.video_count !== 9", javascript)

    def test_root_page_links_prominently_to_final_review(self) -> None:
        html = (ROOT / "web/public/index.html").read_text(encoding="utf-8")
        self.assertGreaterEqual(html.count('href="/final-nine/"'), 2)
        self.assertTrue((PAGE / "styles.css").is_file())
        self.assertTrue((PAGE / "app.js").is_file())

    def test_nine_video_manual_xyz_workbench_contract(self) -> None:
        html = (PAGE / "index.html").read_text(encoding="utf-8")
        javascript = (PAGE / "app.js").read_text(encoding="utf-8")

        self.assertIn('id="manual-calibration"', html)
        self.assertIn('id="alignment-canvas"', html)
        self.assertIn('id="global-x"', html)
        self.assertIn('id="global-y"', html)
        self.assertIn('id="global-z"', html)
        self.assertIn('id="annotation-video"', html)
        self.assertIn('id="annotation-mode"', html)
        self.assertIn('id="video-x"', html)
        self.assertIn('id="video-y"', html)
        self.assertIn('id="video-z"', html)
        self.assertRegex(html, r'id="global-y"[^>]+value="-44"')
        self.assertIn('id="enable-residual"', html)
        self.assertIn('id="link-hands"', html)
        self.assertIn('id="alignment-reset"', html)
        self.assertIn('id="alignment-export"', html)
        self.assertIn("TAKE_007 HAND-LOCAL REAR OFFSET", html)
        self.assertIn("20 mm", html)
        self.assertIn("每段视频自己的", html)
        self.assertIn("第 09 段无 MOCAP", html)

        self.assertIn('gt_calib.manual_xyz_workbench.v1', javascript)
        self.assertIn('gt_calib.final_nine_manual_xyz_state.v3', javascript)
        self.assertIn('gt_calib.final_nine_manual_xyz.v1', javascript)
        self.assertIn('gt_calib.final_nine.manual_xyz.v3', javascript)
        self.assertIn('take007_alignment.json', javascript)
        self.assertIn('dtype !== "float32-le"', javascript)
        self.assertIn('camera.world_to_color', javascript)
        self.assertIn('metadata.video?.source_width', javascript)
        self.assertIn('metadata.video?.source_height', javascript)
        self.assertIn('const radial = 1 + k1 * r2', javascript)
        self.assertIn('localStorage.setItem(WORKBENCH_STORAGE_KEY', javascript)
        self.assertIn('final_nine_manual_xyz_annotations.json', javascript)
        self.assertIn('global_world_xyz_mm:', javascript)
        self.assertIn('per_video_world_xyz_mm:', javascript)
        self.assertIn('left_residual_world_xyz_mm:', javascript)
        self.assertIn('right_residual_world_xyz_mm:', javascript)
        self.assertIn('video_sha256:', javascript)
        self.assertIn('units: "mm"', javascript)
        self.assertIn('coordinate_frame: "per_video_mocap_world"', javascript)
        self.assertIn('global_translation + per_video_translation + optional_side_residual', javascript)

    def test_applied_profile_is_the_baseline_and_drafts_are_content_bound(self) -> None:
        html = (PAGE / "index.html").read_text(encoding="utf-8")
        javascript = (PAGE / "app.js").read_text(encoding="utf-8")

        self.assertIn("function loadAppliedManualProfile(manifest, descriptor)", javascript)
        self.assertIn("descriptor?.applied_manual_profile", javascript)
        self.assertIn("safeRelativePath(profileDescriptor.path)", javascript)
        self.assertNotIn("workbenchAssetPath(profileDescriptor.path", javascript)
        self.assertIn("validateAppliedManualProfile(await response.json(), manifest)", javascript)
        self.assertIn("adjustment.global = appliedProfileVector(profile.global_world_xyz_mm", javascript)
        self.assertIn("row.per_video_world_xyz_mm", javascript)
        self.assertIn("row.left_residual_world_xyz_mm", javascript)
        self.assertIn("row.right_residual_world_xyz_mm", javascript)
        self.assertIn("manifestVideoFingerprints(manifest)", javascript)
        self.assertIn("`${entry.order}:${entry.id}:${entry.filename}:${entry.sha256}`", javascript)
        self.assertIn("manifest_video_fingerprints", javascript)
        self.assertIn("workbench_metadata_sha256", javascript)
        self.assertIn("applied_profile_sha256", javascript)
        self.assertIn("exactDraftBindingMatches(parsed?.binding, binding)", javascript)
        self.assertIn("workbench.baselineAdjustment = appliedProfile.adjustment", javascript)
        self.assertIn("workbench.baselineAdjustment,", javascript)
        self.assertIn("恢复已应用 profile 基线", html)

    def test_static_old_take_copy_matches_bvh_delivery_intervals(self) -> None:
        html = (PAGE / "index.html").read_text(encoding="utf-8")
        take01 = html[html.index('data-pair="take01"'):html.index('data-pair="take02"')]
        take02 = html[html.index('data-pair="take02"'):html.index('data-pair="take03"')]
        take03 = html[html.index('data-pair="take03"'):html.index('data-pair="take007"')]

        self.assertIn("1747 帧", take01)
        self.assertEqual(take01.count("1747 frames"), 2)
        self.assertNotIn("1748", take01)
        self.assertIn("1805 帧", take02)
        self.assertEqual(take02.count("1805 frames"), 2)
        self.assertIn("1799 帧", take03)
        self.assertEqual(take03.count("1799 frames"), 2)
        self.assertNotIn("1800", take03)
        self.assertGreaterEqual(html.count("BVH 正向运动学"), 3)
        self.assertGreaterEqual(html.count("Human.cma"), 3)
        self.assertNotIn("rear wrist point", html)
        self.assertNotIn("rear wrist proxy", html)

    def test_selector_is_manifest_driven_and_switches_real_preview_sources(self) -> None:
        javascript = (PAGE / "app.js").read_text(encoding="utf-8")

        self.assertIn("selector.replaceChildren(...manifest.videos.map", javascript)
        self.assertIn("workbench.adjustment.selectedVideoId = videoId", javascript)
        self.assertIn("switchSelectedPreview();", javascript)
        self.assertIn('safeRelativePath(`videos/${entry.filename}`)', javascript)
        self.assertIn("workbench.metadata.video.path", javascript)
        self.assertIn('const LAYER_VIDEO_IDS = { mocap: "take007-mocap-markers", solved: "take007-solved" }', javascript)
        self.assertIn('workbench.adjustment.showMocap = true', javascript)
        self.assertIn('workbench.adjustment.showSolved = true', javascript)
        self.assertIn("正式已烘焙 MP4", javascript)
        self.assertIn("需离线重渲染", javascript)

    def test_export_is_strictly_bound_to_all_nine_manifest_entries(self) -> None:
        javascript = (PAGE / "app.js").read_text(encoding="utf-8")

        self.assertIn("manifest.videos.map((entry) =>", javascript)
        self.assertIn("payload.video_annotations.length !== 9", javascript)
        self.assertIn("item.video_id !== source.id", javascript)
        self.assertIn("item.filename !== source.filename", javascript)
        self.assertIn("item.video_sha256 !== source.sha256", javascript)
        self.assertIn("item.apply_translation !== shouldApply", javascript)
        self.assertIn('"no-glove-calibration"', javascript)
        self.assertIn('"no_mocap_hand_pose"', javascript)
        self.assertIn("item.effective_left_world_xyz_mm !== null", javascript)
        self.assertIn("item.effective_right_world_xyz_mm !== null", javascript)
        self.assertIn("Side-specific", (ROOT / "src/gt_calib_delivery/manual_profiles.py").read_text(encoding="utf-8"))

    def test_manifest_backed_assets_use_sha_cache_busting(self) -> None:
        javascript = (PAGE / "app.js").read_text(encoding="utf-8")

        self.assertIn("function cacheBustedPath(path, sha256)", javascript)
        self.assertIn('sha256=${sha256}', javascript)
        self.assertIn("entry.poster_sha256", javascript)
        self.assertIn("entry.metrics_sha256", javascript)
        self.assertIn("descriptor.sha256", javascript)
        self.assertIn("metadata.video.sha256", javascript)


if __name__ == "__main__":
    unittest.main()
