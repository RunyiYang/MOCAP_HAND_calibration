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

    def test_take007_manual_xyz_workbench_contract(self) -> None:
        html = (PAGE / "index.html").read_text(encoding="utf-8")
        javascript = (PAGE / "app.js").read_text(encoding="utf-8")

        self.assertIn('id="manual-calibration"', html)
        self.assertIn('id="alignment-canvas"', html)
        self.assertIn('id="global-x"', html)
        self.assertIn('id="global-y"', html)
        self.assertIn('id="global-z"', html)
        self.assertIn('id="enable-residual"', html)
        self.assertIn('id="link-hands"', html)
        self.assertIn('id="alignment-reset"', html)
        self.assertIn('id="alignment-export"', html)
        self.assertIn("DEFAULT HAND-LOCAL REAR OFFSET", html)
        self.assertIn("20 mm", html)

        self.assertIn('gt_calib.manual_xyz_workbench.v1', javascript)
        self.assertIn('gt_calib.manual_xyz_profile.v1', javascript)
        self.assertIn('take007_alignment.json', javascript)
        self.assertIn('dtype !== "float32-le"', javascript)
        self.assertIn('camera.world_to_color', javascript)
        self.assertIn('metadata.video?.source_width', javascript)
        self.assertIn('metadata.video?.source_height', javascript)
        self.assertIn('const radial = 1 + k1 * r2', javascript)
        self.assertIn('localStorage.setItem(WORKBENCH_STORAGE_KEY', javascript)
        self.assertIn('take007_manual_xyz_calibration.json', javascript)
        self.assertIn('global_world_xyz_mm:', javascript)
        self.assertIn('left_world_xyz_mm:', javascript)
        self.assertIn('right_world_xyz_mm:', javascript)
        self.assertIn('source_take:', javascript)
        self.assertIn('units: "mm"', javascript)
        self.assertIn('p_world + global_translation + side_residual', javascript)


if __name__ == "__main__":
    unittest.main()
