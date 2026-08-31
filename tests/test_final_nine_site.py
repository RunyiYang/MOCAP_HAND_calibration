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
        self.assertIn("没有 anatomical topology", html)
        self.assertIn("72.4%", html)
        self.assertIn("无手套世界坐标系", html)
        self.assertIn('/downloads/final-nine/manifest.json', html)
        self.assertIn('fetch(`${DELIVERY_ROOT}/manifest.json`', javascript)
        self.assertIn("manifest?.video_count !== 9", javascript)

    def test_root_page_links_prominently_to_final_review(self) -> None:
        html = (ROOT / "web/public/index.html").read_text(encoding="utf-8")
        self.assertGreaterEqual(html.count('href="/final-nine/"'), 2)
        self.assertTrue((PAGE / "styles.css").is_file())
        self.assertTrue((PAGE / "app.js").is_file())


if __name__ == "__main__":
    unittest.main()
