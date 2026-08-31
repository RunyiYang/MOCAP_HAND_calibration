from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from web import build_site


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return path


def _synthetic_delivery(root: Path) -> Path:
    root.mkdir(parents=True)
    videos: list[dict] = []
    for order in range(1, 10):
        filename = f"{order:02d}_video.mp4"
        video = _write(root / "videos" / filename, f"video-{order}\n")
        poster_rel = f"posters/{order:02d}.jpg"
        metrics_rel = f"metrics/{order:02d}.json"
        frame_map_rel = f"frame_maps/{order:02d}.csv"
        poster = _write(root / poster_rel, f"poster-{order}\n")
        metrics = _write(root / metrics_rel, "{}\n")
        frame_map = _write(root / frame_map_rel, "frame,timestamp\n")
        videos.append(
            {
                "order": order,
                "id": f"video-{order}",
                "filename": filename,
                "bytes": video.stat().st_size,
                "sha256": _sha256(video),
                "poster": poster_rel,
                "poster_sha256": _sha256(poster),
                "metrics": metrics_rel,
                "metrics_sha256": _sha256(metrics),
                "frame_map": frame_map_rel,
                "frame_map_sha256": _sha256(frame_map),
            }
        )

    calibration = _write(root / "calibration/camera_to_world.json", "{}\n")
    workbench_metadata = _write(
        root / "calibration-workbench/take007_alignment.json", "{}\n"
    )
    clean_rgb = _write(
        root / "calibration-workbench/take007_clean_rgb.mp4", b"clean-rgb\n"
    )
    mocap = _write(root / "calibration-workbench/take007_mocap.f32", b"mocap\n")
    solved = _write(root / "calibration-workbench/take007_solved.f32", b"solved\n")
    manual_profile = _write(
        root / "calibration-workbench/applied_manual_profile.json", "{}\n"
    )
    manifest = {
        "schema": build_site.FINAL_DELIVERY_SCHEMA,
        "status": "pass",
        "video_count": 9,
        "videos": videos,
        "calibration": {
            "path": "calibration/camera_to_world.json",
            "sha256": _sha256(calibration),
        },
        "calibration_workbench": {
            "path": "calibration-workbench/take007_alignment.json",
            "sha256": _sha256(workbench_metadata),
            "clean_rgb": {
                "path": "calibration-workbench/take007_clean_rgb.mp4",
                "sha256": _sha256(clean_rgb),
            },
            "mocap_trajectory": {
                "path": "calibration-workbench/take007_mocap.f32",
                "sha256": _sha256(mocap),
            },
            "solved_trajectory": {
                "path": "calibration-workbench/take007_solved.f32",
                "sha256": _sha256(solved),
            },
            "applied_manual_profile": {
                "path": "calibration-workbench/applied_manual_profile.json",
                "sha256": _sha256(manual_profile),
            },
        },
    }
    _write(root / "README.md", "synthetic delivery\n")
    _write(
        root / "validation.json",
        json.dumps(
            {
                "schema": "gt_calib.delivery_validation.v1",
                "status": "pass",
                "video_count": 9,
                "failures": [],
            }
        )
        + "\n",
    )
    _write(root / "manifest.json", json.dumps(manifest) + "\n")
    checksum_lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksum_lines.append(
                f"{_sha256(path)}  {path.relative_to(root).as_posix()}"
            )
    _write(root / "SHA256SUMS.txt", "\n".join(checksum_lines) + "\n")
    return root


def _synthetic_public(root: Path) -> Path:
    entries: list[dict] = []
    for relative in build_site.AUTHORED_PUBLIC_ASSETS:
        path = _write(root / relative, f"authored:{relative}\n")
        entries.append(
            {
                "path": relative,
                "source": f"web/public/{relative}",
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    site_data = _write(root / "data/site-data.json", "{}\n")
    entries.append(
        {
            "path": "data/site-data.json",
            "source": "synthetic",
            "bytes": site_data.stat().st_size,
            "sha256": _sha256(site_data),
        }
    )
    _write(
        root / build_site.ASSET_MANIFEST_RELATIVE,
        json.dumps(
            {
                "schema": "gt-calib.asset-manifest.v1",
                "generatedAt": "2026-08-31T00:00:00+00:00",
                "cloudflareMaxAssetBytes": build_site.MAX_STATIC_ASSET_BYTES,
                "assets": entries,
            }
        )
        + "\n",
    )
    return root


class FinalDeliveryWebMirrorTests(unittest.TestCase):
    def test_valid_delivery_replaces_existing_tree_after_staged_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            source = _synthetic_delivery(temporary_path / "source")
            destination = temporary_path / "public/downloads/final-nine"
            _write(destination / "stale.txt", "stale\n")

            entries = build_site.mirror_final_delivery(source, destination)

            self.assertFalse((destination / "stale.txt").exists())
            self.assertEqual(
                json.loads((destination / "manifest.json").read_text())["video_count"],
                9,
            )
            self.assertEqual(
                {entry["path"] for entry in entries},
                {
                    f"downloads/final-nine/{path.relative_to(source).as_posix()}"
                    for path in source.rglob("*")
                    if path.is_file()
                },
            )
            self.assertFalse(any(path.is_symlink() for path in destination.rglob("*")))
            self.assertEqual(
                list(destination.parent.glob(".final-nine-publish-*")), []
            )

    def test_hash_failure_preserves_previous_complete_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            source = _synthetic_delivery(temporary_path / "source")
            destination = temporary_path / "public/downloads/final-nine"
            sentinel = _write(destination / "sentinel.txt", "previous-complete\n")
            # Preserve the declared byte count so this exercises the digest
            # gate rather than the earlier size gate.
            (source / "videos/01_video.mp4").write_bytes(b"TAMPER!\n")

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                build_site.mirror_final_delivery(source, destination)

            self.assertEqual(sentinel.read_text(), "previous-complete\n")
            self.assertEqual(
                list(destination.parent.glob(".final-nine-publish-*")), []
            )

    def test_symlink_and_oversized_assets_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            source = _synthetic_delivery(temporary_path / "source")
            destination = temporary_path / "public/downloads/final-nine"
            sentinel = _write(destination / "sentinel.txt", "previous-complete\n")
            (source / "unexpected-link").symlink_to(source / "README.md")

            with self.assertRaisesRegex(ValueError, "Symlinks are forbidden"):
                build_site.mirror_final_delivery(source, destination)
            self.assertEqual(sentinel.read_text(), "previous-complete\n")

            (source / "unexpected-link").unlink()
            with mock.patch.object(build_site, "MAX_STATIC_ASSET_BYTES", 4):
                with self.assertRaisesRegex(ValueError, "limit exceeded"):
                    build_site.mirror_final_delivery(source, destination)
            self.assertEqual(sentinel.read_text(), "previous-complete\n")

    def test_clean_clone_assembly_needs_only_tracked_public_and_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            public = _synthetic_public(temporary_path / "web/public")
            source = _synthetic_delivery(temporary_path / "final_9_video_delivery")

            output = build_site.assemble_deploy_site(public, source)

            self.assertEqual(output, public.resolve())
            mirror = public / "downloads/final-nine"
            self.assertEqual(len(list((mirror / "videos").glob("*.mp4"))), 9)
            payload = json.loads(
                (public / build_site.ASSET_MANIFEST_RELATIVE).read_text()
            )
            self.assertEqual(
                payload["deploymentAssembly"]["mode"],
                "tracked_public_plus_tracked_final_delivery",
            )
            inventoried = {item["path"] for item in payload["assets"]}
            actual = {
                path.relative_to(public).as_posix()
                for path in public.rglob("*")
                if path.is_file()
                and path.relative_to(public).as_posix()
                != build_site.ASSET_MANIFEST_RELATIVE
            }
            self.assertEqual(inventoried, actual)

    def test_corrupt_tracked_asset_stops_before_creating_delivery_mirror(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            public = _synthetic_public(temporary_path / "web/public")
            source = _synthetic_delivery(temporary_path / "final_9_video_delivery")
            (public / "index.html").write_text("tampered\n")

            with self.assertRaisesRegex(ValueError, "byte mismatch|SHA-256 mismatch"):
                build_site.assemble_deploy_site(public, source)

            self.assertFalse((public / "downloads/final-nine").exists())


if __name__ == "__main__":
    unittest.main()
