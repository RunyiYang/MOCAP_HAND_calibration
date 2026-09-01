from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
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


def _syntactic_faststart_mp4(*, moov_before_mdat: bool = True) -> bytes:
    """Return topology-valid ISO-BMFF bytes, not a decodable media fixture."""

    def box(kind: bytes, payload: bytes = b"") -> bytes:
        return struct.pack(">I4s", 8 + len(payload), kind) + payload

    ftyp = box(
        b"ftyp",
        b"isom" + struct.pack(">I", 0x200) + b"isomiso2avc1mp41",
    )
    moov = box(b"moov")
    mdat = box(b"mdat", b"\0")
    return ftyp + (moov + box(b"free") + mdat if moov_before_mdat else mdat + moov)


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


def _synthetic_imu_comparison(root: Path) -> Path:
    root.mkdir(parents=True)
    manual_profile = _write(
        root / "calibration/applied_manual_profile.json",
        json.dumps({"schema": "gt_calib.final_nine_manual_xyz.v1"}) + "\n",
    )
    videos: list[dict] = []
    for order, video_id, filename, frame_count, metrics_schema in (
        build_site.IMU_COMPARISON_VIDEO_CONTRACTS
    ):
        video = _write(root / "videos" / filename, _syntactic_faststart_mp4())
        poster_rel = f"posters/{order:02d}.jpg"
        metrics_rel = f"metrics/{order:02d}.json"
        frame_map_rel = f"frame_maps/{order:02d}.csv"
        poster = _write(root / poster_rel, f"poster-{order}\n")
        old_take = metrics_schema.endswith("skeleton_bvh.v1")
        strict_key = (
            "strict_25ms_scientific_subset"
            if old_take
            else "evaluation_strict_20ms_camera_valid_subset"
        )
        tip_key = "non_thumb_fingertip_epe_mm" if old_take else "five_tip_epe_mm"
        summary = {
            side: {
                "strict_frames": 1,
                "joint_median_mm": 10.0 + side_index,
                "joint_p95_mm": 20.0 + side_index,
                "tip_median_mm": 30.0 + side_index,
            }
            for side_index, side in enumerate(("left", "right"))
        }
        metrics_payload = {
            "schema": metrics_schema,
            "status": "comparison_complete",
            "sides": {
                side: {
                    strict_key: {
                        "frame_count": values["strict_frames"],
                        "pooled_joint_epe_mm": {
                            "median": values["joint_median_mm"],
                            "p95": values["joint_p95_mm"],
                        },
                        tip_key: {"median": values["tip_median_mm"]},
                    }
                }
                for side, values in summary.items()
            },
        }
        metrics = _write(
            root / metrics_rel,
            json.dumps(metrics_payload) + "\n",
        )
        calibration_stop = round(frame_count * 0.20)
        rows = [
            "output_frame,side,phase,solver_sample_index,solver_sample_age_ms,strict_timing_valid"
        ]
        for frame_index in range(frame_count):
            phase = (
                "calibration"
                if not old_take and frame_index < calibration_stop
                else "evaluation"
            )
            strict = int(
                frame_index == (0 if old_take else calibration_stop)
            )
            for side in ("left", "right"):
                rows.append(
                    f"{frame_index},{side},{phase},{frame_index},0.0,{strict}"
                )
        frame_map = _write(root / frame_map_rel, "\n".join(rows) + "\n")
        videos.append(
            {
                "order": order,
                "id": video_id,
                "filename": filename,
                "bytes": video.stat().st_size,
                "sha256": _sha256(video),
                "poster": poster_rel,
                "poster_sha256": _sha256(poster),
                "metrics": metrics_rel,
                "metrics_sha256": _sha256(metrics),
                "frame_map": frame_map_rel,
                "frame_map_sha256": _sha256(frame_map),
                "codec": "h264",
                "pixel_format": "yuv420p",
                "faststart": True,
                "frame_count": frame_count,
                "summary": summary,
            }
        )
    manifest = {
        "schema": build_site.IMU_COMPARISON_SCHEMA,
        "status": "pass",
        "comparison_count": 4,
        "canonical_final_nine_unchanged": True,
        "display_policy": {
            "sampling": "nearest observed solver row",
            "pose_interpolation": False,
            "pose_smoothing": False,
            "validity_gate_hides_pose": False,
            "stale_pose_is_drawn": True,
            "stale_pose_in_strict_metrics": False,
        },
        "manual_profile": {
            "path": "calibration/applied_manual_profile.json",
            "bytes": manual_profile.stat().st_size,
            "sha256": _sha256(manual_profile),
            "source": {
                "path": "calibration_profiles/operator_y_minus_44_all_hand_overlays.v1.json",
                "bytes": manual_profile.stat().st_size,
                "sha256": _sha256(manual_profile),
            },
        },
        "videos": videos,
    }
    _write(root / "README.md", "synthetic IMU comparison\n")
    _write(root / "index.html", "<html></html>\n")
    _write(root / "styles.css", "body{}\n")
    _write(root / "manifest.json", json.dumps(manifest) + "\n")
    _write(
        root / "validation.json",
        json.dumps(
            {
                "schema": "gt_calib.imu_mocap_comparison_validation.v1",
                "status": "pass",
                "comparison_count": 4,
                "full_decode": True,
                "failures": [],
            }
        )
        + "\n",
    )
    checksum_lines = [
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS.txt"
    ]
    _write(root / "SHA256SUMS.txt", "\n".join(checksum_lines) + "\n")
    return root


def _refresh_imu_comparison_hashes(root: Path) -> None:
    """Rebind a deliberately mutated synthetic bundle for semantic negatives."""

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["videos"]:
        video = root / "videos" / item["filename"]
        poster = root / item["poster"]
        metrics = root / item["metrics"]
        frame_map = root / item["frame_map"]
        item["bytes"] = video.stat().st_size
        item["sha256"] = _sha256(video)
        item["poster_sha256"] = _sha256(poster)
        item["metrics_sha256"] = _sha256(metrics)
        item["frame_map_sha256"] = _sha256(frame_map)
    manifest_path.write_text(json.dumps(manifest) + "\n")
    checksum_lines = [
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS.txt"
    ]
    (root / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n")


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

    def test_imu_comparison_rejects_text_disguised_as_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_comparison(Path(temporary) / "comparison")
            manifest = json.loads((source / "manifest.json").read_text())
            (source / "videos" / manifest["videos"][0]["filename"]).write_text(
                "not an mp4\n"
            )
            _refresh_imu_comparison_hashes(source)

            with self.assertRaisesRegex(ValueError, "MP4 box|missing ftyp"):
                build_site.validate_imu_comparison_for_web(source)

    def test_imu_comparison_rejects_moov_after_mdat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_comparison(Path(temporary) / "comparison")
            manifest = json.loads((source / "manifest.json").read_text())
            video = source / "videos" / manifest["videos"][0]["filename"]
            video.write_bytes(_syntactic_faststart_mp4(moov_before_mdat=False))
            _refresh_imu_comparison_hashes(source)

            with self.assertRaisesRegex(ValueError, "moov box does not precede mdat"):
                build_site.validate_imu_comparison_for_web(source)

    def test_imu_comparison_rejects_manual_profile_source_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_comparison(Path(temporary) / "comparison")
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["manual_profile"]["source"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest) + "\n")
            _refresh_imu_comparison_hashes(source)

            with self.assertRaisesRegex(ValueError, "source provenance is invalid"):
                build_site.validate_imu_comparison_for_web(source)

    def test_imu_comparison_rejects_noncanonical_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_comparison(Path(temporary) / "comparison")
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["videos"][0]["id"] = "arbitrary-comparison"
            manifest_path.write_text(json.dumps(manifest) + "\n")
            _refresh_imu_comparison_hashes(source)

            with self.assertRaisesRegex(ValueError, "Canonical IMU comparison identity"):
                build_site.validate_imu_comparison_for_web(source)

    def test_imu_comparison_rejects_metrics_summary_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_comparison(Path(temporary) / "comparison")
            manifest = json.loads((source / "manifest.json").read_text())
            metrics_path = source / manifest["videos"][0]["metrics"]
            metrics = json.loads(metrics_path.read_text())
            strict = metrics["sides"]["left"]["strict_25ms_scientific_subset"]
            strict["pooled_joint_epe_mm"]["median"] = 999.0
            metrics_path.write_text(json.dumps(metrics) + "\n")
            _refresh_imu_comparison_hashes(source)

            with self.assertRaisesRegex(ValueError, "manifest summary mismatch"):
                build_site.validate_imu_comparison_for_web(source)

    def test_imu_comparison_rejects_frame_map_strict_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_comparison(Path(temporary) / "comparison")
            manifest = json.loads((source / "manifest.json").read_text())
            frame_map_path = source / manifest["videos"][0]["frame_map"]
            text = frame_map_path.read_text()
            text = text.replace(
                "0,left,evaluation,0,0.0,1",
                "0,left,evaluation,0,0.0,0",
                1,
            )
            frame_map_path.write_text(text)
            _refresh_imu_comparison_hashes(source)

            with self.assertRaisesRegex(ValueError, "strict CSV counts mismatch"):
                build_site.validate_imu_comparison_for_web(source)

    def test_clean_clone_assembly_needs_only_tracked_public_and_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            public = _synthetic_public(temporary_path / "web/public")
            source = _synthetic_delivery(temporary_path / "final_9_video_delivery")
            comparison = _synthetic_imu_comparison(
                temporary_path / "imu_mocap_comparison_delivery"
            )

            output = build_site.assemble_deploy_site(public, source, comparison)

            self.assertEqual(output, public.resolve())
            mirror = public / "downloads/final-nine"
            self.assertEqual(len(list((mirror / "videos").glob("*.mp4"))), 9)
            comparison_mirror = public / "downloads/imu-mocap"
            self.assertEqual(
                len(list((comparison_mirror / "videos").glob("*.mp4"))), 4
            )
            payload = json.loads(
                (public / build_site.ASSET_MANIFEST_RELATIVE).read_text()
            )
            self.assertEqual(
                payload["deploymentAssembly"]["mode"],
                "tracked_public_plus_two_tracked_deliveries",
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
            comparison = _synthetic_imu_comparison(
                temporary_path / "imu_mocap_comparison_delivery"
            )
            (public / "index.html").write_text("tampered\n")

            with self.assertRaisesRegex(ValueError, "byte mismatch|SHA-256 mismatch"):
                build_site.assemble_deploy_site(public, source, comparison)

            self.assertFalse((public / "downloads/final-nine").exists())
            self.assertFalse((public / "downloads/imu-mocap").exists())


if __name__ == "__main__":
    unittest.main()
