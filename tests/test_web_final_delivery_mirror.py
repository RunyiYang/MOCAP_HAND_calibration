from __future__ import annotations

import copy
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


def _synthetic_anatomical_ik_receipt() -> dict:
    def distribution(*, maximum: float, p95: float | None = None) -> dict:
        selected_p95 = maximum if p95 is None else p95
        return {
            "count": 100,
            "mean": selected_p95 / 2.0,
            "median": selected_p95 / 2.0,
            "p95": selected_p95,
            "max": maximum,
            "rmse": selected_p95 / 2.0,
        }

    validation = {
        "solver": build_site.IMU_VISUAL_LAB_ANATOMICAL_SOLVER,
        "dip_to_pip_flexion_ratio": build_site.IMU_VISUAL_LAB_ANATOMICAL_RATIO,
        "active_bend_threshold_deg": (
            build_site.IMU_VISUAL_LAB_ANATOMICAL_THRESHOLD_DEG
        ),
        "opposite_bend_count": 0,
        "active_bend_pair_count": 100,
        "opposite_bend_fraction_active_gt_5deg": 0.0,
        "pip_flexion_deg": distribution(maximum=90.0, p95=85.0),
        "dip_flexion_deg": distribution(maximum=60.0, p95=55.0),
        "thumb_flexion_deg": distribution(maximum=100.0, p95=95.0),
        "fixed_bone_length_drift_mm": distribution(maximum=1e-10, p95=1e-11),
        "smoothed_cmm_base_tip_endpoint_epe_mm": distribution(
            maximum=0.1, p95=0.05
        ),
        "bend_plane_temporal_delta_deg": distribution(maximum=10.0, p95=5.0),
        "limits": dict(build_site.IMU_VISUAL_LAB_ANATOMICAL_LIMITS),
        "acceptance": dict(build_site.IMU_VISUAL_LAB_ANATOMICAL_ACCEPTANCE),
    }
    return {
        side: {"anatomical_validation": copy.deepcopy(validation)}
        for side in ("left", "right")
    }


def _synthetic_imu_visual_lab(root: Path) -> Path:
    root.mkdir(parents=True)
    method_ids = build_site.IMU_VISUAL_LAB_METHOD_IDS
    videos: list[dict] = []
    order = 0
    for take_id, source, source_first, frame_count in build_site.IMU_VISUAL_LAB_SEGMENTS:
        for method_id in method_ids:
            order += 1
            stem = f"{order:02d}_{take_id}_{method_id}"
            filename = f"{stem}.mp4"
            video = _write(root / "videos" / filename, _syntactic_faststart_mp4())
            poster_rel = f"posters/{stem}.jpg"
            metrics_rel = f"metrics/{stem}.json"
            poster = _write(root / poster_rel, f"poster:{stem}\n")
            reference_median = float(order)
            reference_p95 = float(order) + 0.5
            jitter_p95 = float(order) / 10.0
            metrics = _write(
                root / metrics_rel,
                json.dumps(
                    {
                        "schema": build_site.IMU_VISUAL_LAB_METRICS_SCHEMA,
                        "status": "complete",
                        "take_id": take_id,
                        "method_id": method_id,
                        "source": source,
                        "source_first": source_first,
                        "rendered_frames": frame_count,
                        "source_assets": {
                            "camera_intrinsics": (
                                build_site.IMU_VISUAL_LAB_CAMERA_INTRINSICS[take_id]
                            ),
                        },
                        "take_preparation": {
                            "cmm_guided_ik": _synthetic_anatomical_ik_receipt(),
                        },
                        "display_contract": build_site.IMU_VISUAL_LAB_DISPLAY_CONTRACT,
                        "pooled_sides": {
                            "primary_reference_epe_mm": {
                                "median": reference_median,
                                "p95": reference_p95,
                            },
                            "root_relative_high_frequency_residual_mm": {
                                "p95": jitter_p95,
                            },
                        },
                    }
                )
                + "\n",
            )
            motion_rel = f"motions/{stem}.json"
            joint_count = 62
            values_per_frame = joint_count * 3
            motion = _write(
                root / motion_rel,
                json.dumps(
                    {
                        "schema": build_site.IMU_VISUAL_LAB_MOTION_SCHEMA,
                        "take_id": take_id,
                        "method_id": method_id,
                        "frame_count": frame_count,
                        "fps": 30.0,
                        "units": "mm",
                        "encoding": {
                            "kind": "frame-major-flat-int32-json",
                            "components": "XYZ",
                            "quantum_mm": 0.1,
                            "origin_mm": [0.0, 0.0, 0.0],
                            "joint_count": joint_count,
                            "values_per_frame": values_per_frame,
                        },
                        "layers": build_site.IMU_VISUAL_LAB_MOTION_LAYERS,
                        "view": {
                            "focus": "per-frame midpoint of MOCAP left/right wrists",
                            "reference_extent_mm": float(frame_count * 100),
                            "fixed_across_methods_for_take": True,
                        },
                        "frames": [
                            [0] * values_per_frame for _frame in range(frame_count)
                        ],
                        "validation": {
                            "finite_frames": frame_count,
                            "maximum_quantization_error_mm": 0.05,
                            "status": "pass",
                        },
                    },
                    separators=(",", ":"),
                )
                + "\n",
            )
            videos.append(
                {
                    "order": order,
                    "id": f"{take_id}-{method_id}",
                    "take_id": take_id,
                    "method_id": method_id,
                    "source": source,
                    "source_first": source_first,
                    "filename": filename,
                    "poster": poster_rel,
                    "metrics": metrics_rel,
                    "motion": motion_rel,
                    "summary": {
                        "reference_median_mm": reference_median,
                        "reference_p95_mm": reference_p95,
                        "jitter_p95_mm": jitter_p95,
                        "finite_frames": frame_count,
                    },
                    "frame_count": frame_count,
                    "codec": "h264",
                    "pixel_format": "yuv420p",
                    "width": 960,
                    "height": 540,
                    "fps": 30,
                    "bytes": video.stat().st_size,
                    "sha256": _sha256(video),
                    "poster_sha256": _sha256(poster),
                    "metrics_sha256": _sha256(metrics),
                    "motion_bytes": motion.stat().st_size,
                    "motion_sha256": _sha256(motion),
                    "faststart": True,
                }
            )

    _write(root / "README.md", "synthetic IMU visualization lab\n")
    _write(root / "index.html", "<html></html>\n")
    _write(root / "styles.css", "body{}\n")
    _write(root / "app.js", "'use strict';\n")
    _write(
        root / "manifest.json",
        json.dumps(
            {
                "schema": build_site.IMU_VISUAL_LAB_SCHEMA,
                "status": "pass",
                "take_count": 4,
                "method_count": 7,
                "video_count": 28,
                "dataset_scope": build_site.IMU_VISUAL_LAB_DATASET_SCOPE,
                "methods": [{"id": method_id} for method_id in method_ids],
                "videos": videos,
            }
        )
        + "\n",
    )
    _write(
        root / "validation.json",
        json.dumps(
            {
                "schema": build_site.IMU_VISUAL_LAB_VALIDATION_SCHEMA,
                "status": "pass",
                "video_count": 28,
                "full_decode": True,
                "failures": [],
            }
        )
        + "\n",
    )
    _refresh_imu_visual_lab_hashes(root)
    return root


def _refresh_imu_visual_lab_hashes(
    root: Path,
    *,
    refresh_receipt: bool = True,
) -> None:
    """Refresh declared asset hashes and the exact checksum closure."""

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["videos"]:
        video = root / "videos" / item["filename"]
        poster = root / item["poster"]
        metrics = root / item["metrics"]
        motion = root / item["motion"]
        item["bytes"] = video.stat().st_size
        item["sha256"] = _sha256(video)
        item["poster_sha256"] = _sha256(poster)
        item["metrics_sha256"] = _sha256(metrics)
        item["motion_bytes"] = motion.stat().st_size
        item["motion_sha256"] = _sha256(motion)
    manifest_path.write_text(json.dumps(manifest) + "\n")
    if refresh_receipt:
        _write(
            root / "validation.json",
            json.dumps(
                {
                    "schema": build_site.IMU_VISUAL_LAB_VALIDATION_SCHEMA,
                    "status": "pass",
                    "video_count": 28,
                    "full_decode": True,
                    "manifest_sha256": _sha256(manifest_path),
                    "videos": [
                        {
                            "id": item["id"],
                            "sha256": item["sha256"],
                            "frame_count": item["frame_count"],
                            "codec": item["codec"],
                            "pixel_format": item["pixel_format"],
                            "width": item["width"],
                            "height": item["height"],
                            "fps": float(item["fps"]),
                            "decoded": True,
                        }
                        for item in manifest["videos"]
                    ],
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

    def test_visual_lab_validates_and_atomically_replaces_existing_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            source = _synthetic_imu_visual_lab(temporary_path / "lab")
            destination = temporary_path / "public/downloads/imu-visual-lab"
            _write(destination / "stale.txt", "stale\n")

            entries = build_site.mirror_imu_visual_lab(source, destination)

            self.assertFalse((destination / "stale.txt").exists())
            self.assertEqual(len(list((destination / "videos").glob("*.mp4"))), 28)
            self.assertEqual(len(entries), 119)
            self.assertEqual(
                {entry["path"] for entry in entries},
                {
                    f"downloads/imu-visual-lab/{path.relative_to(source).as_posix()}"
                    for path in source.rglob("*")
                    if path.is_file()
                },
            )
            self.assertFalse(any(path.is_symlink() for path in destination.rglob("*")))
            self.assertEqual(
                list(destination.parent.glob(".imu-visual-lab-publish-*")), []
            )

    def test_visual_lab_rejects_unreferenced_file_even_when_checksummed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            _write(source / "unreferenced.json", "{}\n")
            _refresh_imu_visual_lab_hashes(source)

            with self.assertRaisesRegex(ValueError, "file inventory mismatch"):
                build_site.validate_imu_visual_lab_for_web(source)

    def test_visual_lab_rejects_incomplete_matrix_and_wrong_media_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["videos"][1]["method_id"] = manifest["videos"][0]["method_id"]
            manifest_path.write_text(json.dumps(manifest) + "\n")
            _refresh_imu_visual_lab_hashes(source)
            with self.assertRaisesRegex(ValueError, "canonical.*identity"):
                build_site.validate_imu_visual_lab_for_web(source)

        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["videos"][0]["width"] = 959
            manifest_path.write_text(json.dumps(manifest) + "\n")
            _refresh_imu_visual_lab_hashes(source)
            with self.assertRaisesRegex(ValueError, "browser media contract"):
                build_site.validate_imu_visual_lab_for_web(source)

    def test_visual_lab_rejects_noncanonical_manifest_methods_and_entry_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["methods"][0], manifest["methods"][1] = (
                manifest["methods"][1],
                manifest["methods"][0],
            )
            manifest_path.write_text(json.dumps(manifest) + "\n")
            _refresh_imu_visual_lab_hashes(source)
            with self.assertRaisesRegex(ValueError, "manifest methods.*canonical"):
                build_site.validate_imu_visual_lab_for_web(source)

        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["videos"][0]["id"] = "take005-not-the-method"
            manifest_path.write_text(json.dumps(manifest) + "\n")
            _refresh_imu_visual_lab_hashes(source)
            with self.assertRaisesRegex(ValueError, "canonical.*identity"):
                build_site.validate_imu_visual_lab_for_web(source)

        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["videos"][0]["take_id"] = "take99"
            manifest_path.write_text(json.dumps(manifest) + "\n")
            _refresh_imu_visual_lab_hashes(source)
            with self.assertRaisesRegex(ValueError, "canonical.*identity"):
                build_site.validate_imu_visual_lab_for_web(source)

    def test_visual_lab_rejects_metrics_contract_and_summary_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            manifest = json.loads((source / "manifest.json").read_text())
            metrics_path = source / manifest["videos"][0]["metrics"]
            metrics = json.loads(metrics_path.read_text())
            metrics["display_contract"]["connector_or_error_lines"] = True
            metrics_path.write_text(json.dumps(metrics) + "\n")
            _refresh_imu_visual_lab_hashes(source)
            with self.assertRaisesRegex(ValueError, "metrics contract failed"):
                build_site.validate_imu_visual_lab_for_web(source)

        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["videos"][0]["summary"]["reference_p95_mm"] += 1.0
            manifest_path.write_text(json.dumps(manifest) + "\n")
            _refresh_imu_visual_lab_hashes(source)
            with self.assertRaisesRegex(ValueError, "summary mismatch"):
                build_site.validate_imu_visual_lab_for_web(source)

    def test_visual_lab_rejects_missing_or_tampered_anatomical_ik_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            manifest = json.loads((source / "manifest.json").read_text())
            metrics_path = source / manifest["videos"][0]["metrics"]
            metrics = json.loads(metrics_path.read_text())
            del metrics["take_preparation"]["cmm_guided_ik"]["left"]
            metrics_path.write_text(json.dumps(metrics) + "\n")
            _refresh_imu_visual_lab_hashes(source)

            with self.assertRaisesRegex(ValueError, "anatomical IK contract missing"):
                build_site.validate_imu_visual_lab_for_web(source)

        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            manifest = json.loads((source / "manifest.json").read_text())
            metrics_path = source / manifest["videos"][0]["metrics"]
            metrics = json.loads(metrics_path.read_text())
            validation = metrics["take_preparation"]["cmm_guided_ik"]["right"][
                "anatomical_validation"
            ]
            validation["acceptance"]["no_opposite_active_bends"] = False
            metrics_path.write_text(json.dumps(metrics) + "\n")
            _refresh_imu_visual_lab_hashes(source)

            with self.assertRaisesRegex(ValueError, "anatomical IK acceptance failed"):
                build_site.validate_imu_visual_lab_for_web(source)

    def test_visual_lab_symlink_and_tamper_fail_without_replacing_mirror(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            source = _synthetic_imu_visual_lab(temporary_path / "lab")
            destination = temporary_path / "public/downloads/imu-visual-lab"
            sentinel = _write(destination / "sentinel.txt", "previous-complete\n")
            (source / "unexpected-link").symlink_to(source / "README.md")
            with self.assertRaisesRegex(ValueError, "Symlinks are forbidden"):
                build_site.mirror_imu_visual_lab(source, destination)
            self.assertEqual(sentinel.read_text(), "previous-complete\n")

            (source / "unexpected-link").unlink()
            manifest = json.loads((source / "manifest.json").read_text())
            video = source / "videos" / manifest["videos"][0]["filename"]
            payload = bytearray(video.read_bytes())
            payload[-1] ^= 0x01
            video.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                build_site.mirror_imu_visual_lab(source, destination)
            self.assertEqual(sentinel.read_text(), "previous-complete\n")

    def test_visual_lab_rejects_motion_tamper_and_invalid_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            manifest = json.loads((source / "manifest.json").read_text())
            motion = source / manifest["videos"][0]["motion"]
            payload = bytearray(motion.read_bytes())
            payload[-1] = 0x20
            motion.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                build_site.validate_imu_visual_lab_for_web(source)

        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            manifest = json.loads((source / "manifest.json").read_text())
            motion = source / manifest["videos"][0]["motion"]
            payload = json.loads(motion.read_text())
            payload["frames"][0].pop()
            motion.write_text(json.dumps(payload) + "\n")
            _refresh_imu_visual_lab_hashes(source)
            with self.assertRaisesRegex(ValueError, "motion frames contract failed"):
                build_site.validate_imu_visual_lab_for_web(source)

    def test_visual_lab_rejects_dataset_scope_and_source_window_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["dataset_scope"]["policy"] = "mixed legacy and new datasets"
            manifest_path.write_text(json.dumps(manifest) + "\n")
            _refresh_imu_visual_lab_hashes(source)

            with self.assertRaisesRegex(ValueError, "dataset_scope.*canonical"):
                build_site.validate_imu_visual_lab_for_web(source)

        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["videos"][0]["source_first"] += 1
            manifest_path.write_text(json.dumps(manifest) + "\n")
            _refresh_imu_visual_lab_hashes(source)

            with self.assertRaisesRegex(ValueError, "canonical.*identity"):
                build_site.validate_imu_visual_lab_for_web(source)

    def test_visual_lab_rejects_motion_units_and_topology_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            manifest = json.loads((source / "manifest.json").read_text())
            motion_path = source / manifest["videos"][0]["motion"]
            motion = json.loads(motion_path.read_text())
            motion["units"] = "cm"
            motion_path.write_text(json.dumps(motion) + "\n")
            _refresh_imu_visual_lab_hashes(source)

            with self.assertRaisesRegex(ValueError, "motion identity contract"):
                build_site.validate_imu_visual_lab_for_web(source)

        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            manifest = json.loads((source / "manifest.json").read_text())
            motion_path = source / manifest["videos"][0]["motion"]
            motion = json.loads(motion_path.read_text())
            motion["encoding"]["joint_count"] = 61
            motion["encoding"]["values_per_frame"] = 61 * 3
            motion["frames"] = [frame[: 61 * 3] for frame in motion["frames"]]
            motion_path.write_text(json.dumps(motion) + "\n")
            _refresh_imu_visual_lab_hashes(source)

            with self.assertRaisesRegex(ValueError, "motion encoding contract"):
                build_site.validate_imu_visual_lab_for_web(source)

        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            manifest = json.loads((source / "manifest.json").read_text())
            motion_path = source / manifest["videos"][0]["motion"]
            motion = json.loads(motion_path.read_text())
            motion["layers"][0]["chains"][0] = [0, 1]
            motion_path.write_text(json.dumps(motion) + "\n")
            _refresh_imu_visual_lab_hashes(source)

            with self.assertRaisesRegex(ValueError, "motion layers contract"):
                build_site.validate_imu_visual_lab_for_web(source)

    def test_visual_lab_rejects_frame_count_and_canonical_path_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["videos"][0]["frame_count"] += 1
            manifest_path.write_text(json.dumps(manifest) + "\n")
            _refresh_imu_visual_lab_hashes(source)

            with self.assertRaisesRegex(ValueError, "canonical.*identity"):
                build_site.validate_imu_visual_lab_for_web(source)

        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["videos"][0]["poster"] = manifest["videos"][1]["poster"]
            manifest_path.write_text(json.dumps(manifest) + "\n")
            _refresh_imu_visual_lab_hashes(source)

            with self.assertRaisesRegex(ValueError, "poster path"):
                build_site.validate_imu_visual_lab_for_web(source)

    def test_visual_lab_rejects_refreshed_video_with_stale_decode_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            manifest = json.loads((source / "manifest.json").read_text())
            video_path = source / "videos" / manifest["videos"][0]["filename"]
            payload = bytearray(video_path.read_bytes())
            payload[-1] ^= 0x01
            video_path.write_bytes(payload)
            _refresh_imu_visual_lab_hashes(source, refresh_receipt=False)

            with self.assertRaisesRegex(ValueError, "validation receipt is stale"):
                build_site.validate_imu_visual_lab_for_web(source)

    def test_visual_lab_path_escape_and_oversized_asset_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["videos"][0]["poster"] = "../escape.jpg"
            manifest_path.write_text(json.dumps(manifest) + "\n")
            with self.assertRaisesRegex(ValueError, "Unsafe final-delivery path"):
                build_site.validate_imu_visual_lab_for_web(source)

        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["videos"][0]["motion"] = manifest["videos"][0]["metrics"]
            manifest_path.write_text(json.dumps(manifest) + "\n")
            with self.assertRaisesRegex(ValueError, "Invalid.*motion path"):
                build_site.validate_imu_visual_lab_for_web(source)

        with tempfile.TemporaryDirectory() as temporary:
            source = _synthetic_imu_visual_lab(Path(temporary) / "lab")
            with mock.patch.object(build_site, "MAX_STATIC_ASSET_BYTES", 4):
                with self.assertRaisesRegex(ValueError, "limit exceeded"):
                    build_site.validate_imu_visual_lab_for_web(source)

    def test_clean_clone_assembly_needs_only_tracked_public_and_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            public = _synthetic_public(temporary_path / "web/public")
            source = _synthetic_delivery(temporary_path / "final_9_video_delivery")
            comparison = _synthetic_imu_comparison(
                temporary_path / "imu_mocap_comparison_delivery"
            )
            visual_lab = _synthetic_imu_visual_lab(
                temporary_path / "imu_mocap_visualization_lab"
            )

            output = build_site.assemble_deploy_site(
                public, source, comparison, visual_lab
            )

            self.assertEqual(output, public.resolve())
            mirror = public / "downloads/final-nine"
            self.assertEqual(len(list((mirror / "videos").glob("*.mp4"))), 9)
            comparison_mirror = public / "downloads/imu-mocap"
            self.assertEqual(
                len(list((comparison_mirror / "videos").glob("*.mp4"))), 4
            )
            visual_lab_mirror = public / "downloads/imu-visual-lab"
            self.assertEqual(
                len(list((visual_lab_mirror / "videos").glob("*.mp4"))), 28
            )
            payload = json.loads(
                (public / build_site.ASSET_MANIFEST_RELATIVE).read_text()
            )
            self.assertEqual(
                payload["deploymentAssembly"]["mode"],
                "tracked_public_plus_three_tracked_deliveries",
            )
            self.assertEqual(
                payload["deploymentAssembly"]["imuVisualLabSchema"],
                build_site.IMU_VISUAL_LAB_SCHEMA,
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
            visual_lab = _synthetic_imu_visual_lab(
                temporary_path / "imu_mocap_visualization_lab"
            )
            (public / "index.html").write_text("tampered\n")

            with self.assertRaisesRegex(ValueError, "byte mismatch|SHA-256 mismatch"):
                build_site.assemble_deploy_site(
                    public, source, comparison, visual_lab
                )

            self.assertFalse((public / "downloads/final-nine").exists())
            self.assertFalse((public / "downloads/imu-mocap").exists())
            self.assertFalse((public / "downloads/imu-visual-lab").exists())


if __name__ == "__main__":
    unittest.main()
