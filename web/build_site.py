#!/usr/bin/env python3
"""Build the allow-listed Cloudflare static delivery directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = Path(__file__).resolve().parent
PUBLIC_ROOT = WEB_ROOT / "public"
MAX_STATIC_ASSET_BYTES = 25 * 1024 * 1024
FINAL_DELIVERY_SCHEMA = "gt_calib.final_nine_video_delivery.v1"
FINAL_DELIVERY_SOURCE = PROJECT_ROOT / "final_9_video_delivery"
FINAL_DELIVERY_PUBLIC = PUBLIC_ROOT / "downloads/final-nine"
FINAL_DELIVERY_WEB_PREFIX = "downloads/final-nine/"
ASSET_MANIFEST_RELATIVE = "data/asset-manifest.json"
AUTHORED_PUBLIC_ASSETS = (
    "index.html",
    "styles.css",
    "app.js",
    "404.html",
    "favicon.svg",
    "_headers",
    "bvh/index.html",
    "bvh/bvh-viewer.css",
    "bvh/bvh-viewer.js",
    "final-nine/index.html",
    "final-nine/styles.css",
    "final-nine/app.js",
)


def _load_source_build_dependencies() -> None:
    """Load raw-data rebuild dependencies only for ``--source`` builds.

    Deployment assembly intentionally remains standard-library-only so a
    Cloudflare Git checkout does not need UV, OpenCV, NumPy, ffmpeg, or any
    ignored acquisition/output directory.
    """

    global DEPTH_OVERLAY_SCHEMA, depth_output_stem
    global BVH_DATASET_ROOT, BVH_OUTPUT_DIR, BVH_EXPORT_SCHEMA
    global BVH_FORMAL_SEGMENTS, BVH_MANIFEST_SCHEMA
    global discover_formal_take, export_bvh_segments

    if "depth_output_stem" in globals():
        return
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from depth_mocap_overlay import DEPTH_OVERLAY_SCHEMA as depth_schema
    from depth_mocap_overlay import depth_output_stem as output_stem
    from bvh_web_export import (
        DEFAULT_DATASET_ROOT as dataset_root,
        DEFAULT_OUTPUT_DIR as output_dir,
        EXPORT_SCHEMA as export_schema,
        FORMAL_SEGMENTS as formal_segments,
        MANIFEST_SCHEMA as manifest_schema,
        discover_formal_take as discover_take,
        export_segments as export_segments,
    )

    DEPTH_OVERLAY_SCHEMA = depth_schema
    depth_output_stem = output_stem
    BVH_DATASET_ROOT = dataset_root
    BVH_OUTPUT_DIR = output_dir
    BVH_EXPORT_SCHEMA = export_schema
    BVH_FORMAL_SEGMENTS = formal_segments
    BVH_MANIFEST_SCHEMA = manifest_schema
    discover_formal_take = discover_take
    export_bvh_segments = export_segments


DEPTH_OUTPUT_DIR = PROJECT_ROOT / "outputs/depth_mocap_alignment_review"
MOCAP_RAW_ALIGNMENT_DIR = PROJECT_ROOT / "outputs/mocap_video_alignment_review"
MOCAP_REAR_ALIGNMENT_DIR = PROJECT_ROOT / "outputs/mocap_video_rear_mount_review"
MOCAP_WRIST_SEMANTICS_DIR = PROJECT_ROOT / "outputs/mocap_wrist_semantics_audit"
EXPECTED_DEPTH_COMMON_FRAMES = {
    "01_210814_Take_000": 1747,
    "02_210955_Take_001": 1804,
    "03_211139_Take_002": 1799,
}
DEPTH_ALIGNMENT_FIELDS = (
    "output_frame",
    "source_depth_frame",
    "bag_message_timestamp_ns",
    "depth_device_timestamp_us",
    "depth_frame_number",
    "source_bag_elapsed_s",
    "matched_clip_bag_elapsed_s",
    "same_index_rgb_device_timestamp_us",
    "depth_minus_rgb_device_timestamp_ms",
    "cmavatar_target_timestamp_ns",
    "mocap_low_frame_counter",
    "mocap_high_frame_counter",
    "mocap_low_timestamp_ns",
    "mocap_high_timestamp_ns",
    "interpolation_alpha",
    "bracket_span_ms",
    "nearest_mocap_sample_delta_ms",
    "raw_valid_pixel_fraction",
    "projected_joint_inside_count",
    "projected_joint_nonzero_depth_count",
    "left_wrist_depth_x",
    "left_wrist_depth_y",
    "left_wrist_inside_frame",
    "right_wrist_depth_x",
    "right_wrist_depth_y",
    "right_wrist_inside_frame",
)

TAKES = (
    {
        "id": "take-000",
        "segment": "01_210814_Take_000",
        "label": "Take 01",
        "title": "canonical 配准来源，不是独立测试",
        "diagnosticTitle": "旧模式配准来源，不是独立测试",
        "role": "calibration",
        "roleZh": "标定集",
        "summary": "本段在 wrist-local frame 拟合 canonical rotation + scale；它不是独立测试。",
        "metrics": "outputs/mocap_root_fusion_strict25/01_210814_Take_000_mocap_root_fused_glove_calibration_fit.metrics.json",
        "video": "outputs/mocap_root_fusion_review/01_210814_Take_000_mocap_root_fused_glove_calibration_fit_h264.mp4",
        "poster": "outputs/mocap_root_fusion_review/01_210814_Take_000_mocap_root_fused_glove_calibration_fit.contact.jpg",
        "diagnosticSummary": "保留 glove wrist orientation，只共享 MOCAP wrist translation 的旧诊断基线。",
        "diagnosticMetrics": "outputs/holdout_01_strict25/01_210814_Take_000_glove_vs_mocap_calibration_fit.metrics.json",
        "diagnosticVideo": "outputs/holdout_01_review/01_210814_Take_000_glove_vs_mocap_calibration_fit_h264.mp4",
        "diagnosticPoster": "outputs/holdout_01_review/01_210814_Take_000_glove_vs_mocap_calibration_fit.contact.jpg",
        "mocapTitle": "后腕模块 proxy · 1748 帧动作匹配",
        "mocapSummary": "每手原始 21 joints 完全不变；正式视频里带白色中心的同侧彩色圆点是 rear wrist module proxy，只用于 VIZ、不是 GT。去除约 255 ms capture→poll 延迟后，动作仍按同一逐帧映射投影。",
        "mocapMetrics": "outputs/mocap_video_rear_mount_review/01_210814_Take_000_mocap_video_aligned_strict25.metrics.json",
        "mocapAlignment": "outputs/mocap_video_rear_mount_review/01_210814_Take_000_mocap_video_aligned_strict25.alignment.csv",
        "mocapVideo": "outputs/mocap_video_rear_mount_review/01_210814_Take_000_mocap_video_aligned_strict25_h264.mp4",
        "mocapPoster": "outputs/mocap_video_rear_mount_review/01_210814_Take_000_mocap_video_aligned_strict25.contact.jpg",
        "mocapRawMetrics": "outputs/mocap_video_alignment_review/01_210814_Take_000_mocap_video_aligned_strict25.metrics.json",
        "mocapRawAlignment": "outputs/mocap_video_alignment_review/01_210814_Take_000_mocap_video_aligned_strict25.alignment.csv",
        "mocapRawVideo": "outputs/mocap_video_alignment_review/01_210814_Take_000_mocap_video_aligned_strict25_h264.mp4",
        "mocapRawPoster": "outputs/mocap_video_alignment_review/01_210814_Take_000_mocap_video_aligned_strict25.contact.jpg",
        "depthTitle": "1747 帧 raw depth–MOCAP 投影",
        "depthSummary": "直接解码 mono16 毫米深度，并在每个 depth 自身采集时刻插值 MOCAP；本视图只检查时序、SE(3) 投影和深度可用覆盖。",
    },
    {
        "id": "take-001",
        "segment": "02_210955_Take_001",
        "label": "Take 02",
        "title": "冻结参数后的独立 validation holdout",
        "diagnosticTitle": "旧模式右手开始退化",
        "role": "validation_holdout",
        "roleZh": "验证留出集",
        "summary": "冻结 Take 01 canonical 参数；MOCAP wrist SE(3) 条件下左右手 median 均约 11.5 mm。",
        "metrics": "outputs/mocap_root_fusion_strict25/02_210955_Take_001_mocap_root_fused_glove_holdout_from_Take_000.metrics.json",
        "video": "outputs/mocap_root_fusion_review/02_210955_Take_001_mocap_root_fused_glove_holdout_from_Take_000_h264.mp4",
        "poster": "outputs/mocap_root_fusion_review/02_210955_Take_001_mocap_root_fused_glove_holdout_from_Take_000.contact.jpg",
        "diagnosticSummary": "旧模式显示右手退化，主要混入 wrist orientation/session-neutral 差异。",
        "diagnosticMetrics": "outputs/holdout_01_strict25/02_210955_Take_001_glove_vs_mocap_holdout_from_Take_000.metrics.json",
        "diagnosticVideo": "outputs/holdout_01_review/02_210955_Take_001_glove_vs_mocap_holdout_from_Take_000_h264.mp4",
        "diagnosticPoster": "outputs/holdout_01_review/02_210955_Take_001_glove_vs_mocap_holdout_from_Take_000.contact.jpg",
        "mocapTitle": "后腕模块 proxy · 1805 帧动作匹配",
        "mocapSummary": "每手原始 21 joints 完全不变；正式视频里带白色中心的同侧彩色圆点是 rear wrist module proxy，只用于 VIZ、不是 GT。共同区间每帧沿用原 CMAvatar bracket，曝光时刻相关性约 0.90。",
        "mocapMetrics": "outputs/mocap_video_rear_mount_review/02_210955_Take_001_mocap_video_aligned_strict25.metrics.json",
        "mocapAlignment": "outputs/mocap_video_rear_mount_review/02_210955_Take_001_mocap_video_aligned_strict25.alignment.csv",
        "mocapVideo": "outputs/mocap_video_rear_mount_review/02_210955_Take_001_mocap_video_aligned_strict25_h264.mp4",
        "mocapPoster": "outputs/mocap_video_rear_mount_review/02_210955_Take_001_mocap_video_aligned_strict25.contact.jpg",
        "mocapRawMetrics": "outputs/mocap_video_alignment_review/02_210955_Take_001_mocap_video_aligned_strict25.metrics.json",
        "mocapRawAlignment": "outputs/mocap_video_alignment_review/02_210955_Take_001_mocap_video_aligned_strict25.alignment.csv",
        "mocapRawVideo": "outputs/mocap_video_alignment_review/02_210955_Take_001_mocap_video_aligned_strict25_h264.mp4",
        "mocapRawPoster": "outputs/mocap_video_alignment_review/02_210955_Take_001_mocap_video_aligned_strict25.contact.jpg",
        "depthTitle": "1804 帧 raw depth–MOCAP 投影",
        "depthSummary": "完整共同区间使用 depth 自身时间戳；骨架落在 raw metric depth 上仅用于检查固定外参与运动一致性。",
    },
    {
        "id": "take-002",
        "segment": "03_211139_Take_002",
        "label": "Take 03",
        "title": "中位数稳定，但 P95 长尾仍高",
        "diagnosticTitle": "旧 wrist orientation 跨 session 明显发散",
        "role": "test_stress_holdout",
        "roleZh": "回顾性测试 / 压力留出集",
        "summary": "冻结 Take 01 canonical 参数；median 稳定在 13 mm 左右，但 P95 长尾仍高。",
        "metrics": "outputs/mocap_root_fusion_strict25/03_211139_Take_002_mocap_root_fused_glove_holdout_from_Take_000.metrics.json",
        "video": "outputs/mocap_root_fusion_review/03_211139_Take_002_mocap_root_fused_glove_holdout_from_Take_000_h264.mp4",
        "poster": "outputs/mocap_root_fusion_review/03_211139_Take_002_mocap_root_fused_glove_holdout_from_Take_000.contact.jpg",
        "diagnosticSummary": "旧模式显示明显发散，用于诊断 glove wrist orientation 的跨 session 漂移。",
        "diagnosticMetrics": "outputs/holdout_01_strict25/03_211139_Take_002_glove_vs_mocap_holdout_from_Take_000.metrics.json",
        "diagnosticVideo": "outputs/holdout_01_review/03_211139_Take_002_glove_vs_mocap_holdout_from_Take_000_h264.mp4",
        "diagnosticPoster": "outputs/holdout_01_review/03_211139_Take_002_glove_vs_mocap_holdout_from_Take_000.contact.jpg",
        "mocapTitle": "后腕模块 proxy · 1800 帧动作匹配",
        "mocapSummary": "每手原始 21 joints 完全不变；正式视频里带白色中心的同侧彩色圆点是 rear wrist module proxy，只用于 VIZ、不是 GT。旧 poll-time 相关性约 0.39，曝光修正后约 0.88。",
        "mocapMetrics": "outputs/mocap_video_rear_mount_review/03_211139_Take_002_mocap_video_aligned_strict25.metrics.json",
        "mocapAlignment": "outputs/mocap_video_rear_mount_review/03_211139_Take_002_mocap_video_aligned_strict25.alignment.csv",
        "mocapVideo": "outputs/mocap_video_rear_mount_review/03_211139_Take_002_mocap_video_aligned_strict25_h264.mp4",
        "mocapPoster": "outputs/mocap_video_rear_mount_review/03_211139_Take_002_mocap_video_aligned_strict25.contact.jpg",
        "mocapRawMetrics": "outputs/mocap_video_alignment_review/03_211139_Take_002_mocap_video_aligned_strict25.metrics.json",
        "mocapRawAlignment": "outputs/mocap_video_alignment_review/03_211139_Take_002_mocap_video_aligned_strict25.alignment.csv",
        "mocapRawVideo": "outputs/mocap_video_alignment_review/03_211139_Take_002_mocap_video_aligned_strict25_h264.mp4",
        "mocapRawPoster": "outputs/mocap_video_alignment_review/03_211139_Take_002_mocap_video_aligned_strict25.contact.jpg",
        "depthTitle": "1799 帧 raw depth–MOCAP 投影",
        "depthSummary": "完整共同区间按 depth timestamp 映射到 CMAvatar；非零深度覆盖只反映传感器可用像素，不代表手部位姿精度。",
    },
)

STATIC_COPIES = {
    "outputs/feishu_dataset_understanding/assets/mocap_pipeline.svg": "assets/mocap-pipeline.svg",
    "outputs/feishu_dataset_understanding/assets/dataset_map.svg": "assets/dataset-map.svg",
    "outputs/holdout_01_strict25/strict_metrics_overview.svg": "assets/strict-metrics-overview.svg",
    "outputs/mocap_root_fusion_strict25/fusion_comparison_overview.svg": "assets/fusion-comparison-overview.svg",
    "docs/dataset/DATA_REQUEST_2026-08-31.md": "downloads/DATA_REQUEST_2026-08-31.md",
    "docs/dataset/request_datalist_2026-08-31.csv": "downloads/request_datalist_2026-08-31.csv",
    "docs/protocols/CROSS_TAKE_EVALUATION.md": "downloads/CROSS_TAKE_EVALUATION.md",
    "docs/calibration/CALIBRATED_ONE_V1.md": "downloads/CALIBRATED_ONE_V1.md",
    "docs/calibration/MOCAP_ROOT_FUSION_V1.md": "downloads/MOCAP_ROOT_FUSION_V1.md",
    "docs/calibration/MOCAP_VIDEO_ALIGNMENT_V1.md": "downloads/MOCAP_VIDEO_ALIGNMENT_V1.md",
    "docs/calibration/DEPTH_MOCAP_ALIGNMENT_V1.md": "downloads/DEPTH_MOCAP_ALIGNMENT_V1.md",
    "docs/calibration/BVH_LOCAL_VIEWER_V1.md": "downloads/BVH_LOCAL_VIEWER_V1.md",
    "docs/calibration/DEPTH_CAMERA_POSE_AUDIT_V1.md": "downloads/DEPTH_CAMERA_POSE_AUDIT_V1.md",
    "docs/calibration/EXPERIMENTAL_ARTICULATION_PROBE.md": "downloads/EXPERIMENTAL_ARTICULATION_PROBE.md",
    "docs/calibration/MOCAP_WRIST_SEMANTICS_V1.md": "downloads/MOCAP_WRIST_SEMANTICS_V1.md",
    "docs/calibration/FINAL_NINE_MANUAL_XYZ_V1.md": "downloads/FINAL_NINE_MANUAL_XYZ_V1.md",
    "docs/daily/2026/2026-08-30.md": "downloads/daily-2026-08-30.md",
    "docs/daily/2026/2026-08-31.md": "downloads/daily-2026-08-31.md",
    "docs/daily/2026/2026-09-01.md": "downloads/daily-2026-09-01.md",
    "docs/evidence/index.csv": "downloads/evidence-index.csv",
    "docs/dataset/datalist.csv": "downloads/dataset-inventory.csv",
    "outputs/holdout_01_strict25/01_210814_Take_000_registration_profile.json": "downloads/registration-profile-take-000.json",
    "outputs/mocap_root_fusion_strict25/01_210814_Take_000_mocap_root_fusion_registration_profile.json": "downloads/mocap-root-fusion-profile-take-000.json",
    "outputs/depth_calibration_audit/depth_camera_pose_audit_summary.png": "assets/depth-camera-pose-audit-summary.png",
    "outputs/depth_calibration_audit/rgb_fixed_camera_registration.jpg": "assets/rgb-fixed-camera-registration.jpg",
    "outputs/depth_calibration_audit/depth_camera_pose_audit.json": "downloads/depth-camera-pose-audit.json",
    "outputs/depth_calibration_audit/d2c_rigid_candidate.json": "downloads/d2c-rigid-candidate-NOT-ACTIVE.json",
    "outputs/mocap_wrist_semantics_audit/01_01_210814_Take_000_heldout_raw-rearproxy-z37_contact.png": "assets/mocap-wrist-semantics-take-01.png",
    "outputs/mocap_wrist_semantics_audit/02_02_210955_Take_001_heldout_raw-rearproxy-z37_contact.png": "assets/mocap-wrist-semantics-take-02.png",
    "outputs/mocap_wrist_semantics_audit/03_03_211139_Take_002_heldout_raw-rearproxy-z37_contact.png": "assets/mocap-wrist-semantics-take-03.png",
    "outputs/mocap_wrist_semantics_audit/summary.json": "downloads/mocap-wrist-semantics-summary.json",
    "outputs/mocap_wrist_semantics_audit/frame_provenance.csv": "downloads/mocap-wrist-semantics-frame-provenance.csv",
    "同步整理_20260829_三段/movementcap_worldcalib_pointcloud_package_20260830/movementcap_ruler_worldcalib/results/manual_final/camera_to_world.json": "downloads/camera-to-world-delivered.json",
}

# Exact aliases produced by the first version of the site. They are derived
# copies; retaining them after the fusion/diagnostic split would leave large,
# untracked duplicates outside the asset manifest.
OBSOLETE_GENERATED_PATHS = (
    "media/take-000.mp4",
    "media/take-001.mp4",
    "media/take-002.mp4",
    "media/take-000-contact.jpg",
    "media/take-001-contact.jpg",
    "media/take-002-contact.jpg",
    "downloads/take-000-strict25.metrics.json",
    "downloads/take-001-strict25.metrics.json",
    "downloads/take-002-strict25.metrics.json",
    "downloads/take-000-gap50.metrics.json",
    "downloads/take-001-gap50.metrics.json",
    "downloads/take-002-gap50.metrics.json",
    "downloads/gt-calib-evidence-2026-08-30.zip",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_cloudflare_size(path: Path) -> None:
    size = path.stat().st_size
    if size > MAX_STATIC_ASSET_BYTES:
        raise ValueError(
            f"Cloudflare Static Assets limit exceeded: {path} is {size} bytes "
            f"(limit {MAX_STATIC_ASSET_BYTES})"
        )


def _safe_delivery_relative_path(value: object, *, field: str) -> PurePosixPath:
    """Parse one manifest/checksum path without allowing path traversal."""

    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Invalid final-delivery path in {field}: {value!r}")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or value != relative.as_posix()
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise ValueError(f"Unsafe final-delivery path in {field}: {value!r}")
    return relative


def _delivery_artifact(
    root: Path,
    relative_value: object,
    *,
    field: str,
    expected_hash: object,
    expected_bytes: object | None = None,
) -> Path:
    """Resolve and cryptographically validate one regular delivery artifact."""

    relative = _safe_delivery_relative_path(relative_value, field=field)
    artifact = root.joinpath(*relative.parts)
    if artifact.is_symlink() or not artifact.is_file():
        raise ValueError(f"Missing or non-regular final-delivery artifact: {relative}")
    ensure_cloudflare_size(artifact)
    if expected_bytes is not None:
        if type(expected_bytes) is not int or expected_bytes < 0:
            raise ValueError(f"Invalid byte count for {relative}: {expected_bytes!r}")
        if artifact.stat().st_size != expected_bytes:
            raise ValueError(
                f"Final-delivery byte mismatch for {relative}: "
                f"{artifact.stat().st_size} != {expected_bytes}"
            )
    if not isinstance(expected_hash, str) or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
        raise ValueError(f"Invalid SHA-256 for {relative}: {expected_hash!r}")
    actual_hash = sha256(artifact)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Final-delivery SHA-256 mismatch for {relative}: "
            f"{actual_hash} != {expected_hash}"
        )
    return artifact


def validate_final_delivery_for_web(source: Path) -> list[dict]:
    """Validate the tracked nine-video delivery using only the standard library.

    This deliberately does not depend on ffmpeg, UV, or the raw acquisition
    folders, so a clean Git/Cloudflare checkout can assemble the static tree.
    The already generated delivery remains the trust boundary: its manifest,
    validation record, complete checksum inventory, byte counts, and every
    regular file are checked before anything under ``web/public`` is touched.
    """

    source_input = Path(source).expanduser()
    if source_input.is_symlink():
        raise ValueError(f"Final delivery is a symlink: {source_input}")
    source = source_input.resolve()
    if not source.is_dir():
        raise ValueError(f"Final delivery is missing: {source}")
    if (source / "_render_scratch").exists():
        raise ValueError("Final delivery contains forbidden _render_scratch")

    files: dict[str, Path] = {}
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlinks are forbidden in final delivery: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Non-regular final-delivery entry: {path}")
        relative = path.relative_to(source).as_posix()
        files[relative] = path
        ensure_cloudflare_size(path)

    required = {"README.md", "manifest.json", "validation.json", "SHA256SUMS.txt"}
    missing_required = sorted(required - files.keys())
    if missing_required:
        raise ValueError(f"Final delivery is missing required files: {missing_required}")

    try:
        manifest = json.loads(files["manifest.json"].read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Final-delivery manifest is not valid UTF-8 JSON") from exc
    videos = manifest.get("videos")
    if (
        manifest.get("schema") != FINAL_DELIVERY_SCHEMA
        or manifest.get("status") != "pass"
        or manifest.get("video_count") != 9
        or not isinstance(videos, list)
        or len(videos) != 9
    ):
        raise ValueError("Final-delivery manifest must be status=pass with exactly 9 videos")

    orders: list[int] = []
    ids: list[str] = []
    video_names: list[str] = []
    for index, item in enumerate(videos):
        if not isinstance(item, dict):
            raise ValueError(f"Final-delivery video entry {index} is not an object")
        order = item.get("order")
        video_id = item.get("id")
        filename = item.get("filename")
        if type(order) is not int or not isinstance(video_id, str) or not video_id:
            raise ValueError(f"Invalid final-delivery video identity at index {index}")
        relative_name = _safe_delivery_relative_path(
            filename, field=f"videos[{index}].filename"
        )
        if len(relative_name.parts) != 1 or relative_name.suffix.lower() != ".mp4":
            raise ValueError(f"Video filename must be one MP4 basename: {filename!r}")
        orders.append(order)
        ids.append(video_id)
        video_names.append(relative_name.name)
        _delivery_artifact(
            source,
            f"videos/{relative_name.name}",
            field=f"videos[{index}].filename",
            expected_hash=item.get("sha256"),
            expected_bytes=item.get("bytes"),
        )
        for path_key, hash_key in (
            ("poster", "poster_sha256"),
            ("metrics", "metrics_sha256"),
            ("frame_map", "frame_map_sha256"),
        ):
            relative_value = item.get(path_key)
            expected_hash = item.get(hash_key)
            if relative_value is None:
                if expected_hash is not None:
                    raise ValueError(
                        f"videos[{index}].{hash_key} must be null when {path_key} is null"
                    )
                continue
            _delivery_artifact(
                source,
                relative_value,
                field=f"videos[{index}].{path_key}",
                expected_hash=expected_hash,
            )

    if sorted(orders) != list(range(1, 10)) or len(set(ids)) != 9 or len(set(video_names)) != 9:
        raise ValueError("Final-delivery video orders, IDs, and filenames must be unique 1..9")
    videos_dir = source / "videos"
    if not videos_dir.is_dir() or videos_dir.is_symlink():
        raise ValueError("Final-delivery videos directory is missing or a symlink")
    actual_video_names = {
        path.name
        for path in videos_dir.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix.lower() == ".mp4"
    }
    if actual_video_names != set(video_names) or any(
        path.is_dir() for path in videos_dir.iterdir()
    ):
        raise ValueError("Final-delivery videos directory does not match the 9-video manifest")

    calibration = manifest.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("Final-delivery calibration manifest entry is missing")
    _delivery_artifact(
        source,
        calibration.get("path"),
        field="calibration.path",
        expected_hash=calibration.get("sha256"),
    )

    workbench = manifest.get("calibration_workbench")
    if workbench is not None:
        if not isinstance(workbench, dict):
            raise ValueError("calibration_workbench must be an object")
        for label, entry in (
            ("metadata", workbench),
            ("clean_rgb", workbench.get("clean_rgb")),
            ("mocap_trajectory", workbench.get("mocap_trajectory")),
            ("solved_trajectory", workbench.get("solved_trajectory")),
        ):
            if not isinstance(entry, dict):
                raise ValueError(f"calibration_workbench.{label} is missing")
            _delivery_artifact(
                source,
                entry.get("path"),
                field=f"calibration_workbench.{label}.path",
                expected_hash=entry.get("sha256"),
            )
        manual_profile = workbench.get("applied_manual_profile")
        if manual_profile is not None:
            if not isinstance(manual_profile, dict):
                raise ValueError(
                    "calibration_workbench.applied_manual_profile must be null or an object"
                )
            _delivery_artifact(
                source,
                manual_profile.get("path"),
                field="calibration_workbench.applied_manual_profile.path",
                expected_hash=manual_profile.get("sha256"),
            )

    try:
        validation = json.loads(files["validation.json"].read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Final-delivery validation record is not valid UTF-8 JSON") from exc
    if (
        validation.get("schema") != "gt_calib.delivery_validation.v1"
        or validation.get("status") != "pass"
        or validation.get("video_count") != 9
        or validation.get("failures") != []
    ):
        raise ValueError("Final-delivery validation record is not a clean 9-video pass")

    checksum_entries: dict[str, str] = {}
    for line_number, line in enumerate(
        files["SHA256SUMS.txt"].read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"Malformed SHA256SUMS.txt line {line_number}")
        digest, value = match.groups()
        relative = _safe_delivery_relative_path(value, field=f"SHA256SUMS.txt:{line_number}")
        relative_text = relative.as_posix()
        if relative_text in checksum_entries:
            raise ValueError(f"Duplicate checksum entry: {relative_text}")
        checksum_entries[relative_text] = digest

    expected_checksum_paths = set(files) - {"SHA256SUMS.txt"}
    if set(checksum_entries) != expected_checksum_paths:
        missing = sorted(expected_checksum_paths - set(checksum_entries))
        extra = sorted(set(checksum_entries) - expected_checksum_paths)
        raise ValueError(f"Checksum inventory mismatch: missing={missing}, extra={extra}")
    for relative, expected_hash in checksum_entries.items():
        actual_hash = sha256(files[relative])
        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA256SUMS mismatch for {relative}: {actual_hash} != {expected_hash}"
            )

    return [
        {
            "path": f"downloads/final-nine/{relative}",
            "source": f"final_9_video_delivery/{relative}",
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for relative, path in sorted(files.items())
    ]


def mirror_final_delivery(source: Path, destination: Path) -> list[dict]:
    """Stage, revalidate, and swap the final delivery into ``web/public``.

    The previous complete tree is retained until the staged copy passes every
    check.  The swap uses same-filesystem directory renames with rollback, so a
    failed build can never expose a partially copied delivery.
    """

    source = Path(source).expanduser()
    destination_input = Path(destination).expanduser()
    if destination_input.is_symlink():
        raise ValueError(f"Unsafe final-delivery web destination: {destination_input}")
    destination = destination_input.resolve()
    source_entries = validate_final_delivery_for_web(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Unsafe final-delivery web destination: {destination}")

    staging_root = Path(
        tempfile.mkdtemp(prefix=".final-nine-publish-", dir=destination.parent)
    )
    staged = staging_root / "final-nine"
    backup = staging_root / "previous-final-nine"
    moved_previous = False
    try:
        shutil.copytree(source, staged, symlinks=False)
        staged_entries = validate_final_delivery_for_web(staged)
        if source_entries != staged_entries:
            raise ValueError("Staged final delivery differs from its validated source")

        if destination.exists():
            destination.rename(backup)
            moved_previous = True
        try:
            staged.rename(destination)
        except Exception:
            if moved_previous and backup.exists() and not destination.exists():
                backup.rename(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return staged_entries
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)


def validate_tracked_public_assets(public_root: Path) -> tuple[dict, list[dict]]:
    """Validate every checked-in public asset recorded by asset-manifest.

    Entries under ``downloads/final-nine`` are intentionally skipped here:
    that ignored mirror is rebuilt and independently validated from the
    tracked delivery.  Every other regular file must have a one-to-one
    manifest entry, exact byte count, and exact SHA-256.
    """

    public_input = Path(public_root).expanduser()
    if public_input.is_symlink():
        raise ValueError(f"Public root is a symlink: {public_input}")
    public_root = public_input.resolve()
    if not public_root.is_dir():
        raise ValueError(f"Public root is missing: {public_root}")
    manifest_path = public_root / ASSET_MANIFEST_RELATIVE
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"Tracked asset manifest is missing: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Tracked asset manifest is not valid UTF-8 JSON") from exc
    assets = payload.get("assets")
    if payload.get("schema") != "gt-calib.asset-manifest.v1" or not isinstance(
        assets, list
    ):
        raise ValueError("Tracked asset manifest schema/assets are invalid")

    checked_entries: list[dict] = []
    checked_paths: set[str] = set()
    for index, item in enumerate(assets):
        if not isinstance(item, dict):
            raise ValueError(f"Tracked asset entry {index} is not an object")
        relative = _safe_delivery_relative_path(
            item.get("path"), field=f"asset-manifest.assets[{index}].path"
        ).as_posix()
        if relative in checked_paths:
            raise ValueError(f"Duplicate tracked asset path: {relative}")
        if relative.startswith(FINAL_DELIVERY_WEB_PREFIX):
            continue
        if relative == ASSET_MANIFEST_RELATIVE:
            raise ValueError("Asset manifest must not recursively inventory itself")
        _delivery_artifact(
            public_root,
            relative,
            field=f"asset-manifest.assets[{index}].path",
            expected_hash=item.get("sha256"),
            expected_bytes=item.get("bytes"),
        )
        checked_paths.add(relative)
        checked_entries.append(dict(item))

    missing_authored = sorted(set(AUTHORED_PUBLIC_ASSETS) - checked_paths)
    if missing_authored:
        raise ValueError(
            f"Tracked asset manifest is missing authored assets: {missing_authored}"
        )

    actual_paths: set[str] = set()
    for path in public_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlinks are forbidden in public tree: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Non-regular public-tree entry: {path}")
        relative = path.relative_to(public_root).as_posix()
        ensure_cloudflare_size(path)
        if relative == ASSET_MANIFEST_RELATIVE or relative.startswith(
            FINAL_DELIVERY_WEB_PREFIX
        ):
            continue
        actual_paths.add(relative)
    if actual_paths != checked_paths:
        missing = sorted(checked_paths - actual_paths)
        extra = sorted(actual_paths - checked_paths)
        raise ValueError(
            f"Tracked public-tree inventory mismatch: missing={missing}, extra={extra}"
        )
    return payload, checked_entries


def _atomic_write_json(path: Path, payload: object) -> None:
    """Write JSON next to its destination and replace the old file atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".staging", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        ensure_cloudflare_size(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def assemble_deploy_site(
    public_root: Path = PUBLIC_ROOT,
    final_source: Path = FINAL_DELIVERY_SOURCE,
) -> Path:
    """Assemble a clean-clone deploy tree from tracked, prebuilt artifacts."""

    public_root = Path(public_root).expanduser().resolve()
    payload, tracked_entries = validate_tracked_public_assets(public_root)
    final_destination = public_root / "downloads/final-nine"
    final_entries = mirror_final_delivery(final_source, final_destination)

    combined_entries = tracked_entries + final_entries
    combined_paths = [str(item["path"]) for item in combined_entries]
    if len(combined_paths) != len(set(combined_paths)):
        raise ValueError("Deployment assembly produced duplicate asset paths")

    actual_paths: set[str] = set()
    for path in public_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlinks are forbidden in assembled public tree: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Non-regular assembled public-tree entry: {path}")
        ensure_cloudflare_size(path)
        relative = path.relative_to(public_root).as_posix()
        if relative != ASSET_MANIFEST_RELATIVE:
            actual_paths.add(relative)
    if actual_paths != set(combined_paths):
        missing = sorted(set(combined_paths) - actual_paths)
        extra = sorted(actual_paths - set(combined_paths))
        raise ValueError(
            f"Assembled public-tree inventory mismatch: missing={missing}, extra={extra}"
        )

    delivery_manifest = json.loads(
        (Path(final_source) / "manifest.json").read_text(encoding="utf-8")
    )
    assembled_manifest = dict(payload)
    assembled_manifest["cloudflareMaxAssetBytes"] = MAX_STATIC_ASSET_BYTES
    assembled_manifest["deploymentAssembly"] = {
        "mode": "tracked_public_plus_tracked_final_delivery",
        "finalDeliverySchema": FINAL_DELIVERY_SCHEMA,
        "finalDeliveryGeneratedAt": delivery_manifest.get("generated_at_utc"),
    }
    assembled_manifest["assets"] = sorted(
        combined_entries, key=lambda item: str(item["path"])
    )
    asset_manifest_path = public_root / ASSET_MANIFEST_RELATIVE
    _atomic_write_json(asset_manifest_path, assembled_manifest)

    published_files = [path for path in public_root.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in published_files)
    largest = max(published_files, key=lambda path: path.stat().st_size)
    print(
        f"Assembled {len(published_files)} files "
        f"({total_bytes / 1024 / 1024:.2f} MiB) under {public_root}"
    )
    print(f"Largest asset: {largest}")
    return public_root


def copy_asset(source_rel: str, destination_rel: str, manifest: list[dict]) -> None:
    source = PROJECT_ROOT / source_rel
    destination = PUBLIC_ROOT / destination_rel
    if not source.is_file():
        raise FileNotFoundError(f"Required site asset is missing: {source}")
    if source.is_symlink():
        raise ValueError(f"Refusing to publish symlink: {source}")
    ensure_cloudflare_size(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    manifest.append(
        {
            "path": destination_rel,
            "source": source_rel,
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }
    )


def metric_summary(spec: dict, *, diagnostic: bool = False) -> dict:
    prefix = "diagnostic" if diagnostic else ""
    metrics_key = f"{prefix}Metrics" if prefix else "metrics"
    summary_key = f"{prefix}Summary" if prefix else "summary"
    variant = "diagnostic" if diagnostic else "fusion"
    source = PROJECT_ROOT / spec[metrics_key]
    payload = json.loads(source.read_text(encoding="utf-8"))
    sides = payload["sides"]
    left = sides["left"]
    right = sides["right"]
    sync = payload["synchronization"]
    expected_schema = (
        "gt_calib.glove_mocap_rgb_preview.v2"
        if diagnostic
        else "gt_calib.mocap_root_conditioned_articulation_metrics.v2"
    )
    if payload.get("schema") != expected_schema:
        raise ValueError(
            f"Refusing stale {variant} metrics {source}: expected {expected_schema}, "
            f"got {payload.get('schema')}"
        )
    if "subtracted row by row" not in sync.get("timestamp_semantics", ""):
        raise ValueError(f"Refusing poll-time metrics without exposure correction: {source}")
    if "camera_capture_to_thor_poll_latency_ms" not in sync:
        raise ValueError(f"Missing capture-to-poll latency evidence: {source}")
    if "nearest_mocap_sample_delta_ms" not in sync:
        raise ValueError(f"Missing canonical nearest-sample metric: {source}")

    def side_summary(side: dict) -> dict:
        joint = side["root_normalized_non_thumb_joint_epe_mm"]
        tip = side["root_normalized_non_thumb_fingertip_epe_mm"]
        frame = side["root_normalized_frame_mpjpe_mm"]
        return {
            "jointMedianMm": round(joint["median"], 2),
            "jointP95Mm": round(joint["p95"], 2),
            "tipMedianMm": round(tip["median"], 2),
            "frameMpjpeMeanMm": round(frame["mean"], 2),
        }

    valid_frames = min(left["valid_rgb_frames"], right["valid_rgb_frames"])
    synchronized_frames = sync["synchronized_rgb_frames"]
    protocol = payload["evaluation_protocol"]
    left_summary = side_summary(left)
    right_summary = side_summary(right)
    return {
        "id": spec["id"],
        "segment": spec["segment"],
        "label": spec["label"],
        "title": spec["diagnosticTitle"] if diagnostic else spec["title"],
        "role": spec["role"],
        "roleZh": spec["roleZh"],
        "summary": spec[summary_key],
        "variant": variant,
        "video": f"/media/{spec['id']}-{variant}.mp4",
        "poster": f"/media/{spec['id']}-{variant}-contact.jpg",
        "metricsDownload": f"/downloads/{spec['id']}-{variant}-strict25.metrics.json",
        "validFrames": valid_frames,
        "synchronizedFrames": synchronized_frames,
        "coveragePercent": round(100.0 * valid_frames / synchronized_frames, 1),
        "nearestMocapMedianMs": round(
            sync["nearest_mocap_sample_delta_ms"]["median"], 3
        ),
        "nearestMocapP95Ms": round(
            sync["nearest_mocap_sample_delta_ms"]["p95"], 3
        ),
        "maxInterpolationGapMs": sync["max_interpolation_gap_ms"],
        "registrationSource": protocol["registration_source_segment"],
        "targetUsedInFit": protocol["target_segment_used_in_registration_fit"],
        "isCrossTakeHoldout": protocol["is_cross_take_holdout"],
        "left": left_summary,
        "right": right_summary,
        "cards": [
            {
                "label": "左手",
                "primary": left_summary["jointMedianMm"],
                "primaryLabel": "joint median · mm",
                "secondary": left_summary["tipMedianMm"],
                "secondaryLabel": "tip median · mm",
            },
            {
                "label": "右手",
                "primary": right_summary["jointMedianMm"],
                "primaryLabel": "joint median · mm",
                "secondary": right_summary["tipMedianMm"],
                "secondaryLabel": "tip median · mm",
            },
        ],
    }


def validate_recorded_artifact(
    path: Path, record: dict, *, label: str
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} artifact is missing: {path}")
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError(f"{label} byte count disagrees with metrics: {path}")
    if sha256(path) != record.get("sha256"):
        raise ValueError(f"{label} SHA256 disagrees with metrics: {path}")


def mocap_summary(spec: dict) -> dict:
    source = PROJECT_ROOT / spec["mocapMetrics"]
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "gt_calib.mocap_video_alignment.v2":
        raise ValueError(f"Unsupported MOCAP/video metrics schema: {source}")
    if payload.get("segment") != spec["segment"]:
        raise ValueError(f"MOCAP/video metrics segment mismatch: {source}")

    spatial = payload["spatial_alignment"]
    rear_proxy = spatial.get("rear_wrist_mount_visualization", {})
    proxy_gates = {
        "enabled": rear_proxy.get("enabled") is True,
        "wrist_index": rear_proxy.get("wrist_joint_index") == 0,
        "middle_mcp_index": rear_proxy.get("middle_mcp_joint_index") == 9,
        "scale": float(rear_proxy.get("scale", -1.0)) == 0.8,
        "joints_unchanged": rear_proxy.get("original_21_joint_positions_modified")
        is False,
        "not_gt_joint": rear_proxy.get("additional_mocap_gt_joint") is False,
    }
    if not all(proxy_gates.values()):
        raise ValueError(
            f"MOCAP rear-wrist visualization contract failed for {source}: "
            f"{proxy_gates}"
        )
    world_translation = spatial.get("mocap_world_translation", {})
    if (
        world_translation.get("enabled") is not False
        or world_translation.get("translation_xyz_mm") != [0.0, 0.0, 0.0]
    ):
        raise ValueError(
            f"Rear-wrist site view must keep the delivered 21 joints at the raw "
            f"world coordinates: {source}"
        )

    artifacts = payload["artifacts"]
    candidate_artifacts = {
        "mocapVideo": "h264_delivery_video",
        "mocapPoster": "contact_sheet",
        "mocapAlignment": "frame_mapping_csv",
    }
    for spec_key, artifact_key in candidate_artifacts.items():
        validate_recorded_artifact(
            PROJECT_ROOT / spec[spec_key],
            artifacts[artifact_key],
            label=f"rear-proxy {artifact_key}",
        )

    raw_source = PROJECT_ROOT / spec["mocapRawMetrics"]
    raw_payload = json.loads(raw_source.read_text(encoding="utf-8"))
    if raw_payload.get("schema") != "gt_calib.mocap_video_alignment.v2":
        raise ValueError(f"Unsupported raw MOCAP/video metrics schema: {raw_source}")
    raw_artifacts = raw_payload["artifacts"]
    for spec_key, artifact_key in {
        "mocapRawVideo": "h264_delivery_video",
        "mocapRawPoster": "contact_sheet",
        "mocapRawAlignment": "frame_mapping_csv",
    }.items():
        validate_recorded_artifact(
            PROJECT_ROOT / spec[spec_key],
            raw_artifacts[artifact_key],
            label=f"raw-21-joint {artifact_key}",
        )
    if sha256(PROJECT_ROOT / spec["mocapAlignment"]) != sha256(
        PROJECT_ROOT / spec["mocapRawAlignment"]
    ):
        raise ValueError(
            f"Rear-proxy visualization changed the frame mapping for {spec['segment']}"
        )

    frames = payload["video_frame_contract"]
    temporal = payload["temporal_alignment"]
    visual = temporal["visual_motion_validation"]
    corrected = visual["acquisition_time_corrected"]
    legacy = visual["legacy_callback_time"]
    acceptance = temporal["acceptance"]
    if not acceptance.get("pass"):
        raise ValueError(f"MOCAP/video temporal acceptance failed: {source}")
    if not frames.get("published_clip_all_frames_mocap_valid"):
        raise ValueError(f"MOCAP/video clip is not frame-complete: {source}")
    content = frames["bag_mp4_frame_content_validation"]
    if not frames.get("bag_message_i_equals_mp4_frame_i_content_validated"):
        raise ValueError(f"BAG/MP4 content identity was not validated: {source}")
    if not content["acceptance"].get("pass"):
        raise ValueError(f"BAG/MP4 decoded-content gate failed: {source}")
    if not content.get("full_frame_content_comparison"):
        raise ValueError(f"Review site requires full-frame BAG/MP4 evidence: {source}")
    latency = temporal["camera_capture_to_thor_poll_latency_ms"]
    nearest = temporal["nearest_mocap_sample_delta_ms"]
    ptp_fit = temporal["mocap_sample_timebase"][
        "display_timestamp_fit_absolute_residual_ms"
    ]
    count = int(frames["published_clip_frames"])
    content_sample_count = int(content["sample_count"])
    content_source_count = int(frames["source_mp4_frames"])
    content_full_frame = bool(content["full_frame_content_comparison"])
    distinguishable_count = int(content["distinguishable_sample_count"])
    distinguishable_same_count = int(
        content["distinguishable_same_index_best_count"]
    )
    return {
        "id": spec["id"],
        "segment": spec["segment"],
        "label": spec["label"],
        "title": spec["mocapTitle"],
        "role": spec["role"],
        "roleZh": spec["roleZh"],
        "summary": spec["mocapSummary"],
        "variant": "mocap",
        "badgeLabel": "REAR WRIST PROXY · VIZ-ONLY",
        "video": f"/media/{spec['id']}-mocap.mp4",
        "poster": f"/media/{spec['id']}-mocap-contact.jpg",
        "rawVideoDownload": f"/media/{spec['id']}-mocap-raw.mp4",
        "rawPosterDownload": f"/media/{spec['id']}-mocap-raw-contact.jpg",
        "metricsDownload": f"/downloads/{spec['id']}-mocap-alignment.metrics.json",
        "alignmentDownload": f"/downloads/{spec['id']}-mocap-frame-map.csv",
        "visualizationOnly": True,
        "rearMountProxyEnabled": True,
        "rearMountProxyScale": 0.8,
        "deliveredJointCountPerHand": 21,
        "original21JointPositionsModified": False,
        "additionalGroundTruthJoint": False,
        "alignmentIdenticalToRaw21JointView": True,
        "validFrames": count,
        "synchronizedFrames": count,
        "coveragePercent": 100.0,
        "captureLatencyMedianMs": round(latency["median"], 3),
        "captureLatencyP95Ms": round(latency["p95"], 3),
        "nearestMedianMs": round(nearest["median"], 3),
        "nearestP95Ms": round(nearest["p95"], 3),
        "visualLegacyCorr": round(legacy["zero_adjustment_correlation"], 4),
        "visualCorrectedCorr": round(corrected["zero_adjustment_correlation"], 4),
        "visualGain": round(visual["zero_adjustment_correlation_gain"], 4),
        "residualLagMs": int(corrected["best_timestamp_adjustment_ms"]),
        "clockUncertaintyMs": round(
            temporal["clock_model_estimated_uncertainty_ms"], 3
        ),
        "ptpFitP95Ms": round(ptp_fit["p95"], 3),
        "wristInsidePercent": round(
            100.0 * spatial["wrist_inside_rgb_frame_fraction"], 2
        ),
        "timecodeExactPercent": round(
            100.0
            * frames["timecode_to_bag_timestamp_check"]["exact_match_fraction"],
            2,
        ),
        "contentFullFrame": content_full_frame,
        "contentSampleCount": content_sample_count,
        "contentSourceFrameCount": content_source_count,
        "contentEvidenceLabel": (
            f"全 {content_sample_count} 帧"
            if content_full_frame
            else f"全时段固定 {content_sample_count} 帧"
        ),
        "contentSameIndexBestCount": int(content["same_index_best_neighbor_count"]),
        "contentSameIndexBestPercent": round(
            100.0 * float(content["same_index_best_neighbor_fraction"]), 2
        ),
        "contentDistinguishableCount": distinguishable_count,
        "contentDistinguishableSameIndexCount": distinguishable_same_count,
        "contentMseMax": round(content["same_index_luma_mse"]["max"], 3),
        "contentPsnrMinDb": round(
            content["same_index_luma_psnr_db"]["min"], 3
        ),
        "contentDhashMax": int(content["same_index_dhash64_hamming"]["max"]),
        "contentBestGlobalOffsetFrames": int(content["best_global_offset_frames"]),
        "cards": [
            {
                "label": "视觉相位 QA",
                "primary": round(corrected["zero_adjustment_correlation"], 3),
                "primaryLabel": "corrected zero-lag corr",
                "secondary": int(corrected["best_timestamp_adjustment_ms"]),
                "secondaryLabel": "残余峰值 · ms（不回写）",
            },
            {
                "label": "时间链证据",
                "primary": 100.0,
                "primaryLabel": "共同区间 MOCAP coverage · %",
                "secondary": round(
                    100.0
                    * frames["timecode_to_bag_timestamp_check"][
                        "exact_match_fraction"
                    ],
                    2,
                ),
                "secondaryLabel": "timecode ↔ BAG exact · %",
            },
            {
                "label": "RGB 内容同帧 gate",
                "primary": f"{content_sample_count}/{content_source_count}",
                "primaryLabel": (
                    "全帧 BAG message i ↔ MP4 frame i"
                    if content_full_frame
                    else "抽样 BAG message i ↔ MP4 frame i"
                ),
                "secondary": f"{distinguishable_same_count}/{distinguishable_count}",
                "secondaryLabel": "可辨帧 best offset = 0",
            },
        ],
    }


def depth_artifact_paths(spec: dict) -> dict[str, Path]:
    stem = depth_output_stem(spec["segment"], 25.0)
    return {
        "metrics": DEPTH_OUTPUT_DIR / f"{stem}.metrics.json",
        "video": DEPTH_OUTPUT_DIR / f"{stem}_h264.mp4",
        "poster": DEPTH_OUTPUT_DIR / f"{stem}.contact.jpg",
        "alignment": DEPTH_OUTPUT_DIR / f"{stem}.alignment.csv",
    }


def _rigid_rotation_diagnostics(matrix: list[list[float]]) -> tuple[float, float]:
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("world_to_depth_camera must be a 4x4 matrix")
    if any(abs(float(value) - expected) > 1e-12 for value, expected in zip(matrix[3], (0, 0, 0, 1))):
        raise ValueError("world_to_depth_camera has an invalid homogeneous row")
    rotation = [[float(matrix[row][column]) for column in range(3)] for row in range(3)]
    determinant = (
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    orthogonality_error = max(
        abs(
            sum(rotation[row][left] * rotation[row][right] for row in range(3))
            - (1.0 if left == right else 0.0)
        )
        for left in range(3)
        for right in range(3)
    )
    return determinant, orthogonality_error


def validate_depth_alignment_csv(path: Path, expected_rows: int) -> dict:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != DEPTH_ALIGNMENT_FIELDS:
            raise ValueError(f"Unexpected depth frame-map schema: {path}")
        row_count = 0
        first_source_frame: int | None = None
        previous_source_frame: int | None = None
        previous_timestamp_us: int | None = None
        for row in reader:
            output_frame = int(row["output_frame"])
            source_frame = int(row["source_depth_frame"])
            timestamp_us = int(row["depth_device_timestamp_us"])
            inside_count = int(row["projected_joint_inside_count"])
            nonzero_count = int(row["projected_joint_nonzero_depth_count"])
            if output_frame != row_count:
                raise ValueError(f"Non-contiguous output_frame in depth frame map: {path}")
            if first_source_frame is None:
                first_source_frame = source_frame
            if previous_source_frame is not None and source_frame != previous_source_frame + 1:
                raise ValueError(f"Non-contiguous source_depth_frame in depth frame map: {path}")
            if previous_timestamp_us is not None and timestamp_us <= previous_timestamp_us:
                raise ValueError(f"Non-increasing depth timestamp in frame map: {path}")
            if not (0 <= nonzero_count <= inside_count <= 42):
                raise ValueError(f"Invalid projected-joint coverage count in frame map: {path}")
            if float(row["bracket_span_ms"]) > 25.0:
                raise ValueError(f"Depth frame map exceeds the strict 25 ms gate: {path}")
            previous_source_frame = source_frame
            previous_timestamp_us = timestamp_us
            row_count += 1
    if row_count != expected_rows:
        raise ValueError(
            f"Depth frame map has {row_count} rows, expected {expected_rows}: {path}"
        )
    return {
        "rows": row_count,
        "firstSourceDepthFrame": first_source_frame,
        "lastSourceDepthFrame": previous_source_frame,
    }


def depth_summary(spec: dict) -> dict:
    paths = depth_artifact_paths(spec)
    source = paths["metrics"]
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != DEPTH_OVERLAY_SCHEMA:
        raise ValueError(f"Unsupported depth/MOCAP metrics schema: {source}")
    if payload.get("segment") != spec["segment"]:
        raise ValueError(f"Depth/MOCAP metrics segment mismatch: {source}")

    frames = payload["depth_frame_contract"]
    temporal = payload["temporal_alignment"]
    spatial = payload["spatial_projection_coverage"]
    pairing = frames.get("rgb_depth_pairing", {})
    expected_frames = EXPECTED_DEPTH_COMMON_FRAMES[spec["segment"]]
    if int(frames["published_clip_frames"]) != expected_frames:
        raise ValueError(
            f"Depth/MOCAP clip does not contain the full expected common interval: {source}"
        )
    frame_gates = {
        "all_mocap_valid": frames.get("published_clip_all_frames_mocap_valid") is True,
        "full_common_interval": frames.get("covers_full_synchronized_depth_interval") is True,
        "decoded_timestamp_match": frames.get("decoded_device_timestamps_match_bag_index") is True,
        "decoded_timestamp_increasing": frames.get("decoded_device_timestamps_strictly_increasing") is True,
        "decoded_frame_number_increasing": frames.get("decoded_frame_numbers_strictly_increasing") is True,
        "rgb_depth_pairing_pass": pairing.get("pass") is True,
        "same_index_nearest": pairing.get("same_index_is_nearest_for_all_paired_messages") is True,
        "extra_tail_not_forced": frames.get("extra_depth_tail_is_not_forced_onto_rgb") is True,
        "raw_metric_contract": frames.get("encoding") == "mono16" and frames.get("units") == "millimetres",
        "native_resolution": frames.get("resolution") == [640, 576],
    }
    if not all(frame_gates.values()):
        raise ValueError(f"Depth frame/timestamp gate failed for {source}: {frame_gates}")

    bracket = temporal["mocap_bracket_span_ms"]
    nearest = temporal["nearest_mocap_sample_delta_ms"]
    depth_rgb = temporal["paired_depth_minus_rgb_device_timestamp_ms"]
    timing_gates = {
        "uses_depth_timestamp": temporal.get("depth_timestamp_used_directly_not_rgb_timestamp_substituted") is True,
        "all_interpolated": temporal.get("all_published_frames_have_valid_interpolation") is True,
        "strict_bracket": float(bracket["max"])
        <= float(temporal["max_allowed_bracket_span_ms"])
        <= 25.0,
        "clock_model": float(temporal["clock_model_estimated_uncertainty_ms"])
        <= 1.0,
        "depth_follows_rgb": 0.0 < float(depth_rgb["min"])
        <= float(depth_rgb["max"]) < 5.0,
    }
    if not all(timing_gates.values()):
        raise ValueError(f"Depth/MOCAP timing gate failed for {source}: {timing_gates}")

    determinant, orthogonality_error = _rigid_rotation_diagnostics(
        spatial["world_to_depth_camera"]
    )
    camera_contract = spatial.get("recorded_camera_contract", {})
    camera_contract_gates = camera_contract.get("gates", {})
    se3_gates = {
        "frozen_no_refit": spatial.get("world_to_depth_camera_frozen_no_refit") is True,
        "fixed_camera": spatial.get("fixed_camera_assumption") is True,
        "right_handed": abs(determinant - 1.0) <= 1e-9,
        "orthonormal": orthogonality_error <= 1e-9,
        "reported_determinant": abs(float(spatial["rotation_determinant"]) - determinant)
        <= 1e-12,
        "reported_orthogonality": abs(
            float(spatial["rotation_orthogonality_max_abs_error"])
            - orthogonality_error
        )
        <= 1e-12,
        "not_surface_accuracy": spatial.get("skeleton_to_surface_error_measured") is False,
        "no_independent_labels": spatial.get("independent_depth_joint_labels_available") is False,
        "recorded_camera_contract_pass": camera_contract.get("pass") is True,
        "recorded_camera_contract_gates": bool(camera_contract_gates)
        and all(camera_contract_gates.values()),
    }
    if not all(se3_gates.values()):
        raise ValueError(f"Depth/MOCAP SE(3) gate failed for {source}: {se3_gates}")

    artifact_keys = {
        "video": "h264_delivery_video",
        "poster": "contact_sheet",
        "alignment": "frame_mapping_csv",
    }
    for path_key, metrics_key in artifact_keys.items():
        path = paths[path_key]
        if not path.is_file():
            raise FileNotFoundError(f"Required depth/MOCAP site asset is missing: {path}")
        recorded = payload["artifacts"][metrics_key]
        if sha256(path) != recorded["sha256"]:
            raise ValueError(f"Depth/MOCAP artifact hash disagrees with metrics: {path}")
    alignment = validate_depth_alignment_csv(paths["alignment"], expected_frames)

    inside_percent = 100.0 * float(spatial["inside_depth_frame_fraction"])
    nonzero_joint_percent = 100.0 * float(
        spatial["raw_depth_valid_at_projected_joint_pixel_fraction"]
    )
    raw_valid_percent = 100.0 * float(
        frames["raw_valid_pixel_fraction"]["median"]
    )
    count = int(frames["published_clip_frames"])
    return {
        "id": spec["id"],
        "segment": spec["segment"],
        "label": spec["label"],
        "title": spec["depthTitle"],
        "role": "fixed_depth_pose_review",
        "roleZh": "00 外参冻结后的 review",
        "summary": spec["depthSummary"],
        "variant": "depth",
        "badgeLabel": "00-FROZEN DEPTH POSE REVIEW",
        "coverageLabel": "共同区间时序覆盖",
        "video": f"/media/{spec['id']}-depth.mp4",
        "poster": f"/media/{spec['id']}-depth-contact.jpg",
        "metricsDownload": f"/downloads/{spec['id']}-depth-mocap.metrics.json",
        "alignmentDownload": f"/downloads/{spec['id']}-depth-frame-map.csv",
        "validFrames": count,
        "synchronizedFrames": count,
        "coveragePercent": 100.0,
        "depthMessageCount": int(frames["raw_bag_depth_messages"]),
        "rgbMessageCount": int(frames["raw_bag_rgb_messages"]),
        "depthExtraTailFrames": int(frames["depth_minus_rgb_message_count"]),
        "alignmentRows": alignment["rows"],
        "firstSourceDepthFrame": alignment["firstSourceDepthFrame"],
        "lastSourceDepthFrame": alignment["lastSourceDepthFrame"],
        "depthRgbMedianMs": round(float(depth_rgb["median"]), 3),
        "depthRgbP95Ms": round(float(depth_rgb["p95"]), 3),
        "nearestMedianMs": round(float(nearest["median"]), 3),
        "nearestP95Ms": round(float(nearest["p95"]), 3),
        "bracketMaxMs": round(float(bracket["max"]), 3),
        "clockUncertaintyMs": round(
            float(temporal["clock_model_estimated_uncertainty_ms"]), 3
        ),
        "insideDepthFramePercent": round(inside_percent, 2),
        "nonzeroDepthAtJointPercent": round(nonzero_joint_percent, 2),
        "rawValidPixelMedianPercent": round(raw_valid_percent, 2),
        "rotationDeterminant": round(determinant, 9),
        "rotationOrthogonalityError": f"{orthogonality_error:.3e}",
        "timestampGatesPassed": True,
        "se3GatesPassed": True,
        "skeletonToSurfaceAccuracyMeasured": False,
        "coverageInterpretation": "nonzero depth coverage 是 projected joint pixel 上的传感器可用率，不是 pose accuracy。",
        "cards": [
            {
                "label": "Depth 时间同步",
                "primary": round(float(nearest["median"]), 3),
                "primaryLabel": "nearest MOCAP median · ms",
                "secondary": round(float(bracket["max"]), 3),
                "secondaryLabel": "bracket max · ms",
            },
            {
                "label": "投影 / 非零深度诊断",
                "primary": round(inside_percent, 2),
                "primaryLabel": "投影在 640×576 画内 · %",
                "secondary": round(nonzero_joint_percent, 2),
                "secondaryLabel": "joint pixel 非零深度 · %（非精度）",
            },
            {
                "label": "冻结 Depth SE(3)",
                "primary": f"{determinant:.6f}",
                "primaryLabel": "rotation determinant",
                "secondary": f"{orthogonality_error:.3e}",
                "secondaryLabel": "max |RᵀR−I|",
            },
        ],
    }


def load_requests() -> list[dict]:
    source = PROJECT_ROOT / "docs/dataset/request_datalist_2026-08-31.csv"
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [
        {
            "id": row["request_id"],
            "priority": row["priority"],
            "item": row["item"],
            "files": row["requested_files"],
            "acceptance": row["acceptance"],
            "dueDate": row["due_date"],
            "status": row["status"],
        }
        for row in rows
    ]


def load_copy_message() -> str:
    source = PROJECT_ROOT / "docs/dataset/DATA_REQUEST_2026-08-31.md"
    lines = source.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line.startswith("> "):
            return line[2:].strip()
    raise ValueError(f"No blockquote copy message found in {source}")


def load_depth_calibration_audit() -> dict:
    source = PROJECT_ROOT / "outputs/depth_calibration_audit/depth_camera_pose_audit.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "gt_calib.depth_camera_pose_audit.v1":
        raise ValueError(f"Unsupported depth calibration audit schema: {source}")
    depth_pose = payload["depth_camera_pose"]
    repeatability = payload["reference_00"]["temporal_repeatability"]
    fixed_view = payload["cross_take_fixed_camera"]["rgb_background_registration"]
    table = payload["cross_take_fixed_camera"]["raw_depth_table_plane"]
    d2c = payload["depth_to_color"]
    if not depth_pose.get("is_valid_se3_at_1e-9"):
        raise ValueError("Review site requires a valid delivered depth-camera SE(3)")
    if not repeatability.get("pass"):
        raise ValueError("Review site requires the 00 blocked repeatability gate")
    if not fixed_view.get("all_takes_pass"):
        raise ValueError("Review site requires the fixed-view RGB background gate")
    if d2c.get("is_valid_rotation_at_1e-6"):
        raise ValueError("D2C audit unexpectedly claims a rigid delivered rotation")
    return {
        "status": payload["status"],
        "depthPoseIsSE3": True,
        "depthCameraOriginWorldMm": [
            round(float(value), 3)
            for value in depth_pose["depth_camera_origin_in_world_mm"]
        ],
        "referenceRotationDeltaMaxDeg": round(
            repeatability["max_rotation_delta_deg"], 4
        ),
        "referenceOriginDeltaMaxMm": round(
            repeatability["max_origin_delta_mm"], 4
        ),
        "rgbFixedViewAllTakesPass": True,
        "rgbFixedViewMaxTranslationPx": round(
            max(
                take["summary"]["translation_magnitude_px_max"]
                for take in fixed_view["takes"]
            ),
            4,
        ),
        "rgbFixedViewMaxRotationDeg": round(
            max(
                take["summary"]["absolute_rotation_deg_max"]
                for take in fixed_view["takes"]
            ),
            4,
        ),
        "depthPlaneReview": not table["acceptance_pass"],
        "depthPlaneNormalDeltaDeg": [
            round(take["relative_to_00"]["plane_normal_angle_deg"], 3)
            for take in table["takes"]
        ],
        "depthPlaneDDeltaMm": [
            round(take["relative_to_00"]["plane_d_delta_mm"], 2)
            for take in table["takes"]
        ],
        "d2cIsRigid": False,
        "d2cDeterminant": round(d2c["determinant"], 6),
        "d2cOrthogonalityError": round(
            d2c["max_abs_orthogonality_error"], 6
        ),
        "d2cRigidCandidateProjectionP95Px": round(
            d2c["affine_vs_rigid_workspace_projection_delta_px"]["p95"], 3
        ),
        "auditDownload": "/downloads/depth-camera-pose-audit.json",
        "deliveredCalibrationDownload": "/downloads/camera-to-world-delivered.json",
        "rigidCandidateDownload": "/downloads/d2c-rigid-candidate-NOT-ACTIVE.json",
        "reportDownload": "/downloads/DEPTH_CAMERA_POSE_AUDIT_V1.md",
    }


def load_wrist_semantics_audit() -> dict:
    source = MOCAP_WRIST_SEMANTICS_DIR / "summary.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "gt_calib.mocap_wrist_semantics_audit.v1":
        raise ValueError(f"Unsupported wrist-semantics audit schema: {source}")
    if payload.get("assertions", {}).get("all_passed") is not True:
        raise ValueError(f"Wrist-semantics audit assertions did not pass: {source}")
    if payload.get("selection", {}).get("take02_output_661_excluded") is not True:
        raise ValueError(f"Wrist-semantics audit reused the hypothesis frame: {source}")

    contacts: list[dict] = []
    for take in payload.get("takes", []):
        take_key = str(take["take_key"])
        contact = MOCAP_WRIST_SEMANTICS_DIR / take["contact_sheet"]
        if sha256(contact) != take["contact_sheet_sha256"]:
            raise ValueError(f"Wrist-semantics contact hash mismatch: {contact}")
        contacts.append(
            {
                "take": f"Take {take_key}",
                "segment": take["segment"],
                "image": f"/assets/mocap-wrist-semantics-take-{take_key}.png",
                "selectedFrames": take["selected_output_frames"],
            }
        )
    if len(contacts) != 3:
        raise ValueError(f"Expected three wrist-semantics contacts: {source}")

    provenance = MOCAP_WRIST_SEMANTICS_DIR / payload["frame_provenance"]["path"]
    if sha256(provenance) != payload["frame_provenance"]["sha256"]:
        raise ValueError(f"Wrist-semantics provenance hash mismatch: {provenance}")
    if int(payload["frame_provenance"]["row_count"]) != 24:
        raise ValueError(f"Wrist-semantics audit must contain 24 held-out frames: {source}")

    rear_proxy = payload["views"]["rear_mount_proxy"]
    if (
        rear_proxy.get("modifies_delivered_21_joint_hands") is not False
        or rear_proxy.get("status")
        != "display_only_semantic_proxy_not_physical_ground_truth"
    ):
        raise ValueError(f"Unsafe wrist proxy claim in audit: {source}")
    return {
        "status": payload["status"],
        "heldOutFrames": 24,
        "hypothesisFrameExcluded": True,
        "deliveredJointCountPerHand": 21,
        "originalJointPositionsModified": False,
        "rearProxyFormula": rear_proxy["formula"],
        "rearProxyStatus": rear_proxy["status"],
        "contacts": contacts,
        "summaryDownload": "/downloads/mocap-wrist-semantics-summary.json",
        "provenanceDownload": "/downloads/mocap-wrist-semantics-frame-provenance.csv",
        "reportDownload": "/downloads/MOCAP_WRIST_SEMANTICS_V1.md",
    }


def create_evidence_zip(manifest: list[dict]) -> Path:
    destination = PUBLIC_ROOT / "downloads/gt-calib-evidence-2026-08-31.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    evidence = [
        entry for entry in manifest if entry["path"].startswith("downloads/")
    ]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in evidence:
            file_path = PUBLIC_ROOT / entry["path"]
            archive.write(file_path, arcname=entry["path"].removeprefix("downloads/"))
        checksums = "".join(
            f"{entry['sha256']}  {entry['path'].removeprefix('downloads/')}\n"
            for entry in evidence
        )
        archive.writestr("SHA256SUMS.txt", checksums)
    ensure_cloudflare_size(destination)
    manifest.append(
        {
            "path": "downloads/gt-calib-evidence-2026-08-31.zip",
            "source": "generated allow-listed evidence bundle (videos excluded)",
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }
    )
    return destination


def build_source_site() -> int:
    """Regenerate every source-grounded site artifact from local raw outputs."""

    _load_source_build_dependencies()
    PUBLIC_ROOT.mkdir(parents=True, exist_ok=True)
    for relative in OBSOLETE_GENERATED_PATHS:
        obsolete = PUBLIC_ROOT / relative
        if obsolete.is_symlink():
            raise ValueError(f"Refusing to remove unexpected symlink: {obsolete}")
        if obsolete.is_file():
            obsolete.unlink()

    manifest: list[dict] = []
    summaries: list[dict] = []
    diagnostic_summaries: list[dict] = []
    mocap_summaries: list[dict] = []
    depth_summaries: list[dict] = []

    bvh_manifest = export_bvh_segments(
        BVH_DATASET_ROOT,
        BVH_OUTPUT_DIR,
        BVH_FORMAL_SEGMENTS,
        alignment_dir=MOCAP_REAR_ALIGNMENT_DIR,
        target_fps=60.0,
        quantum_bvh_units=0.001,
    )
    if bvh_manifest.get("schema") != BVH_MANIFEST_SCHEMA:
        raise ValueError("BVH browser manifest schema mismatch")
    if bvh_manifest.get("export_schema") != BVH_EXPORT_SCHEMA:
        raise ValueError("BVH browser payload schema mismatch")
    if bvh_manifest.get("validation", {}).get("status") != "pass":
        raise ValueError("BVH browser export did not pass validation")

    copy_asset(
        (BVH_OUTPUT_DIR / "manifest.json").relative_to(PROJECT_ROOT).as_posix(),
        "bvh/data/manifest.json",
        manifest,
    )
    for entry in bvh_manifest["takes"]:
        key = str(entry["segment_key"])
        _, left_bvh, right_bvh, take_name = discover_formal_take(BVH_DATASET_ROOT, key)
        take_number = take_name.removeprefix("Take_")
        copy_asset(
            (BVH_OUTPUT_DIR / str(entry["file"])).relative_to(PROJECT_ROOT).as_posix(),
            f"bvh/data/{entry['file']}",
            manifest,
        )
        copy_asset(
            left_bvh.relative_to(PROJECT_ROOT).as_posix(),
            f"bvh/data/take-{take_number}-skeleton-0.bvh",
            manifest,
        )
        copy_asset(
            right_bvh.relative_to(PROJECT_ROOT).as_posix(),
            f"bvh/data/take-{take_number}-skeleton-1.bvh",
            manifest,
        )

    for spec in TAKES:
        summaries.append(metric_summary(spec))
        diagnostic_summaries.append(metric_summary(spec, diagnostic=True))
        mocap_summaries.append(mocap_summary(spec))
        depth_summaries.append(depth_summary(spec))
        copy_asset(spec["video"], f"media/{spec['id']}-fusion.mp4", manifest)
        copy_asset(
            spec["poster"], f"media/{spec['id']}-fusion-contact.jpg", manifest
        )
        copy_asset(
            spec["metrics"],
            f"downloads/{spec['id']}-fusion-strict25.metrics.json",
            manifest,
        )
        copy_asset(
            spec["diagnosticVideo"],
            f"media/{spec['id']}-diagnostic.mp4",
            manifest,
        )
        copy_asset(
            spec["diagnosticPoster"],
            f"media/{spec['id']}-diagnostic-contact.jpg",
            manifest,
        )
        copy_asset(
            spec["diagnosticMetrics"],
            f"downloads/{spec['id']}-diagnostic-strict25.metrics.json",
            manifest,
        )
        copy_asset(spec["mocapVideo"], f"media/{spec['id']}-mocap.mp4", manifest)
        copy_asset(
            spec["mocapPoster"], f"media/{spec['id']}-mocap-contact.jpg", manifest
        )
        copy_asset(
            spec["mocapMetrics"],
            f"downloads/{spec['id']}-mocap-alignment.metrics.json",
            manifest,
        )
        copy_asset(
            spec["mocapAlignment"],
            f"downloads/{spec['id']}-mocap-frame-map.csv",
            manifest,
        )
        copy_asset(
            spec["mocapRawVideo"], f"media/{spec['id']}-mocap-raw.mp4", manifest
        )
        copy_asset(
            spec["mocapRawPoster"],
            f"media/{spec['id']}-mocap-raw-contact.jpg",
            manifest,
        )
        depth_paths = depth_artifact_paths(spec)
        copy_asset(
            depth_paths["video"].relative_to(PROJECT_ROOT).as_posix(),
            f"media/{spec['id']}-depth.mp4",
            manifest,
        )
        copy_asset(
            depth_paths["poster"].relative_to(PROJECT_ROOT).as_posix(),
            f"media/{spec['id']}-depth-contact.jpg",
            manifest,
        )
        copy_asset(
            depth_paths["metrics"].relative_to(PROJECT_ROOT).as_posix(),
            f"downloads/{spec['id']}-depth-mocap.metrics.json",
            manifest,
        )
        copy_asset(
            depth_paths["alignment"].relative_to(PROJECT_ROOT).as_posix(),
            f"downloads/{spec['id']}-depth-frame-map.csv",
            manifest,
        )

    for source, destination in STATIC_COPIES.items():
        copy_asset(source, destination, manifest)

    for take_number, segment in (("000", "01"), ("001", "02"), ("002", "03")):
        matches = sorted(
            (PROJECT_ROOT / "outputs/mocap_root_fusion_gap50").glob(
                f"{segment}_*.metrics.json"
            )
        )
        if len(matches) != 1:
            raise ValueError(f"Expected one 50 ms metrics file for segment {segment}, got {matches}")
        relative = matches[0].relative_to(PROJECT_ROOT).as_posix()
        copy_asset(
            relative,
            f"downloads/take-{take_number}-fusion-gap50.metrics.json",
            manifest,
        )

    data = {
        "schema": "gt-calib.review-site.v2",
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reportDate": "2026-08-31",
        "status": "verified_exposure_time_alignment_with_full_frame_rgb_identity",
        "datasetIds": [
            "ds-20260829-take000",
            "ds-20260829-take001",
            "ds-20260829-take002",
        ],
        "protocol": {
            "rgbFrameIdentity": "BAG color message i 与 MP4 frame i 必须通过解码内容及 ±2 帧错位反证 gate",
            "cameraTimestamp": "每行 color_device_timestamp_us 对应 RGB 采集；逐行减去 capture-to-poll 延迟",
            "mocapTimestamp": "PtpTimeStamp 提供高分辨率相对节拍；对 Windows TimeStamp 做 affine anchor",
            "interpolation": "在相机曝光时刻对 120 Hz MOCAP 做线性位置插值；不做 per-take magic offset",
            "fit": "Take 01 / Take_000",
            "freeze": "每只手一个 wrist-local canonical R + uniform scale",
            "rootSource": "目标 Take 的 MOCAP wrist position + Human GlobalQ",
            "articulationSource": "目标 Take 的 glove root-local 20 点",
            "evaluate": "冻结参数后回顾性评估 Take 02 validation + Take 03 test/stress；目标 finger joints 不参与拟合",
            "metricFrame": "MOCAP-wrist SE(3)-conditioned root-normalized",
            "gate": "25 ms bracketing-source interpolation span",
        },
        "headline": "修正约 255 ms capture→poll 延迟后，三段视频运动与 MOCAP 在零偏移处相关性达到 0.876–0.899。",
        "primaryFinding": {
            "captureLatencyMedianRangeMs": [254.008, 255.25],
            "legacyZeroLagCorrelationRange": [0.3947, 0.6261],
            "correctedZeroLagCorrelationRange": [0.8761, 0.8987],
            "residualPeakAdjustmentMs": [10, 16, 17],
            "residualWasApplied": False,
        },
        "mocapTakes": mocap_summaries,
        "depthTakes": depth_summaries,
        "takes": summaries,
        "diagnosticTakes": diagnostic_summaries,
        "calibrationAudit": load_depth_calibration_audit(),
        "wristSemanticsAudit": load_wrist_semantics_audit(),
        "requests": load_requests(),
        "copyMessage": load_copy_message(),
        "limitations": [
            "当前“逐帧匹配”表示 BAG/MP4 内容索引 gate 通过，且共同区间每个 RGB 帧都有曝光时刻 MOCAP bracket；没有硬件触发就不能声称零时间误差。",
            "视觉相关性的 10–17 ms 残余只用于 QA，不会作为每段手工时间偏移回写。",
            "PtpTimeStamp 的高分辨率相对节拍已使用，但绝对 PTP epoch 契约未交付，仍由 Windows TimeStamp 做 affine anchor。",
            "融合结果每帧直接使用目标 Take 的 MOCAP wrist position + orientation；不是独立 glove wrist 或完整 6DoF 精度。",
            "当前数字是 MOCAP-wrist SE(3)-conditioned 的 finger articulation 一致性误差，不是无条件外部 GT 精度。",
            "P95 长尾仍高，说明部分姿态、插值边界或遮挡段尚未可靠解决。",
            "Take 02/03 的 target fingers 未进入 fit，但结果已用于本轮模型审查；它们不是新的全盲 final test。",
            "Human.cma 缺少逐帧 confidence、visibility、occlusion、gap-fill 与 residual。",
            "三段各自采集 session-neutral，但 neutral/recenter 原始资产和 replay contract 尚未交付。",
            "相机外参尚未用完整 21 关节独立 2D 真值做 held-out 验证；当前投影在画内不等于像素精度。",
            "交付的 depth→color 3×3 轻微非正交；当前忠实使用原始 forward calibration，未静默投影成刚体旋转。",
            "00 depth pose 可复算且 fixed-view 背景检查通过，但这仍不是独立 absolute pose accuracy；raw-depth 桌面跨 Take 差异保持 review。",
            "Depth review 直接使用 raw mono16 毫米值和每个 depth frame 自身的设备时间戳；不会用 RGB timestamp 替代约 1.31 ms 后到的 depth timestamp。",
            "Depth review 的 nonzero depth coverage 只表示投影关节点所在像素有可用量程；MOCAP joint 不是可见表面点，因此该覆盖率不是 pose accuracy。",
            "推荐 MOCAP 视频中带白色中心的同侧彩色圆点是 wrist 与 middle MCP 的 rear wrist module 显示外推；原始每手 21 joints 没有被修改，该 proxy 不是实测节点或 GT。Held-out audit contact 另用橙色菱形标记同一 proxy。",
        ],
    }

    data_dir = PUBLIC_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    site_data = data_dir / "site-data.json"
    site_data.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ensure_cloudflare_size(site_data)
    manifest.append(
        {
            "path": "data/site-data.json",
            "source": "generated from MOCAP/video + raw-depth/MOCAP alignment + fusion + diagnostic strict metrics and request CSV",
            "bytes": site_data.stat().st_size,
            "sha256": sha256(site_data),
        }
    )

    create_evidence_zip(manifest)

    for relative in AUTHORED_PUBLIC_ASSETS:
        path = PUBLIC_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Authored public file is missing: {path}")
        ensure_cloudflare_size(path)
        manifest.append(
            {
                "path": relative,
                "source": f"web/public/{relative}",
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    # The downloadable final package is intentionally ignored under
    # web/public so Git stores only one reviewed copy.  Recreate that mirror
    # from the tracked delivery on every build, after all authored/source
    # assets have passed their own checks and before emitting the aggregate
    # Cloudflare asset manifest.
    manifest.extend(
        mirror_final_delivery(FINAL_DELIVERY_SOURCE, FINAL_DELIVERY_PUBLIC)
    )

    manifest_path = data_dir / "asset-manifest.json"
    manifest_payload = {
        "schema": "gt-calib.asset-manifest.v1",
        "generatedAt": data["generatedAt"],
        "cloudflareMaxAssetBytes": MAX_STATIC_ASSET_BYTES,
        "assets": sorted(manifest, key=lambda item: item["path"]),
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    published_files = [path for path in PUBLIC_ROOT.rglob("*") if path.is_file()]
    oversized = [path for path in published_files if path.stat().st_size > MAX_STATIC_ASSET_BYTES]
    symlinks = [path for path in PUBLIC_ROOT.rglob("*") if path.is_symlink()]
    if oversized or symlinks:
        raise ValueError(f"Unsafe publish tree: oversized={oversized}, symlinks={symlinks}")

    total_bytes = sum(path.stat().st_size for path in published_files)
    print(
        f"Built {len(published_files)} files ({total_bytes / 1024 / 1024:.2f} MiB) "
        f"under {PUBLIC_ROOT}"
    )
    print(f"Largest asset: {max(published_files, key=lambda path: path.stat().st_size)}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble tracked Cloudflare assets or rebuild them from local sources."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--assemble",
        action="store_true",
        help="Assemble a deploy tree from tracked assets (default; clean-clone safe).",
    )
    mode.add_argument(
        "--source",
        action="store_true",
        help="Regenerate the full review site from local raw/outputs sources.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.source:
        return build_source_site()
    assemble_deploy_site(PUBLIC_ROOT, FINAL_DELIVERY_SOURCE)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        raise
