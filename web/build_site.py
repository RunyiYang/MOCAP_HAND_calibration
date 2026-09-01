#!/usr/bin/env python3
"""Build the allow-listed Cloudflare static delivery directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
IMU_COMPARISON_SCHEMA = "gt_calib.imu_solved_pose_mocap_comparison_delivery.v1"
IMU_COMPARISON_SOURCE = PROJECT_ROOT / "imu_mocap_comparison_delivery"
IMU_COMPARISON_PUBLIC = PUBLIC_ROOT / "downloads/imu-mocap"
IMU_COMPARISON_WEB_PREFIX = "downloads/imu-mocap/"
IMU_VISUAL_LAB_SCHEMA = "gt_calib.imu_mocap_visualization_lab.v1"
IMU_VISUAL_LAB_VALIDATION_SCHEMA = (
    "gt_calib.imu_mocap_visualization_lab_validation.v1"
)
IMU_VISUAL_LAB_METRICS_SCHEMA = (
    "gt_calib.imu_mocap_visualization_method_metrics.v1"
)
IMU_VISUAL_LAB_MOTION_SCHEMA = "gt_calib.imu_mocap_visualization_motion.v1"
IMU_VISUAL_LAB_SOURCE = PROJECT_ROOT / "imu_mocap_visualization_lab"
IMU_VISUAL_LAB_PUBLIC = PUBLIC_ROOT / "downloads/imu-visual-lab"
IMU_VISUAL_LAB_WEB_PREFIX = "downloads/imu-visual-lab/"
IMU_VISUAL_LAB_TAKE_COUNT = 4
IMU_VISUAL_LAB_METHOD_COUNT = 7
IMU_VISUAL_LAB_VIDEO_COUNT = 28
IMU_VISUAL_LAB_TAKE_IDS = ("take005", "take006a", "take006b", "take007")
IMU_VISUAL_LAB_SEGMENTS = (
    (
        "take005",
        "camera_glove_recording_20260831_161402/Take_005",
        0,
        1867,
    ),
    (
        "take006a",
        "camera_glove_recording_20260831_161610/Take_006",
        0,
        1309,
    ),
    (
        "take006b",
        "camera_glove_recording_20260831_161610/Take_006",
        1309,
        1310,
    ),
    (
        "take007",
        "camera_glove_recording_20260831_161912/Take_007",
        0,
        1981,
    ),
)
IMU_VISUAL_LAB_CAMERA_INTRINSICS = {
    "take005": {
        "path": (
            "thor_new4_20260831_processed/"
            "camera_glove_recording_20260831_161402/"
            "rgbd_unpack/camera_1_intrinsics.json"
        ),
        "bytes": 2814,
        "sha256": "05090e1bd9d3f66a29010de6a43afddc99ac9ab07873761fb0c6b295874a642b",
    },
    "take006a": {
        "path": (
            "thor_new4_20260831_processed/"
            "camera_glove_recording_20260831_161610/"
            "rgbd_unpack/camera_1_intrinsics.json"
        ),
        "bytes": 2814,
        "sha256": "8d650632f8f0de8460a4bade8766329b6924d76a2e89481b149b833e8d502d1c",
    },
    "take006b": {
        "path": (
            "thor_new4_20260831_processed/"
            "camera_glove_recording_20260831_161610/"
            "rgbd_unpack/camera_1_intrinsics.json"
        ),
        "bytes": 2814,
        "sha256": "8d650632f8f0de8460a4bade8766329b6924d76a2e89481b149b833e8d502d1c",
    },
    "take007": {
        "path": (
            "thor_new4_20260831_processed/"
            "camera_glove_recording_20260831_161912/"
            "rgbd_unpack/camera_1_intrinsics.json"
        ),
        "bytes": 2814,
        "sha256": "7298ba3d82904c6d6b386b8fd58a3e2f773ead4f388702adce06161ef81287b8",
    },
}
IMU_VISUAL_LAB_METHOD_IDS = (
    "s2_continuous",
    "s2_gaussian",
    "kalman_rts",
    "posterior_75",
    "posterior_98",
    "rbf_self_fit",
    "guided_ik",
)
IMU_VISUAL_LAB_DISPLAY_CONTRACT = {
    "pose_drawn_on_every_frame": True,
    "validity_gate_hides_pose": False,
    "stale_color_change": False,
    "connector_or_error_lines": False,
    "per_frame_error_text": False,
}
IMU_VISUAL_LAB_ANATOMICAL_SOLVER = "constant-curvature exact endpoint IK"
IMU_VISUAL_LAB_ANATOMICAL_RATIO = 0.65
IMU_VISUAL_LAB_ANATOMICAL_THRESHOLD_DEG = 5.0
IMU_VISUAL_LAB_ANATOMICAL_LIMITS = {
    "pip_max_deg": 110.0,
    "dip_max_deg": 75.0,
    "thumb_max_deg": 115.0,
    "endpoint_p95_max_mm": 0.7,
    "bend_plane_temporal_p95_max_deg": 8.0,
    "bend_plane_temporal_max_deg": 45.0,
}
IMU_VISUAL_LAB_ANATOMICAL_ACCEPTANCE = {
    "no_opposite_active_bends": True,
    "pip_within_limit": True,
    "dip_within_limit": True,
    "thumb_within_limit": True,
    "fixed_bone_lengths": True,
    "endpoint_p95_within_0_7_mm": True,
    "bend_plane_temporal_p95_within_8_deg": True,
    "bend_plane_temporal_max_within_45_deg": True,
}
IMU_VISUAL_LAB_ANATOMICAL_DISTRIBUTIONS = (
    "pip_flexion_deg",
    "dip_flexion_deg",
    "thumb_flexion_deg",
    "fixed_bone_length_drift_mm",
    "smoothed_cmm_base_tip_endpoint_epe_mm",
    "bend_plane_temporal_delta_deg",
)
IMU_VISUAL_LAB_MOCAP_CHAINS = [[0, 2, 1], [0, 4, 3], [0, 6, 5], [0, 8, 7], [0, 10, 9]]
IMU_VISUAL_LAB_RESULT_CHAINS = [
    [0, 1, 2, 3],
    [0, 4, 5, 6, 7],
    [0, 8, 9, 10, 11],
    [0, 12, 13, 14, 15],
    [0, 16, 17, 18, 19],
]
IMU_VISUAL_LAB_MOTION_LAYERS = [
    {
        "id": "mocap-left",
        "kind": "mocap",
        "side": "left",
        "joint_offset": 0,
        "joint_count": 11,
        "chains": IMU_VISUAL_LAB_MOCAP_CHAINS,
    },
    {
        "id": "mocap-right",
        "kind": "mocap",
        "side": "right",
        "joint_offset": 11,
        "joint_count": 11,
        "chains": IMU_VISUAL_LAB_MOCAP_CHAINS,
    },
    {
        "id": "imu-left",
        "kind": "result",
        "side": "left",
        "joint_offset": 22,
        "joint_count": 20,
        "chains": IMU_VISUAL_LAB_RESULT_CHAINS,
    },
    {
        "id": "imu-right",
        "kind": "result",
        "side": "right",
        "joint_offset": 42,
        "joint_count": 20,
        "chains": IMU_VISUAL_LAB_RESULT_CHAINS,
    },
]
IMU_VISUAL_LAB_DATASET_SCOPE = {
    "root": "thor_new4_20260831_processed",
    "policy": "new dataset only",
    "published_segments": [
        {
            "take_id": take_id,
            "source": source,
            "source_first": source_first,
            "frame_count": frame_count,
        }
        for take_id, source, source_first, frame_count in IMU_VISUAL_LAB_SEGMENTS
    ],
    "take006_split": {
        "reason": (
            "publish four action-review segments from three complete "
            "synchronized action recordings"
        ),
        "windows_are_disjoint": True,
        "source_output_frames": 2619,
        "split_output_index": 1309,
    },
}
IMU_COMPARISON_VIDEO_CONTRACTS = (
    (
        1,
        "take01-imu-solved-vs-bvh",
        "01_take01_imu_solved_vs_bvh_mocap.mp4",
        1747,
        "gt_calib.imu_solved_pose_vs_skeleton_bvh.v1",
    ),
    (
        2,
        "take02-imu-solved-vs-bvh",
        "02_take02_imu_solved_vs_bvh_mocap.mp4",
        1805,
        "gt_calib.imu_solved_pose_vs_skeleton_bvh.v1",
    ),
    (
        3,
        "take03-imu-solved-vs-bvh",
        "03_take03_imu_solved_vs_bvh_mocap.mp4",
        1799,
        "gt_calib.imu_solved_pose_vs_skeleton_bvh.v1",
    ),
    (
        4,
        "take007-imu-solved-vs-cmm",
        "04_take007_imu_solved_vs_cmm_mocap.mp4",
        1981,
        "gt_calib.imu_solved_pose_vs_cmm_markers.v1",
    ),
)
GENERATED_DELIVERY_WEB_PREFIXES = (
    FINAL_DELIVERY_WEB_PREFIX,
    IMU_COMPARISON_WEB_PREFIX,
    IMU_VISUAL_LAB_WEB_PREFIX,
)
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


def _validate_faststart_mp4_structure(path: Path) -> None:
    """Check the top-level ISO-BMFF boxes without requiring ffprobe.

    This is intentionally a minimum clean-clone check, not a codec decoder.  It
    prevents arbitrary text files from satisfying a self-reported MP4 contract
    and independently verifies that the metadata box precedes media payload.
    """

    size = path.stat().st_size
    offset = 0
    boxes: list[tuple[bytes, int]] = []
    with path.open("rb") as handle:
        while offset < size:
            remaining = size - offset
            if remaining < 8:
                raise ValueError(f"Truncated MP4 box header: {path}")
            handle.seek(offset)
            header = handle.read(8)
            box_size = int.from_bytes(header[:4], "big")
            box_type = header[4:8]
            header_size = 8
            if box_size == 1:
                extended = handle.read(8)
                if len(extended) != 8:
                    raise ValueError(f"Truncated extended MP4 box header: {path}")
                box_size = int.from_bytes(extended, "big")
                header_size = 16
            elif box_size == 0:
                box_size = remaining
            if box_size < header_size or box_size > remaining:
                raise ValueError(f"Invalid top-level MP4 box size in {path}")
            boxes.append((box_type, offset))
            offset += box_size

    offsets: dict[bytes, list[int]] = {}
    for box_type, box_offset in boxes:
        offsets.setdefault(box_type, []).append(box_offset)
    if b"ftyp" not in offsets or b"moov" not in offsets or b"mdat" not in offsets:
        raise ValueError(f"MP4 is missing ftyp/moov/mdat top-level boxes: {path}")
    if min(offsets[b"moov"]) > min(offsets[b"mdat"]):
        raise ValueError(f"MP4 moov box does not precede mdat: {path}")


def _load_json_object(path: Path, *, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


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


def validate_imu_comparison_for_web(source: Path) -> list[dict]:
    """Validate the tracked four-video supplemental comparison bundle."""

    source_input = Path(source).expanduser()
    if source_input.is_symlink():
        raise ValueError(f"IMU comparison delivery is a symlink: {source_input}")
    source = source_input.resolve()
    if not source.is_dir():
        raise ValueError(f"IMU comparison delivery is missing: {source}")
    if (source.with_name(source.name + ".staging")).exists():
        raise ValueError("IMU comparison staging directory is still present")

    files: dict[str, Path] = {}
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlinks are forbidden in IMU comparison: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Non-regular IMU comparison entry: {path}")
        ensure_cloudflare_size(path)
        files[path.relative_to(source).as_posix()] = path

    required = {
        "README.md",
        "index.html",
        "styles.css",
        "manifest.json",
        "validation.json",
        "SHA256SUMS.txt",
    }
    missing = sorted(required - set(files))
    if missing:
        raise ValueError(f"IMU comparison delivery is missing files: {missing}")
    manifest = _load_json_object(
        files["manifest.json"], label="IMU comparison manifest"
    )
    videos = manifest.get("videos")
    if (
        manifest.get("schema") != IMU_COMPARISON_SCHEMA
        or manifest.get("status") != "pass"
        or manifest.get("comparison_count") != 4
        or not isinstance(videos, list)
        or len(videos) != 4
        or manifest.get("canonical_final_nine_unchanged") is not True
    ):
        raise ValueError("IMU comparison manifest must be a clean four-video supplement")
    policy = manifest.get("display_policy", {})
    if not (
        policy.get("sampling") == "nearest observed solver row"
        and policy.get("pose_interpolation") is False
        and policy.get("pose_smoothing") is False
        and policy.get("validity_gate_hides_pose") is False
        and policy.get("stale_pose_is_drawn") is True
        and policy.get("stale_pose_in_strict_metrics") is False
    ):
        raise ValueError("IMU comparison display/scientific-mask separation is invalid")

    manual_profile = manifest.get("manual_profile")
    if not isinstance(manual_profile, dict):
        raise ValueError("IMU comparison packaged manual profile is missing")
    if manual_profile.get("path") != "calibration/applied_manual_profile.json":
        raise ValueError("IMU comparison packaged manual profile path is not canonical")
    packaged_profile = _delivery_artifact(
        source,
        manual_profile.get("path"),
        field="imu.manual_profile.path",
        expected_hash=manual_profile.get("sha256"),
        expected_bytes=manual_profile.get("bytes"),
    )
    profile_payload = _load_json_object(
        packaged_profile, label="IMU comparison packaged manual profile"
    )
    if profile_payload.get("schema") != "gt_calib.final_nine_manual_xyz.v1":
        raise ValueError("IMU comparison packaged manual profile schema is invalid")
    profile_source = manual_profile.get("source")
    if not isinstance(profile_source, dict):
        raise ValueError("IMU comparison manual profile source provenance is missing")
    source_path = _safe_delivery_relative_path(
        profile_source.get("path"), field="imu.manual_profile.source.path"
    )
    if (
        source_path.suffix.lower() != ".json"
        or profile_source.get("bytes") != manual_profile.get("bytes")
        or profile_source.get("sha256") != manual_profile.get("sha256")
    ):
        raise ValueError("IMU comparison manual profile source provenance is invalid")

    orders: list[int] = []
    names: list[str] = []
    for index, (item, contract) in enumerate(
        zip(videos, IMU_COMPARISON_VIDEO_CONTRACTS, strict=True)
    ):
        if not isinstance(item, dict):
            raise ValueError(f"IMU comparison video entry {index} is not an object")
        (
            expected_order,
            expected_id,
            expected_filename,
            expected_frame_count,
            expected_metrics_schema,
        ) = contract
        order = item.get("order")
        video_id = item.get("id")
        filename = item.get("filename")
        if (
            type(order) is not int
            or order != expected_order
            or video_id != expected_id
            or filename != expected_filename
        ):
            raise ValueError(f"Canonical IMU comparison identity mismatch at index {index}")
        name = _safe_delivery_relative_path(
            filename, field=f"imu.videos[{index}].filename"
        )
        if len(name.parts) != 1 or name.suffix.lower() != ".mp4":
            raise ValueError(f"IMU comparison filename must be one MP4 basename: {filename}")
        video = _delivery_artifact(
            source,
            f"videos/{name.name}",
            field=f"imu.videos[{index}].filename",
            expected_hash=item.get("sha256"),
            expected_bytes=item.get("bytes"),
        )
        _validate_faststart_mp4_structure(video)
        if (
            item.get("codec") != "h264"
            or item.get("pixel_format") != "yuv420p"
            or item.get("faststart") is not True
            or item.get("frame_count") != expected_frame_count
        ):
            raise ValueError(f"IMU comparison browser media contract failed: {video}")
        _delivery_artifact(
            source,
            item.get("poster"),
            field=f"imu.videos[{index}].poster",
            expected_hash=item.get("poster_sha256"),
        )
        metrics = _delivery_artifact(
            source,
            item.get("metrics"),
            field=f"imu.videos[{index}].metrics",
            expected_hash=item.get("metrics_sha256"),
        )
        frame_map = _delivery_artifact(
            source,
            item.get("frame_map"),
            field=f"imu.videos[{index}].frame_map",
            expected_hash=item.get("frame_map_sha256"),
        )

        metrics_payload = _load_json_object(
            metrics, label=f"IMU comparison metrics for {expected_id}"
        )
        if (
            metrics_payload.get("schema") != expected_metrics_schema
            or metrics_payload.get("status") != "comparison_complete"
        ):
            raise ValueError(f"IMU comparison metrics contract failed: {expected_id}")
        metrics_sides = metrics_payload.get("sides")
        manifest_summary = item.get("summary")
        if (
            not isinstance(metrics_sides, dict)
            or set(metrics_sides) != {"left", "right"}
            or not isinstance(manifest_summary, dict)
            or set(manifest_summary) != {"left", "right"}
        ):
            raise ValueError(f"IMU comparison side metrics are incomplete: {expected_id}")
        strict_metric_key = (
            "strict_25ms_scientific_subset"
            if expected_metrics_schema
            == "gt_calib.imu_solved_pose_vs_skeleton_bvh.v1"
            else "evaluation_strict_20ms_camera_valid_subset"
        )
        tip_metric_key = (
            "non_thumb_fingertip_epe_mm"
            if expected_metrics_schema
            == "gt_calib.imu_solved_pose_vs_skeleton_bvh.v1"
            else "five_tip_epe_mm"
        )
        strict_frame_counts: dict[str, int] = {}
        for side in ("left", "right"):
            side_metrics = metrics_sides[side]
            side_summary = manifest_summary[side]
            if not isinstance(side_metrics, dict) or not isinstance(side_summary, dict):
                raise ValueError(f"IMU comparison {side} metrics are invalid: {expected_id}")
            strict_metrics = side_metrics.get(strict_metric_key)
            if not isinstance(strict_metrics, dict):
                raise ValueError(f"IMU comparison strict metrics are missing: {expected_id}")
            pooled = strict_metrics.get("pooled_joint_epe_mm")
            tips = strict_metrics.get(tip_metric_key)
            if not isinstance(pooled, dict) or not isinstance(tips, dict):
                raise ValueError(f"IMU comparison strict distributions are missing: {expected_id}")
            strict_frames = strict_metrics.get("frame_count")
            if type(strict_frames) is not int or not 0 <= strict_frames <= expected_frame_count:
                raise ValueError(f"IMU comparison strict frame count is invalid: {expected_id}")
            expected_summary = {
                "strict_frames": strict_frames,
                "joint_median_mm": pooled.get("median"),
                "joint_p95_mm": pooled.get("p95"),
                "tip_median_mm": tips.get("median"),
            }
            if side_summary != expected_summary:
                raise ValueError(f"IMU comparison manifest summary mismatch: {expected_id} {side}")
            strict_frame_counts[side] = strict_frames

        with frame_map.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required_columns = {
                "output_frame",
                "side",
                "solver_sample_index",
                "solver_sample_age_ms",
                "strict_timing_valid",
            }
            if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
                raise ValueError(f"IMU comparison frame-map columns are invalid: {expected_id}")
            frame_rows = list(reader)
        if len(frame_rows) != expected_frame_count * 2:
            raise ValueError(f"IMU comparison frame-map row count mismatch: {expected_id}")
        frame_keys: set[tuple[int, str]] = set()
        observed_strict_counts = {"left": 0, "right": 0}
        last_sample_index = {"left": -1, "right": -1}
        take007 = expected_metrics_schema == "gt_calib.imu_solved_pose_vs_cmm_markers.v1"
        calibration_stop = round(expected_frame_count * 0.20)
        for row in frame_rows:
            try:
                output_frame = int(row["output_frame"])
                side = row["side"]
                sample_index = int(row["solver_sample_index"])
                sample_age_ms = float(row["solver_sample_age_ms"])
                strict_value = int(row["strict_timing_valid"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"IMU comparison frame-map row is malformed: {expected_id}") from exc
            if (
                not 0 <= output_frame < expected_frame_count
                or side not in observed_strict_counts
                or sample_index < 0
                or not math.isfinite(sample_age_ms)
                or sample_age_ms < 0.0
                or strict_value not in (0, 1)
            ):
                raise ValueError(f"IMU comparison frame-map row is invalid: {expected_id}")
            if sample_index < last_sample_index[side]:
                raise ValueError(f"IMU comparison solver sample order regressed: {expected_id}")
            last_sample_index[side] = sample_index
            key = (output_frame, side)
            if key in frame_keys:
                raise ValueError(f"Duplicate IMU comparison frame-map row: {expected_id} {key}")
            frame_keys.add(key)
            if take007:
                expected_phase = (
                    "calibration" if output_frame < calibration_stop else "evaluation"
                )
                if row.get("phase") != expected_phase:
                    raise ValueError(f"IMU comparison frame-map phase is invalid: {expected_id}")
                scored_strict = strict_value and expected_phase == "evaluation"
                if strict_value and sample_age_ms > 20.0 + 1e-6:
                    raise ValueError(f"IMU comparison strict sample is stale: {expected_id}")
            else:
                scored_strict = strict_value
                if strict_value and sample_age_ms > 12.5 + 1e-6:
                    raise ValueError(f"IMU comparison strict sample is stale: {expected_id}")
            if scored_strict:
                observed_strict_counts[side] += 1
        if observed_strict_counts != strict_frame_counts:
            raise ValueError(f"IMU comparison strict CSV counts mismatch: {expected_id}")
        orders.append(order)
        names.append(name.name)
    if orders != [1, 2, 3, 4] or len(set(names)) != 4:
        raise ValueError("IMU comparison orders and filenames must be canonical 1..4")
    actual_names = {
        path.name
        for path in (source / "videos").iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix.lower() == ".mp4"
    }
    if actual_names != set(names):
        raise ValueError("IMU comparison videos directory disagrees with manifest")

    validation = _load_json_object(
        files["validation.json"], label="IMU comparison validation"
    )
    if (
        validation.get("schema")
        != "gt_calib.imu_mocap_comparison_validation.v1"
        or validation.get("status") != "pass"
        or validation.get("comparison_count") != 4
        or validation.get("full_decode") is not True
        or validation.get("failures") != []
    ):
        raise ValueError("IMU comparison validation is not a clean full-decode pass")

    checksum_entries: dict[str, str] = {}
    for line_number, line in enumerate(
        files["SHA256SUMS.txt"].read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"Malformed IMU comparison checksum line {line_number}")
        digest, value = match.groups()
        relative = _safe_delivery_relative_path(
            value, field=f"imu.SHA256SUMS.txt:{line_number}"
        ).as_posix()
        if relative in checksum_entries:
            raise ValueError(f"Duplicate IMU comparison checksum entry: {relative}")
        checksum_entries[relative] = digest
    expected_paths = set(files) - {"SHA256SUMS.txt"}
    if set(checksum_entries) != expected_paths:
        raise ValueError("IMU comparison checksum inventory mismatch")
    for relative, expected_hash in checksum_entries.items():
        if sha256(files[relative]) != expected_hash:
            raise ValueError(f"IMU comparison checksum mismatch: {relative}")

    return [
        {
            "path": f"downloads/imu-mocap/{relative}",
            "source": f"imu_mocap_comparison_delivery/{relative}",
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for relative, path in sorted(files.items())
    ]


def mirror_imu_comparison(source: Path, destination: Path) -> list[dict]:
    """Atomically publish the independently validated comparison supplement."""

    source = Path(source).expanduser()
    destination_input = Path(destination).expanduser()
    if destination_input.is_symlink():
        raise ValueError(f"Unsafe IMU comparison web destination: {destination_input}")
    destination = destination_input.resolve()
    source_entries = validate_imu_comparison_for_web(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Unsafe IMU comparison web destination: {destination}")

    staging_root = Path(
        tempfile.mkdtemp(prefix=".imu-mocap-publish-", dir=destination.parent)
    )
    staged = staging_root / "imu-mocap"
    backup = staging_root / "previous-imu-mocap"
    moved_previous = False
    try:
        shutil.copytree(source, staged, symlinks=False)
        staged_entries = validate_imu_comparison_for_web(staged)
        if source_entries != staged_entries:
            raise ValueError("Staged IMU comparison differs from validated source")
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


def _validate_imu_visual_lab_anatomical_ik(
    metrics: dict,
    *,
    video_id: str,
) -> None:
    """Independently reject missing, stale, or self-inconsistent IK acceptance."""

    preparation = metrics.get("take_preparation")
    guided = preparation.get("cmm_guided_ik") if isinstance(preparation, dict) else None
    if not isinstance(guided, dict) or set(guided) != {"left", "right"}:
        raise ValueError(
            f"IMU visualization lab anatomical IK contract missing: {video_id}"
        )
    distribution_keys = {"count", "mean", "median", "p95", "max", "rmse"}
    required_keys = {
        "solver",
        "dip_to_pip_flexion_ratio",
        "active_bend_threshold_deg",
        "opposite_bend_count",
        "active_bend_pair_count",
        "opposite_bend_fraction_active_gt_5deg",
        *IMU_VISUAL_LAB_ANATOMICAL_DISTRIBUTIONS,
        "limits",
        "acceptance",
    }
    for side in ("left", "right"):
        side_payload = guided.get(side)
        validation = (
            side_payload.get("anatomical_validation")
            if isinstance(side_payload, dict)
            else None
        )
        if not isinstance(validation, dict) or set(validation) != required_keys:
            raise ValueError(
                f"IMU visualization lab anatomical IK contract failed: {video_id}/{side}"
            )
        ratio = validation.get("dip_to_pip_flexion_ratio")
        threshold = validation.get("active_bend_threshold_deg")
        active_count = validation.get("active_bend_pair_count")
        opposite_fraction = validation.get("opposite_bend_fraction_active_gt_5deg")
        if (
            validation.get("solver") != IMU_VISUAL_LAB_ANATOMICAL_SOLVER
            or isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or float(ratio) != IMU_VISUAL_LAB_ANATOMICAL_RATIO
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or float(threshold) != IMU_VISUAL_LAB_ANATOMICAL_THRESHOLD_DEG
            or type(validation.get("opposite_bend_count")) is not int
            or validation.get("opposite_bend_count") != 0
            or type(active_count) is not int
            or active_count <= 0
            or isinstance(opposite_fraction, bool)
            or not isinstance(opposite_fraction, (int, float))
            or float(opposite_fraction) != 0.0
            or validation.get("limits") != IMU_VISUAL_LAB_ANATOMICAL_LIMITS
            or validation.get("acceptance")
            != IMU_VISUAL_LAB_ANATOMICAL_ACCEPTANCE
        ):
            raise ValueError(
                f"IMU visualization lab anatomical IK acceptance failed: {video_id}/{side}"
            )

        distributions: dict[str, dict] = {}
        for name in IMU_VISUAL_LAB_ANATOMICAL_DISTRIBUTIONS:
            values = validation.get(name)
            if (
                not isinstance(values, dict)
                or set(values) != distribution_keys
                or type(values.get("count")) is not int
                or values.get("count") <= 0
                or any(
                    isinstance(values.get(field), bool)
                    or not isinstance(values.get(field), (int, float))
                    or not math.isfinite(float(values[field]))
                    or float(values[field]) < 0.0
                    for field in ("mean", "median", "p95", "max", "rmse")
                )
            ):
                raise ValueError(
                    "IMU visualization lab anatomical IK distribution failed: "
                    f"{video_id}/{side}/{name}"
                )
            distributions[name] = values
        if (
            float(distributions["pip_flexion_deg"]["max"])
            > IMU_VISUAL_LAB_ANATOMICAL_LIMITS["pip_max_deg"] + 1e-7
            or float(distributions["dip_flexion_deg"]["max"])
            > IMU_VISUAL_LAB_ANATOMICAL_LIMITS["dip_max_deg"] + 1e-7
            or float(distributions["thumb_flexion_deg"]["max"])
            > IMU_VISUAL_LAB_ANATOMICAL_LIMITS["thumb_max_deg"] + 1e-7
            or float(distributions["fixed_bone_length_drift_mm"]["max"]) > 1e-7
            or float(
                distributions["smoothed_cmm_base_tip_endpoint_epe_mm"]["p95"]
            )
            > IMU_VISUAL_LAB_ANATOMICAL_LIMITS["endpoint_p95_max_mm"]
            or float(distributions["bend_plane_temporal_delta_deg"]["p95"])
            > IMU_VISUAL_LAB_ANATOMICAL_LIMITS[
                "bend_plane_temporal_p95_max_deg"
            ]
            or float(distributions["bend_plane_temporal_delta_deg"]["max"])
            > IMU_VISUAL_LAB_ANATOMICAL_LIMITS[
                "bend_plane_temporal_max_deg"
            ]
        ):
            raise ValueError(
                f"IMU visualization lab anatomical IK measured limits failed: {video_id}/{side}"
            )


def _validate_imu_visual_lab_metrics(
    path: Path,
    *,
    video_id: str,
    take_id: str,
    method_id: str,
    source: str,
    source_first: int,
    frame_count: int,
    summary: object,
) -> None:
    """Validate formal metrics and the denormalized manifest summary."""

    metrics = _load_json_object(
        path, label=f"IMU visualization lab metrics for {video_id}"
    )
    if (
        metrics.get("schema") != IMU_VISUAL_LAB_METRICS_SCHEMA
        or metrics.get("status") != "complete"
        or metrics.get("take_id") != take_id
        or metrics.get("method_id") != method_id
        or metrics.get("source") != source
        or metrics.get("source_first") != source_first
        or type(metrics.get("rendered_frames")) is not int
        or metrics.get("rendered_frames") != frame_count
        or metrics.get("display_contract") != IMU_VISUAL_LAB_DISPLAY_CONTRACT
    ):
        raise ValueError(
            f"IMU visualization lab metrics contract failed: {video_id}"
        )
    source_assets = metrics.get("source_assets")
    if (
        not isinstance(source_assets, dict)
        or source_assets.get("camera_intrinsics")
        != IMU_VISUAL_LAB_CAMERA_INTRINSICS[take_id]
    ):
        raise ValueError(
            "IMU visualization lab action camera intrinsics provenance failed: "
            f"{video_id}"
        )
    _validate_imu_visual_lab_anatomical_ik(metrics, video_id=video_id)

    pooled = metrics.get("pooled_sides")
    primary = pooled.get("primary_reference_epe_mm") if isinstance(pooled, dict) else None
    high_frequency = (
        pooled.get("root_relative_high_frequency_residual_mm")
        if isinstance(pooled, dict)
        else None
    )
    primary_median = primary.get("median") if isinstance(primary, dict) else None
    primary_p95 = primary.get("p95") if isinstance(primary, dict) else None
    high_frequency_p95 = (
        high_frequency.get("p95") if isinstance(high_frequency, dict) else None
    )
    pooled_values = (primary_median, primary_p95, high_frequency_p95)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in pooled_values
    ):
        raise ValueError(
            f"IMU visualization lab metrics pooled contract failed: {video_id}"
        )

    expected_summary = {
        "reference_median_mm": primary_median,
        "reference_p95_mm": primary_p95,
        "jitter_p95_mm": high_frequency_p95,
        "finite_frames": frame_count,
    }
    if (
        not isinstance(summary, dict)
        or set(summary) != set(expected_summary)
        or any(
            isinstance(summary.get(field), bool)
            or not isinstance(summary.get(field), (int, float))
            or not math.isfinite(float(summary[field]))
            for field in (
                "reference_median_mm",
                "reference_p95_mm",
                "jitter_p95_mm",
            )
        )
        or type(summary.get("finite_frames")) is not int
        or summary != expected_summary
    ):
        raise ValueError(
            f"IMU visualization lab manifest/metrics summary mismatch: {video_id}"
        )


def _validate_imu_visual_lab_motion(
    path: Path,
    *,
    video_id: str,
    take_id: str,
    method_id: str,
    frame_count: int,
    fps: float,
) -> float:
    """Independently validate one browser-decodable dual-layer motion asset."""

    motion = _load_json_object(
        path, label=f"IMU visualization lab motion for {video_id}"
    )
    encoding = motion.get("encoding")
    layers = motion.get("layers")
    frames = motion.get("frames")
    validation = motion.get("validation")
    view = motion.get("view")
    if (
        motion.get("schema") != IMU_VISUAL_LAB_MOTION_SCHEMA
        or motion.get("take_id") != take_id
        or motion.get("method_id") != method_id
        or motion.get("units") != "mm"
        or type(motion.get("frame_count")) is not int
        or motion.get("frame_count") != frame_count
        or isinstance(motion.get("fps"), bool)
        or not isinstance(motion.get("fps"), (int, float))
        or not math.isfinite(float(motion["fps"]))
        or float(motion["fps"]) != fps
    ):
        raise ValueError(
            f"IMU visualization lab motion identity contract failed: {video_id}"
        )

    if not isinstance(encoding, dict):
        raise ValueError(
            f"IMU visualization lab motion encoding contract failed: {video_id}"
        )
    quantum_mm = encoding.get("quantum_mm")
    origin_mm = encoding.get("origin_mm")
    joint_count = encoding.get("joint_count")
    values_per_frame = encoding.get("values_per_frame")
    if (
        encoding.get("kind") != "frame-major-flat-int32-json"
        or encoding.get("components") != "XYZ"
        or isinstance(quantum_mm, bool)
        or not isinstance(quantum_mm, (int, float))
        or not math.isfinite(float(quantum_mm))
        or float(quantum_mm) <= 0.0
        or not isinstance(origin_mm, list)
        or len(origin_mm) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in origin_mm
        )
        or joint_count != 62
        or type(values_per_frame) is not int
        or values_per_frame != joint_count * 3
    ):
        raise ValueError(
            f"IMU visualization lab motion encoding contract failed: {video_id}"
        )

    if layers != IMU_VISUAL_LAB_MOTION_LAYERS:
        raise ValueError(
            f"IMU visualization lab motion layers contract failed: {video_id}"
        )

    if (
        not isinstance(frames, list)
        or len(frames) != frame_count
        or not all(
            isinstance(frame, list)
            and len(frame) == values_per_frame
            and all(
                type(value) is int and -(2**31) <= value < 2**31
                for value in frame
            )
            for frame in frames
        )
    ):
        raise ValueError(
            f"IMU visualization lab motion frames contract failed: {video_id}"
        )

    if not isinstance(validation, dict):
        raise ValueError(
            f"IMU visualization lab motion validation contract failed: {video_id}"
        )
    if not isinstance(view, dict):
        raise ValueError(
            f"IMU visualization lab motion view contract failed: {video_id}"
        )
    reference_extent = view.get("reference_extent_mm")
    if (
        view.get("focus") != "per-frame midpoint of MOCAP left/right wrists"
        or view.get("fixed_across_methods_for_take") is not True
        or isinstance(reference_extent, bool)
        or not isinstance(reference_extent, (int, float))
        or not math.isfinite(float(reference_extent))
        or float(reference_extent) <= 0.0
    ):
        raise ValueError(
            f"IMU visualization lab motion view contract failed: {video_id}"
        )
    maximum_error = validation.get("maximum_quantization_error_mm")
    if (
        validation.get("status") != "pass"
        or type(validation.get("finite_frames")) is not int
        or validation.get("finite_frames") != frame_count
        or isinstance(maximum_error, bool)
        or not isinstance(maximum_error, (int, float))
        or not math.isfinite(float(maximum_error))
        or float(maximum_error) < 0.0
        or float(maximum_error) > float(quantum_mm) / 2.0 + 1e-9
    ):
        raise ValueError(
            f"IMU visualization lab motion validation contract failed: {video_id}"
        )
    return float(reference_extent)


def validate_imu_visual_lab_for_web(source: Path) -> list[dict]:
    """Validate the closed 4-take x 7-method visualization-lab bundle.

    The prebuilt lab is intentionally its own trust boundary.  Cloudflare
    assembly has no raw-data or media-decoder dependency, but it still checks
    every declared byte/hash, MP4 fast-start structure, the complete method
    matrix, the full-decode validation receipt, and the exact file inventory.
    """

    source_input = Path(source).expanduser()
    if source_input.is_symlink():
        raise ValueError(f"IMU visualization lab is a symlink: {source_input}")
    source = source_input.resolve()
    if not source.is_dir():
        raise ValueError(f"IMU visualization lab is missing: {source}")
    if (source.with_name(source.name + ".staging")).exists():
        raise ValueError("IMU visualization lab staging directory is still present")

    files: dict[str, Path] = {}
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlinks are forbidden in IMU visualization lab: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Non-regular IMU visualization lab entry: {path}")
        ensure_cloudflare_size(path)
        files[path.relative_to(source).as_posix()] = path

    required_files = {
        "README.md",
        "index.html",
        "styles.css",
        "app.js",
        "manifest.json",
        "validation.json",
        "SHA256SUMS.txt",
    }
    missing = sorted(required_files - set(files))
    if missing:
        raise ValueError(f"IMU visualization lab is missing files: {missing}")

    manifest = _load_json_object(
        files["manifest.json"], label="IMU visualization lab manifest"
    )
    videos = manifest.get("videos")
    methods = manifest.get("methods")
    if (
        manifest.get("schema") != IMU_VISUAL_LAB_SCHEMA
        or manifest.get("status") != "pass"
        or manifest.get("take_count") != IMU_VISUAL_LAB_TAKE_COUNT
        or manifest.get("method_count") != IMU_VISUAL_LAB_METHOD_COUNT
        or manifest.get("video_count") != IMU_VISUAL_LAB_VIDEO_COUNT
        or not isinstance(videos, list)
        or len(videos) != IMU_VISUAL_LAB_VIDEO_COUNT
    ):
        raise ValueError(
            "IMU visualization lab manifest must be a clean 4-take x "
            "7-method, 28-video pass"
        )
    if (
        not isinstance(methods, list)
        or [method.get("id") if isinstance(method, dict) else None for method in methods]
        != list(IMU_VISUAL_LAB_METHOD_IDS)
    ):
        raise ValueError(
            "IMU visualization lab manifest methods must match the canonical IDs"
        )
    if manifest.get("dataset_scope") != IMU_VISUAL_LAB_DATASET_SCOPE:
        raise ValueError(
            "IMU visualization lab dataset_scope must lock the canonical "
            "thor_new4 source windows"
        )
    expected_segments = {
        take_id: {
            "source": source,
            "source_first": source_first,
            "frame_count": frame_count,
        }
        for take_id, source, source_first, frame_count in IMU_VISUAL_LAB_SEGMENTS
    }

    required_video_fields = {
        "order",
        "id",
        "take_id",
        "method_id",
        "source",
        "source_first",
        "filename",
        "poster",
        "metrics",
        "motion",
        "summary",
        "frame_count",
        "codec",
        "pixel_format",
        "width",
        "height",
        "fps",
        "bytes",
        "sha256",
        "poster_sha256",
        "metrics_sha256",
        "motion_bytes",
        "motion_sha256",
        "faststart",
    }
    orders: list[int] = []
    video_ids: list[str] = []
    take_ids: set[str] = set()
    method_ids: set[str] = set()
    take_method_pairs: set[tuple[str, str]] = set()
    video_names: set[str] = set()
    asset_paths: set[str] = set()
    frame_counts_by_take: dict[str, int] = {}
    reference_extents_by_take: dict[str, float] = {}
    for index, item in enumerate(videos):
        if not isinstance(item, dict):
            raise ValueError(f"IMU visualization lab video entry {index} is not an object")
        missing_fields = sorted(required_video_fields - set(item))
        if missing_fields:
            raise ValueError(
                f"IMU visualization lab video entry {index} is missing fields: "
                f"{missing_fields}"
            )

        order = item.get("order")
        video_id = item.get("id")
        take_id = item.get("take_id")
        method_id = item.get("method_id")
        frame_count = item.get("frame_count")
        if (
            type(order) is not int
            or not isinstance(video_id, str)
            or not video_id
            or not isinstance(take_id, str)
            or not take_id
            or not isinstance(method_id, str)
            or not method_id
            or any("/" in value or "\\" in value for value in (video_id, take_id, method_id))
            or type(frame_count) is not int
            or frame_count <= 0
        ):
            raise ValueError(f"Invalid IMU visualization lab identity at index {index}")
        expected_take_id = IMU_VISUAL_LAB_TAKE_IDS[
            index // IMU_VISUAL_LAB_METHOD_COUNT
        ]
        expected_method_id = IMU_VISUAL_LAB_METHOD_IDS[
            index % IMU_VISUAL_LAB_METHOD_COUNT
        ]
        expected_video_id = f"{expected_take_id}-{expected_method_id}"
        expected_segment = expected_segments[expected_take_id]
        if (
            order != index + 1
            or take_id != expected_take_id
            or method_id != expected_method_id
            or video_id != expected_video_id
            or item.get("source") != expected_segment["source"]
            or item.get("source_first") != expected_segment["source_first"]
            or frame_count != expected_segment["frame_count"]
        ):
            raise ValueError(
                "IMU visualization lab canonical take-major/method-minor identity "
                f"failed at index {index}: expected {expected_video_id}"
            )

        filename = _safe_delivery_relative_path(
            item.get("filename"), field=f"imu_visual_lab.videos[{index}].filename"
        )
        expected_stem = f"{index + 1:02d}_{expected_take_id}_{expected_method_id}"
        if (
            len(filename.parts) != 1
            or filename.name != f"{expected_stem}.mp4"
        ):
            raise ValueError(
                f"IMU visualization lab filename must be one MP4 basename: {filename}"
            )
        video_relative = f"videos/{filename.name}"
        video = _delivery_artifact(
            source,
            video_relative,
            field=f"imu_visual_lab.videos[{index}].filename",
            expected_hash=item.get("sha256"),
            expected_bytes=item.get("bytes"),
        )
        _validate_faststart_mp4_structure(video)

        fps = item.get("fps")
        if (
            item.get("codec") != "h264"
            or item.get("pixel_format") != "yuv420p"
            or item.get("width") != 960
            or item.get("height") != 540
            or isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or float(fps) != 30.0
            or item.get("faststart") is not True
        ):
            raise ValueError(
                f"IMU visualization lab browser media contract failed: {video_id}"
            )

        poster_relative = _safe_delivery_relative_path(
            item.get("poster"), field=f"imu_visual_lab.videos[{index}].poster"
        )
        metrics_relative = _safe_delivery_relative_path(
            item.get("metrics"), field=f"imu_visual_lab.videos[{index}].metrics"
        )
        motion_relative = _safe_delivery_relative_path(
            item.get("motion"), field=f"imu_visual_lab.videos[{index}].motion"
        )
        if (
            poster_relative.as_posix() != f"posters/{expected_stem}.jpg"
        ):
            raise ValueError(f"Invalid IMU visualization lab poster path: {poster_relative}")
        if (
            metrics_relative.as_posix() != f"metrics/{expected_stem}.json"
        ):
            raise ValueError(f"Invalid IMU visualization lab metrics path: {metrics_relative}")
        if (
            motion_relative.as_posix() != f"motions/{expected_stem}.json"
        ):
            raise ValueError(f"Invalid IMU visualization lab motion path: {motion_relative}")
        _delivery_artifact(
            source,
            poster_relative.as_posix(),
            field=f"imu_visual_lab.videos[{index}].poster",
            expected_hash=item.get("poster_sha256"),
        )
        metrics = _delivery_artifact(
            source,
            metrics_relative.as_posix(),
            field=f"imu_visual_lab.videos[{index}].metrics",
            expected_hash=item.get("metrics_sha256"),
        )
        _validate_imu_visual_lab_metrics(
            metrics,
            video_id=video_id,
            take_id=take_id,
            method_id=method_id,
            source=expected_segment["source"],
            source_first=expected_segment["source_first"],
            frame_count=frame_count,
            summary=item.get("summary"),
        )
        motion = _delivery_artifact(
            source,
            motion_relative.as_posix(),
            field=f"imu_visual_lab.videos[{index}].motion",
            expected_hash=item.get("motion_sha256"),
            expected_bytes=item.get("motion_bytes"),
        )
        reference_extent = _validate_imu_visual_lab_motion(
            motion,
            video_id=video_id,
            take_id=take_id,
            method_id=method_id,
            frame_count=frame_count,
            fps=float(fps),
        )
        previous_extent = reference_extents_by_take.setdefault(
            take_id, reference_extent
        )
        if previous_extent != reference_extent:
            raise ValueError(
                f"IMU visualization lab 3D extent changes across methods for {take_id}"
            )

        pair = (take_id, method_id)
        if pair in take_method_pairs:
            raise ValueError(f"Duplicate IMU visualization lab take/method pair: {pair}")
        take_method_pairs.add(pair)
        orders.append(order)
        video_ids.append(video_id)
        take_ids.add(take_id)
        method_ids.add(method_id)
        if filename.name in video_names:
            raise ValueError(f"Duplicate IMU visualization lab filename: {filename.name}")
        video_names.add(filename.name)
        for relative in (
            video_relative,
            poster_relative.as_posix(),
            metrics_relative.as_posix(),
            motion_relative.as_posix(),
        ):
            if relative in asset_paths:
                raise ValueError(f"Duplicate IMU visualization lab asset path: {relative}")
            asset_paths.add(relative)
        previous_frame_count = frame_counts_by_take.setdefault(take_id, frame_count)
        if previous_frame_count != frame_count:
            raise ValueError(
                f"IMU visualization lab methods disagree on frame count for {take_id}"
            )

    expected_pairs = {
        (take, method)
        for take in IMU_VISUAL_LAB_TAKE_IDS
        for method in IMU_VISUAL_LAB_METHOD_IDS
    }
    if (
        orders != list(range(1, IMU_VISUAL_LAB_VIDEO_COUNT + 1))
        or len(set(video_ids)) != IMU_VISUAL_LAB_VIDEO_COUNT
        or take_ids != set(IMU_VISUAL_LAB_TAKE_IDS)
        or method_ids != set(IMU_VISUAL_LAB_METHOD_IDS)
        or take_method_pairs != expected_pairs
    ):
        raise ValueError(
            "IMU visualization lab must contain one unique video for every "
            "4-take x 7-method pair in canonical order"
        )

    actual_video_names = {
        path.name
        for path in (source / "videos").iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix.lower() == ".mp4"
    }
    if actual_video_names != video_names:
        raise ValueError("IMU visualization lab videos directory disagrees with manifest")

    validation = _load_json_object(
        files["validation.json"], label="IMU visualization lab validation"
    )
    expected_validation = {
        "schema": IMU_VISUAL_LAB_VALIDATION_SCHEMA,
        "status": "pass",
        "video_count": IMU_VISUAL_LAB_VIDEO_COUNT,
        "full_decode": True,
        "manifest_sha256": sha256(files["manifest.json"]),
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
            for item in videos
        ],
        "failures": [],
    }
    if validation != expected_validation:
        raise ValueError(
            "IMU visualization lab validation receipt is stale or does not bind "
            "the current manifest/video set"
        )

    expected_file_paths = required_files | asset_paths
    if set(files) != expected_file_paths:
        missing_files = sorted(expected_file_paths - set(files))
        extra_files = sorted(set(files) - expected_file_paths)
        raise ValueError(
            "IMU visualization lab file inventory mismatch: "
            f"missing={missing_files}, extra={extra_files}"
        )

    checksum_entries: dict[str, str] = {}
    for line_number, line in enumerate(
        files["SHA256SUMS.txt"].read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(
                f"Malformed IMU visualization lab checksum line {line_number}"
            )
        digest, value = match.groups()
        relative = _safe_delivery_relative_path(
            value, field=f"imu_visual_lab.SHA256SUMS.txt:{line_number}"
        ).as_posix()
        if relative in checksum_entries:
            raise ValueError(
                f"Duplicate IMU visualization lab checksum entry: {relative}"
            )
        checksum_entries[relative] = digest
    expected_checksum_paths = set(files) - {"SHA256SUMS.txt"}
    if set(checksum_entries) != expected_checksum_paths:
        raise ValueError("IMU visualization lab checksum inventory mismatch")
    for relative, expected_hash in checksum_entries.items():
        if sha256(files[relative]) != expected_hash:
            raise ValueError(f"IMU visualization lab checksum mismatch: {relative}")

    return [
        {
            "path": f"downloads/imu-visual-lab/{relative}",
            "source": f"imu_mocap_visualization_lab/{relative}",
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for relative, path in sorted(files.items())
    ]


def mirror_imu_visual_lab(source: Path, destination: Path) -> list[dict]:
    """Atomically publish the independently validated visualization lab."""

    source = Path(source).expanduser()
    destination_input = Path(destination).expanduser()
    if destination_input.is_symlink():
        raise ValueError(f"Unsafe IMU visualization lab destination: {destination_input}")
    destination = destination_input.resolve()
    source_entries = validate_imu_visual_lab_for_web(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Unsafe IMU visualization lab destination: {destination}")

    staging_root = Path(
        tempfile.mkdtemp(prefix=".imu-visual-lab-publish-", dir=destination.parent)
    )
    staged = staging_root / "imu-visual-lab"
    backup = staging_root / "previous-imu-visual-lab"
    moved_previous = False
    try:
        shutil.copytree(source, staged, symlinks=False)
        staged_entries = validate_imu_visual_lab_for_web(staged)
        if source_entries != staged_entries:
            raise ValueError("Staged IMU visualization lab differs from validated source")
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

    Entries under the generated delivery prefixes are intentionally skipped
    here: those ignored mirrors are rebuilt and independently validated from
    their tracked source bundles.  Every other regular file must have a
    one-to-one manifest entry, exact byte count, and exact SHA-256.
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
        if any(
            relative.startswith(prefix)
            for prefix in GENERATED_DELIVERY_WEB_PREFIXES
        ):
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
        if relative == ASSET_MANIFEST_RELATIVE or any(
            relative.startswith(prefix)
            for prefix in GENERATED_DELIVERY_WEB_PREFIXES
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
    imu_comparison_source: Path = IMU_COMPARISON_SOURCE,
    imu_visual_lab_source: Path = IMU_VISUAL_LAB_SOURCE,
) -> Path:
    """Assemble a clean-clone deploy tree from tracked, prebuilt artifacts."""

    public_root = Path(public_root).expanduser().resolve()
    payload, tracked_entries = validate_tracked_public_assets(public_root)
    final_destination = public_root / "downloads/final-nine"
    final_entries = mirror_final_delivery(final_source, final_destination)
    comparison_destination = public_root / "downloads/imu-mocap"
    comparison_entries = mirror_imu_comparison(
        imu_comparison_source,
        comparison_destination,
    )
    visual_lab_destination = public_root / "downloads/imu-visual-lab"
    visual_lab_entries = mirror_imu_visual_lab(
        imu_visual_lab_source,
        visual_lab_destination,
    )

    combined_entries = (
        tracked_entries + final_entries + comparison_entries + visual_lab_entries
    )
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
    comparison_manifest = json.loads(
        (Path(imu_comparison_source) / "manifest.json").read_text(encoding="utf-8")
    )
    visual_lab_manifest = json.loads(
        (Path(imu_visual_lab_source) / "manifest.json").read_text(encoding="utf-8")
    )
    assembled_manifest = dict(payload)
    assembled_manifest["cloudflareMaxAssetBytes"] = MAX_STATIC_ASSET_BYTES
    assembled_manifest["deploymentAssembly"] = {
        "mode": "tracked_public_plus_three_tracked_deliveries",
        "finalDeliverySchema": FINAL_DELIVERY_SCHEMA,
        "finalDeliveryGeneratedAt": delivery_manifest.get("generated_at_utc"),
        "imuComparisonSchema": IMU_COMPARISON_SCHEMA,
        "imuComparisonGeneratedAt": comparison_manifest.get("generated_at_utc"),
        "imuVisualLabSchema": IMU_VISUAL_LAB_SCHEMA,
        "imuVisualLabGeneratedAt": visual_lab_manifest.get("generated_at_utc"),
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

    # The downloadable deliveries are intentionally ignored under web/public
    # so Git stores only one reviewed copy of each.  Recreate those mirrors on
    # every build after the authored/source assets pass their own checks and
    # before emitting the aggregate Cloudflare asset manifest.
    manifest.extend(
        mirror_final_delivery(FINAL_DELIVERY_SOURCE, FINAL_DELIVERY_PUBLIC)
    )
    manifest.extend(
        mirror_imu_comparison(IMU_COMPARISON_SOURCE, IMU_COMPARISON_PUBLIC)
    )
    manifest.extend(
        mirror_imu_visual_lab(IMU_VISUAL_LAB_SOURCE, IMU_VISUAL_LAB_PUBLIC)
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
