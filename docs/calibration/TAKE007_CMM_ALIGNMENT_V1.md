# Take_007 CMM → RGB / solved-pose alignment v1

## 目的与范围

本协议只定义 `camera_glove_recording_20260831_161912 / Take_007` 的最终
可视化链：把 CMM reflector XYZ 按时间投影到 RGB，并用已确认的十个
surface-marker 对应放置每手 20 点 solved pose。它不把 CMM 表面点升级成
anatomical joint GT，也不把同段拟合 residual 解释成独立 accuracy。

## 输入合同

| 输入 | 合同 |
|---|---|
| RGB | 1920×1080、30 FPS、1982 source frames |
| Camera timecodes | 980 个稀疏 device-time anchors |
| CMM | 120 Hz、8552 个连续 `FrameCounter`、marker XYZ 单位 mm |
| Solved pose | 每手 20 个 root-local keypoint，输入单位 m、内部转 mm |
| Camera/world | 2026-08-31 CS-400 world→color + 161912 自己的 RGB intrinsics |

正式输出只覆盖 source RGB 0..1980；frame 1981 没有 reconstructed camera/CMM
anchor，必须排除。逐帧证据写入
`frame_maps/take007_rgb_to_cmm.csv`。

## Marker 合同

照片 #1..#10 依次是 thumb tip/base、index tip/base、middle tip/base、ring
tip/base、pinky tip/base。每手十点全部显示并参与 solved-pose 的 same-take
orientation/scale fit。#11 是 dorsum black module reflector，只构造 root，禁止
作为第 21 joint 绘制。

Logical CMM tracks：

```text
left  = [(11781 -> 12503), 11978, 12003, 12001, 12006,
         12013, 12045, 12015, 12076, 12051, 12064]
right = [12436, 12435, 11619, 11614, 11616,
         12434, 11625, 11685, 11627, 11630, 12433]
```

左 #1 的 re-ID 只跨两个缺失 CMM 帧。允许的 gap fill 上限为四个内部 CMM
帧；首尾缺失或更长缺口禁止外推。

照片 marker 到 solved index：

```text
#1..#10 -> [3, 1, 7, 4, 11, 8, 15, 12, 19, 16]
```

## 空间算法

Virtual wrist 使用 #11 与四个非拇指 base：

```text
c = mean(p4, p6, p8, p10)
d = (p11 - c) / ||p11 - c||
r = p11 + 20 mm * d
```

这是逐帧 hand-local 后移。禁止把它实现为固定 `world +Z` 或任意固定世界轴
偏移。

对每侧，在所有联合有效帧上由十个对应估计 uniform scale，冻结为 median；
每帧用十点 Kabsch 求 proper rotation。任意 solved point `g_k` 的世界位置为：

```text
q_k = r + delta_global + delta_video + delta_side + s * R * (g_k - g_wrist)
```

当前 final-nine 审核值为 `delta_global=[0,-44,0] mm`，Take_007 的
`delta_video=delta_left=delta_right=[0,0,0] mm`。它是 operator display
correction，必须写入 metrics/header；不能静默烘焙，也不能解释为独立 GT。

当前网页导出 `gt_calib.final_nine_manual_xyz.v1`。其 global 数值会分别在每段
自己的 MOCAP world 中执行；只有 Take_007 video 07/08 可以使用左右手 residual，
video 09 强制 excluded。完整合同见
[FINAL_NINE_MANUAL_XYZ_V1.md](FINAL_NINE_MANUAL_XYZ_V1.md)。

旧 `gt_calib.manual_xyz_profile.v1` loader 继续兼容，但严格只作用于 Take_007，
并要求：

```text
source_recording = camera_glove_recording_20260831_161912
source_take      = Take_007
coordinate_system = mocap_world_mm
units            = mm
rear_offset_mm   = 20
```

三个 XYZ 字段都必须是三个有限数值。`rear_offset_mm` 不是人工自由度；任何非
20 mm 值都会拒绝，以避免 manual translation 偷换已确认的 hand-local root。
Legacy profile 不能移动 video 01-06；要对完整九视频应用 operator correction，
必须使用 `gt_calib.final_nine_manual_xyz.v1`。

## 当前结果

| 项目 | Left | Right |
|---|---:|---:|
| frozen scale | 0.831746 | 0.859253 |
| all-10 residual median | 21.02 mm | 22.11 mm |
| all-10 residual P95 | 52.37 mm | 54.09 mm |
| all-10 residual max | 85.58 mm | 99.16 mm |

联合 solved validity 为 1367/1981（69.01%）。Camera→CMM anchor absolute error
median/P95/max 为 2.238/3.988/4.577 ms；980/980 小于 20 ms。无效 solved 帧
必须显示 unavailable，不得 hold-last 或外推。

Residual 大于用户观察到的手套间约 1 cm 间距并不矛盾：十点 fit 还把表面
reflector 对到了 anatomical solver 点，且手型/尺度/关节定义不同。该表仅是
same-take model consistency diagnostic。

## 产物与 schema

| 产物 | Schema / shape |
|---|---|
| `videos/07_take007_labeled_mocap_markers.mp4` | `gt_calib.take007_labeled_cmm_markers.v2` |
| `videos/08_take007_aligned_hand_pose.mp4` | `gt_calib.take007_cmm_aligned_glove_pose.v2` |
| `calibration-workbench/take007_mocap.f32` | float32-le `[1981,2,11,3]` |
| `calibration-workbench/take007_solved.f32` | float32-le `[1981,2,20,3]` |
| `calibration-workbench/take007_alignment.json` | `gt_calib.manual_xyz_workbench.v1` |
| browser export | `gt_calib.final_nine_manual_xyz.v1`；legacy loader 仍接受 `gt_calib.manual_xyz_profile.v1` |
| `calibration-workbench/applied_manual_profile.json` | 仅在传入 profile 时复制原始 JSON；manifest 记录 SHA-256 |

Workbench 的 CMM node 0 是 synthetic wrist，node 1..10 才是实测 #1..#10。
所有数组顺序均为 `[frame, left/right, node, xyz]`，坐标单位 mm。

当前正式重建应使用
`calibration_profiles/operator_y_minus_44_all_hand_overlays.v1.json`，对 video
01-08 各自在本段 MOCAP world 中应用 `[0,-44,0] mm`，对 video 09 不应用。
输入 JSON 必须原样复制到上述路径，manifest 必须写入其 SHA-256，validation
必须复算并 fail closed；仅在 metrics 或浏览器 localStorage 中记录数值而不保存
输入文件不满足 provenance 合同。

## 最终包验收

新的 BVH-FK + `[0,-44,0] mm` package 已从空 destination 正式重建：41 个
regular files、122,158,302 bytes（116.50 MiB），`SHA256SUMS.txt` 40 行；9/9
H.264 视频通过 full decode，validation `status=pass`、`failures=[]`。应用 profile
SHA-256 为 `2eaf48440999a1efe8a27bb259a98e9305da5132f4783605ede7601f7e3c78d8`。
Take_007 CMM source SHA-256 为
`25bebdbb079d233b7c71ba94402531a9ed09b87ac690def05b012065b5748a08`，并写入
video 07/08 metrics 与 workbench metadata。
当前全量代码回归为 `155 passed in 107.14s`。

## Video 09 不变量

`videos/09_no_glove_world_calibration.mp4` 必须绑定无手套 recording `155410`：

- RGB 来自 155410，98 帧；
- intrinsics 来自 155410 自己的 `camera_1_intrinsics.json`；
- reference frame 为 155410 frame 30；
- world transform 来自 2026-08-31 CS-400 calibration archive。

不得以旧 `00`、Take_006 或 Take_007 的画面/intrinsics 代替。
也不得对 video 09 应用 final-nine global/per-video XYZ；它没有手部 MOCAP
overlay，profile 必须记录 `apply_translation=false` 且全部 residual 为零。

Builder 还执行内容身份 hard gate：解码归档 member
`.../20260831_processed_tabletop_origin/reference_rgb.png` 与 155410 RGB frame 30，
要求 BGR 数组 shape 和每个像素完全一致，否则立即失败。当前状态为
`pixel_identical`；archive PNG SHA-256 为
`e4a31ff908a01af7474b7705077c19d58bf435d07e9eaece2f1be50b5ae83388`，双方
decoded-BGR SHA-256 均为
`cabc821fb4397117c722e1c1b2e2dad7a05e9a63590d35c49fd6b736a4d1bf9e`。
Video 09 metrics 必须保留 `calibration_reference_identity`，使“匹配 155410”
成为可机检的内容身份合同，而不只是路径字符串。

## 复现与人工修正

```bash
uv run gt-calib-delivery render-new \
  --destination outputs/new_capture_review

uv run gt-calib-delivery serve --host 127.0.0.1 --port 8811

uv run gt-calib-delivery render-new \
  --destination outputs/take007_manual_review \
  --manual-profile calibration_profiles/operator_y_minus_44_all_hand_overlays.v1.json

uv run gt-calib-delivery build \
  --destination rebuilt_9_video_delivery \
  --manual-profile calibration_profiles/operator_y_minus_44_all_hand_overlays.v1.json

uv run gt-calib-delivery validate \
  --destination rebuilt_9_video_delivery --full-decode
```

网页地址为 `http://127.0.0.1:8811/final-nine/#manual-calibration`。调整结果是
operator-selected display correction；网页可选择九段并编辑逐段 XYZ，但只有
Take_007 video 07/08 支持左右手 residual/live reprojection，video 09 固定排除。
该修正不是 cross-session shared extrinsic；只有在独立 held-out 数据上冻结并
验证后，才可讨论外部 accuracy。
