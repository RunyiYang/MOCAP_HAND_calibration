# Final-nine operator manual XYZ v1

## 目的与 claim boundary

本协议定义最终九视频的人工 XYZ 显示修正，以及网页导出
`gt_calib.final_nine_manual_xyz.v1` 到离线重建的完整合同。当前审核值为
`global_world_xyz_mm = [0, -44, 0]`：video 01-08 都应用，video 09 强制排除。

这里的 global 只表示“把同一组 operator 数值分别复制到每段自己的 MOCAP
world”。2026-08-29 的 Take01/02/03 与 2026-08-31 的 Take_007 不共享一个由
该 profile 估计出来的物理外参。因此本修正：

- 是可追溯的 display correction；
- 不修改已交付的 camera/world calibration；
- 不是 cross-session shared extrinsic；
- 不是独立 hand-pose GT、camera calibration accuracy 或 6DoF 精度证据。

## 九视频作用域

| # | Video ID | 3D 显示源 | XYZ 行为 |
|---:|---|---|---|
| 01 | `take01-mocap` | Skeleton_0/1 BVH FK，21 joints/hand | global + per-video |
| 02 | `take01-solved` | glove local articulation + synchronized MOCAP wrist SE(3) | global + per-video |
| 03 | `take02-mocap` | Skeleton_0/1 BVH FK，21 joints/hand | global + per-video |
| 04 | `take02-solved` | glove local articulation + synchronized MOCAP wrist SE(3) | global + per-video |
| 05 | `take03-mocap` | Skeleton_0/1 BVH FK，21 joints/hand | global + per-video |
| 06 | `take03-solved` | glove local articulation + synchronized MOCAP wrist SE(3) | global + per-video |
| 07 | `take007-mocap-markers` | CMM #1..#10 + synthetic virtual wrist | global + per-video + left/right residual |
| 08 | `take007-solved` | CMM-conditioned solved 20-joint pose | global + per-video + left/right residual |
| 09 | `no-glove-calibration` | 155410 CS-400 world/camera calibration | excluded；任何 translation 均拒绝 |

旧 final 的 video 01/03/05 曾错误读取 `Human.cma` joint positions。当前正式
builder 必须通过 `--mocap-position-source skeleton-bvh`，分别对
`Skeleton_0.bvh` 与 `Skeleton_1.bvh` 做 FK，得到每手 21 个显示点。
`Human.cma` 对这三段仅保留经过验证的 120 Hz timestamp / `FrameCounter` 轴；
不再提供最终 joint positions。BVH unit 到 MOCAP-world mm 的转换是：

```text
[X, Y, Z]_bvh -> [-10 X, 10 Z, 10 Y]_mocap_world_mm
```

BVH ordinal 0 是 vendor all-zero seed，不能进入 formal final interval。因此最终
旧三组双视频的 RGB/frame contract 为：Take01 source 61..1807、1747 帧；
Take02 source 0..1804、1805 帧；Take03 source 4..1802、1799 帧。Take01/03
不能沿用旧 Human.cma final 的 60/3 起点或 1748/1800 帧数。

## 变换顺序

对可应用视频 `v`，基础修正为：

```text
delta_v = global_world_xyz_mm + per_video_world_xyz_mm[v]
```

只有 Take_007 video 07/08 可以再按手增加：

```text
delta_v,h = delta_v + side_residual_world_xyz_mm[v,h]
```

最终点为 `p_display = p_mocap_world + delta`。平移在 MOCAP 时间插值后、
camera projection 前执行。Video 02/04/06 会先平移同步 MOCAP wrist，再由同一
wrist anchor 移动 solved pose；不会只移动其中一层。Video 07/08 同样保证 CMM
与 anchored solved layer 使用一致基础修正。

Root-normalized articulation residual 对这种共享平移不敏感，因此 residual
不验证 `Y=-44 mm` 的 RGB 对齐效果。视频 header 与 metrics 必须把非零值明确
标成 operator/display-only、not GT。

## `gt_calib.final_nine_manual_xyz.v1` 合同

顶层必需字段：

```json
{
  "schema": "gt_calib.final_nine_manual_xyz.v1",
  "axis_order": ["x", "y", "z"],
  "units": "mm",
  "coordinate_frame": "per_video_mocap_world",
  "global_world_xyz_mm": [0.0, -44.0, 0.0],
  "video_annotations": []
}
```

`video_annotations` 必须恰好九条，并与 manifest 的 canonical order、video ID
和 filename 一致。每条包含 source `video_sha256`、`apply_translation`、
`per_video_world_xyz_mm`、`left_residual_world_xyz_mm`、
`right_residual_world_xyz_mm`。所有 vector 必须恰好三个真实 JSON 数值；numeric
string 与 boolean 均拒绝，`order` 也必须是严格 integer，不能用 `true` 或 `1.0`。

Fail-closed 规则：

- video 01-08 的 `apply_translation` 必须为 `true`；video 09 必须为 `false`；
- video 01-06 与 video 09 的 left/right residual 必须严格为零；
- video 09 的 per-video XYZ 也必须严格为零；
- 轴序、单位、coordinate frame、九条顺序或 SHA-256 形状不匹配即拒绝；
- applied JSON 原样复制为
  `calibration-workbench/applied_manual_profile.json`，manifest 记录其 SHA-256。
- checked-in profile 内的 `source_manifest.sha256` 与每行 `video_sha256` 有意绑定
  应用 correction 之前的 pre-render baseline；video 01-08 重渲染后，当前
  manifest/video SHA 必然不同。这不是 stale profile，profile 文件自身由当前
  manifest 的 applied-profile SHA 固定。
- applied profile 是 immutable render input；`refresh-aux-hashes` 发现 profile 内容
  与 manifest SHA 不一致时必须停止并要求完整 rebuild，不能更新 SHA 来掩盖未重渲染视频。
- workbench clean RGB / MOCAP / solved binary 的 SHA 同时存在 manifest 和
  `take007_alignment.json`；辅助资源刷新必须同步两层，且所有 relative path 必须
  保持在 delivery folder 内，禁止 symlink 与 `..` escape。

旧 `gt_calib.manual_xyz_profile.v1` loader 保留兼容，但它严格绑定
`161912 / Take_007`，只作用于 video 07/08，不能借此静默移动旧 video 01-06。
Take_007 的 `rear_offset_mm=20` 仍是 hand-local virtual-wrist 合同，不是人工
XYZ 自由度。

## 网页操作

本地启动：

```bash
uv run gt-calib-delivery serve --host 127.0.0.1 --port 8811
```

打开 `http://127.0.0.1:8811/final-nine/#manual-calibration`：

1. 下拉框可选择全部九段；网页从 manifest 指向的 applied profile 读取当前
   global，本包为 `[0,-44,0] mm`。
2. 每段都显示其 XYZ 状态；video 01-08 可保存 per-video residual。
3. 只有 video 07/08 开放 left/right residual 和 clean-RGB live reprojection。
4. Video 01-06 的调整写入 JSON，必须离线 rebuild 后才能看到新 MP4。
5. Video 09 只用于审阅，始终显示 excluded，不允许 translation。
6. 导出文件为 `final_nine_manual_xyz_annotations.json`，schema 为
   `gt_calib.final_nine_manual_xyz.v1`。

“恢复”回到 applied profile，不回到源码硬编码常量。浏览器 localStorage 只是
编辑状态，不是交付 provenance；草稿只有在九段有序
`order:id:filename:video_sha256`、workbench metadata SHA 与 applied-profile SHA
全部一致时才加载。任何 package 内容变化都会丢弃旧草稿并回到新 applied
profile。最终仍必须导出 JSON 并用 builder 重建。

## 正式重建与验收

仓库内审核 profile：

```text
calibration_profiles/operator_y_minus_44_all_hand_overlays.v1.json
```

从不存在的 destination 重建并全片验证：

```bash
cd /home/runyi/Project/hands_reloc/GT_calib

uv sync --frozen --group dev
uv run gt-calib-delivery inspect
uv run gt-calib-delivery build \
  --destination rebuilt_9_video_delivery \
  --manual-profile calibration_profiles/operator_y_minus_44_all_hand_overlays.v1.json
uv run gt-calib-delivery validate \
  --destination rebuilt_9_video_delivery --full-decode
```

`render-new` 只重建 Take_007 video 07/08 与 video 09，不能完成旧 video 01-06
的 profile 应用。最终验收必须读取新 destination 的 manifest、metrics、
`applied_manual_profile` SHA-256、checksums 与 9/9 full-decode 结果，不能沿用
零-offset baseline 的 hash、size 或 validation。
