# MOCAP Root Fusion v1：当前推荐的 calibrated one

- 状态：`verified_conditional_fusion`
- 标定源：`01_210814_Take_000`
- 留出评估：`02_210955_Take_001`、`03_211139_Take_002`
- Pose mode：`mocap-root-fusion`
- Claim：`mocap-wrist-6DoF-conditioned glove articulation`
- 禁止的 Claim：independent glove wrist accuracy、absolute glove 6DoF、最终 GT。

> 2026-08-31 时间语义修正：所有下述结果均逐行移除了约 254–255 ms 的 camera capture→host poll 延迟，并使用 `PtpTimeStamp` 高分辨率节拍。旧 poll-time 数字已 superseded；详见 [MOCAP_VIDEO_ALIGNMENT_V1.md](MOCAP_VIDEO_ALIGNMENT_V1.md)。

> 同日补充 source identity gate：正式发布用 `--frame-content-samples 0` 对三段全部 BAG/MP4 frame index 完成 240×135 解码灰度+dHash 同帧核验；所有可辨帧 best-offset=0，global best offset 均为 0。默认 31 帧只作快速回归。该证据不是原始编码 bit-exact，不验证空间精度或硬件零时差，也不把下述 fusion 指标升级为独立空间 GT。

## 1. 结果

当前数据中更合理的融合方式是让 MOCAP 提供每帧 wrist SE(3)，让 glove 只提供 root-local finger articulation：

\[
\hat p^M_{h,j}(t)=p^M_{h,wrist}(t)+R^M_{h,wrist}(t)
\left[s_h C_h p^G_{h,j,local}(t)\right]
\]

其中 \(C_h\in SO(3)\) 和 \(s_h>0\) 只使用 Take 01 的四个非拇指 MCP 在 wrist-local frame 中求解，之后冻结到 Take 02/03。目标 Take 的 finger joints 不参与参数拟合；但目标 Take 的 MOCAP wrist translation 和 orientation 每帧都会进入融合。

严格 25 ms 门限下，non-thumb joint / fingertip pooled EPE：

| Take | Role | Left median | Right median | Left P95 | Right P95 |
|---|---|---:|---:|---:|---:|
| 01 | calibration/self-fit | 13.44 / 22.29 mm | 11.94 / 19.42 mm | 83.53 / 102.43 mm | 85.83 / 104.74 mm |
| 02 | validation holdout | 11.49 / 18.05 mm | 11.39 / 18.88 mm | 86.06 / 104.87 mm | 88.08 / 107.28 mm |
| 03 | test/stress holdout | 12.94 / 23.27 mm | 13.19 / 26.38 mm | 95.44 / 113.80 mm | 97.44 / 110.23 mm |

中位数在三个 Take 上明显稳定；P95 仍高，说明部分姿态/动作中的 finger articulation 仍会严重发散。页面和报告必须同时呈现 median 与长尾边界。

Take02/03 的 finger labels 没有进入参数拟合，但本轮已经查看这两段结果并据此选择/审查模型，因此它们是 retrospective cross-take holdout，不再是全盲 final test。任何后续 articulation 模型晋级都需要锁定后新采 untouched Take04。

## 2. 相对旧模式的变化

旧模式把 glove solver 已经旋到 world 的 keypoints 再乘一个固定 world rotation，只复制 MOCAP wrist translation。它没有使用 `Human.cma` 已有的 wrist `GlobalQ`，固定变换也不能正确表达 local mount correction。

新模式改为：

```text
Human.cma wrist GlobalQ_R/X/Y/Z
  -> wxyz normalize + sign-continuous SLERP
  -> active wrist-local to MOCAP-world rotation

glove local keypoints
  -> Take01 canonical C + scale
  -> target-frame MOCAP wrist rotation + translation
  -> camera projection
```

median 的前后对比：

| Take | Side | 旧 world-wrist comparison joint/tip | 新 root-fusion joint/tip |
|---|---|---:|---:|
| 01 | Left | 32.58 / 54.26 | 13.44 / 22.29 mm |
| 01 | Right | 30.69 / 51.50 | 11.94 / 19.42 mm |
| 02 | Left | 31.94 / 52.00 | 11.49 / 18.05 mm |
| 02 | Right | 40.08 / 62.33 | 11.39 / 18.88 mm |
| 03 | Left | 43.47 / 66.30 | 12.94 / 23.27 mm |
| 03 | Right | 66.25 / 101.86 | 13.19 / 26.38 mm |

这个改善主要说明：原先的大量误差来自 wrist orientation/session-neutral 与错误的坐标组合。它不证明 glove wrist 被标定准确，因为新模式直接采用了 MOCAP wrist orientation。

## 3. Human root 与四元数契约

当前 Human CMA 的有效 root：

| Side | Entity | Quaternion fields |
|---|---|---|
| Left | `Skeleton_0_LeftHand` | `GlobalQ_R/X/Y/Z` |
| Right | `Skeleton_1_RightHand` | `GlobalQ_R/X/Y/Z` |

磁盘列顺序虽然是 X/Y/Z/R，loader 按字段名重排为 wxyz。几何验证表明它是 active wrist-local → MOCAP-world：用正确方向变换后，root-local MCP 跨帧 dispersion P50/P95 约 0.00065/0.00107 mm；使用转置错误方向则为 12.6–19.6/63.0–114.3 mm。

正式供应商 axis/quaternion contract 仍未交付，因此 profile 写为“empirically verified”，不能替代 P0 数据契约。

## 4. Profile

文件：

```text
outputs/mocap_root_fusion_strict25/
  01_210814_Take_000_mocap_root_fusion_registration_profile.json
```

Take 01 参数：

| Side | canonical scale | fit frames | MCP anchor residual median / P95 |
|---|---:|---:|---:|
| Left | 0.9276215 | 1275 | 7.47 / 8.54 mm |
| Right | 0.9244824 | 1275 | 7.70 / 9.78 mm |

完整 canonical rotation、输入文件 SHA256、solver/config hash、session-neutral provenance、quaternion 语义和相机外参 SHA256 保存在 profile。scale 映射的是两个 solver template，不应解释成真人解剖骨长比例。

Profile 的主旋转字段是 `rotation_mocap_root_local_from_glove_local`；同时保留旧键 `rotation_mocap_from_glove` 作为兼容 alias，loader 会在两者同时存在但数值不一致时拒绝载入。Profile 和 metrics 还会显式写入 `target_finger_joints_used_in_fit: false` 与 `quaternion_semantics: empirically verified active root-local-to-mocap-world`。

Profile schema 为 `gt_calib.mocap_root_conditioned_articulation_profile.v2`、`artifact_type=registration_profile`；metrics schema 为 `gt_calib.mocap_root_conditioned_articulation_metrics.v2`、`artifact_type=evaluation_metrics`。v2 profile 额外固定 camera acquisition、capture→poll removal 与 PTP cadence 语义。旧 root profile schema 仍可读取，但新 loader 会对 pose mode、joint contract、root conditioning、quaternion semantics、左右手和 rotation alias 做 fail-closed 校验。

## 5. 正式产物

| 内容 | 路径 |
|---|---|
| 严格 profile/metrics | `outputs/mocap_root_fusion_strict25/` |
| 三段 review 视频/metrics/contact | `outputs/mocap_root_fusion_review/` |
| Take 01 H.264 | `01_..._mocap_root_fused_glove_calibration_fit_h264.mp4` |
| Take 02 H.264 | `02_..._mocap_root_fused_glove_holdout_from_Take_000_h264.mp4` |
| Take 03 H.264 | `03_..._mocap_root_fused_glove_holdout_from_Take_000_h264.mp4` |

三段 H.264 均为 960×540、30 FPS、High/yuv420p、faststart，并已完成全帧解码。

## 6. 复现

严格指标：

```bash
cd /home/runyi/Project/hands_reloc/GT_calib

../hands_reloc/.venv/bin/python gt_calib_viz.py \
  --pose-mode mocap-root-fusion \
  --registration-segment 01 \
  --segment 01 --segment 02 --segment 03 \
  --max-interpolation-gap-ms 25 \
  --analyze-only \
  --output-dir outputs/mocap_root_fusion_strict25
```

Review 视频：

```bash
../hands_reloc/.venv/bin/python gt_calib_viz.py \
  --pose-mode mocap-root-fusion \
  --registration-segment 01 \
  --segment 01 --segment 02 --segment 03 \
  --max-interpolation-gap-ms 25 \
  --output-width 960 \
  --snapshot-count 6 \
  --output-dir outputs/mocap_root_fusion_review
```

## 7. 验证

- 单元/数据契约测试：35/35 passed。
- 包含 wxyz active rotation、sign-continuous SLERP、synthetic root fusion、robust outlier、Take01 profile freeze 和 target finger joint 不进 fit 的检查。
- 三段正式 metrics 与独立只读几何探针一致。
- 三段 H.264 全帧解码通过；contact sheet 显示 wrist/palm 的主要方向错位已消除。

## 8. P95 长尾诊断与 gating 边界

长尾不是插值 gap 或 wrist root 跳变。Take 02 的主要高误差簇位于约 22.2–23.5 s、51.4–52.6 s；Take 03 位于约 11.0–12.6 s、54.5–55.3 s，而且左右手 top-5% frame error 的同帧重叠率分别为 83.6% 和 96.8%。这些都是有效片段内部持续约 0.6–1.6 s 的共同 gesture plateau。

这些片段中，MCP 误差仍约 7.5 mm，但误差沿 PIP、DIP、tip 逐级放大到约 41–109 mm；glove local 手形仍接近伸直，而 MOCAP target fingers 已明显屈曲。因此当前证据指向 finger articulation/solver mismatch，而不是 root SE(3)、时间同步或视频投影。

- 保留的安全硬 gate：common interval、source bracket、finite、`gap <= 25 ms`。
- 不应硬 gate：glove/MOCAP nearest age、root 速度/角速度、glove local speed/freeze；这些量与长尾相关性弱或跨 Take 不稳定。
- `glove mean extension >= 0.996` 可把 Take 03 frame-tip P95 降至约 30–37 mm，但同步覆盖只剩约 33%，本质是只承诺近伸直姿态域的 selective abstention，不是 calibration 变准。
- 任何使用目标 MOCAP finger joints、target extension、MPJPE 或按 02/03 error 选择时间段的 gate 都是评估泄漏。

所以 v1 默认不增加“看起来更漂亮”的长尾过滤；质量 telemetry 可以记录，但一般手势覆盖下仍需要真正的 articulation calibration 或新的 solver 输入。

## 9. 当前仍缺什么

- glove root translation 恒为零，任何静态 calibration 都无法恢复运行时 3D translation；必须持续使用 MOCAP Body、vision 或 depth。
- Human root 与 Body Hand 基本是同一 MOCAP asset 链，不是独立验证；仍缺刚体 Marker 定义和逐帧 residual/validity。
- 三个 Take 使用不同 session-neutral，但 neutral JSON 和 hash 对应的 solver assets 没有交付。
- 当前 camera transform 的线性块不是严格 SO(3)，且没有独立 held-out joint pixel 真值；只足够做 review overlay。
- Human.cma 缺少 confidence、occlusion、gap-fill 与 residual，无法解释 P95 长尾里多少来自 MOCAP 求解质量。
- 每骨段长度比反映两个 solver template；没有受试者测量和 raw Marker 时，不应继续堆高自由度让曲线“看起来更贴”。

因此，MOCAP Root Fusion v1 是当前数据能支持的最好 calibrated visualization/fusion baseline；它明确用 MOCAP root 条件化 glove articulation，不是独立 wrist/6DoF GT。
