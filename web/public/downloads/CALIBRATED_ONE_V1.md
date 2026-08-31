# Calibrated One v1：定义、证据与升级路径

> 该文档保留“glove wrist orientation + MOCAP wrist translation”诊断基线。当前推荐的 calibrated fusion 已升级为 [MOCAP_ROOT_FUSION_V1.md](MOCAP_ROOT_FUSION_V1.md)：它使用完整 MOCAP wrist SE(3) 和 glove local articulation，并单独声明条件化边界。

> 2026-08-31 时间语义修正：下述数字已逐行移除约 254–255 ms 的 camera capture→host poll 延迟。旧 poll-time 数字已 superseded；动作匹配主协议见 [MOCAP_VIDEO_ALIGNMENT_V1.md](MOCAP_VIDEO_ALIGNMENT_V1.md)。

- 状态：`verified_exposure_corrected_diagnostic_baseline`
- 标定源：`01_210814_Take_000`
- 留出评估：`02_210955_Take_001`、`03_211139_Take_002`
- 当前 profile：`outputs/holdout_01_strict25/01_210814_Take_000_registration_profile.json`
- 关键边界：当前不是 absolute wrist / full 6DoF GT。

## 1. 当前“calibrated one”具体是什么

对每只手 (h\in\{L,R\})，当前输出为：

\[
\hat p^{M}_{h,j}(t)
= p^{M}_{h,wrist}(t)
+ s_h R_h p^{G}_{h,j}(t)
\]

其中：

- (p^{G}_{h,j}(t)) 是 glove solver 输出的 root-relative world keypoint；glove wrist 恒为零。
- (p^{M}_{h,wrist}(t)) 是同步后的 MOCAP wrist，每帧复制到绿色骨架。
- (R_h\in SO(3)) 和统一尺度 (s_h>0) 只用 Take 01 的四个非拇指 MCP 拟合。
- 拇指和 fingertip 不参与拟合；02/03 不参与任何 rotation 或 scale 重估。
- 当前实现不做 per-frame rotation refit，所以腕部方向差异仍会保留在视频和指标中。

这一定义得到的是“以 MOCAP wrist 为共同原点的已配准手形/姿态”，而不是拥有独立全局位置的 glove 6DoF。

## 2. 已冻结的参数

| Side | scale | fit frames | Take 01 MCP anchor residual median / P95 |
|---|---:|---:|---:|
| Left | 0.8865779 | 1209 | 16.90 / 47.96 mm |
| Right | 0.8986121 | 1125 | 16.39 / 39.71 mm |

完整 3×3 rotation、输入文件 SHA256、solver/config hash、session-neutral provenance 和相机外参 SHA256 均保存在 registration profile 中，不在文档里手工复制。

## 3. 时间与相机标定

时间链路：

```text
RGB device timestamp
  -> capture_to_poll = thor_realtime - RGB device timestamp
  -> exposure-corrected Thor monotonic / CMAvatar clock
  -> PtpTimeStamp cadence affine-anchored to Windows TimeStamp
  -> MOCAP 与 glove 线性插值
  -> 拒绝 bracket span > 25 ms 的帧
```

三个 Take 的 capture→poll median 为 254.008–255.250 ms。曝光目标到最近 120 Hz MOCAP 样本的 median 约 2.1 ms、P95 约 4.0 ms；这是采样量化距离，不是端到端同步精度。25 ms 是包围目标时间的两个源样本之间的跨度。动作相位 QA 的残余峰值 +10–17 ms 只用于验收，不回写到 profile。

空间链路：

```text
p_color = R_color_from_mocap_world * p_mocap_world + t_color_from_mocap_world
```

当前相机外参来自 CS-400 Marker 人工识别与 109 帧原始深度中值。它用于视频投影；当前 root-normalized 3D 指标本身不依赖相机投影。该外参尚未经过完整 21 关节独立 2D 标注的 held-out 验证。

## 4. 当前交付物

| 内容 | 路径 | 解释 |
|---|---|---|
| Take 01 calibration-fit 视频 | `outputs/holdout_01_review/01_210814_Take_000_glove_vs_mocap_calibration_fit_h264.mp4` | 标定段 self-fit，可看当前 calibrated v1 |
| Take 02 holdout 视频 | `outputs/holdout_01_review/02_210955_Take_001_glove_vs_mocap_holdout_from_Take_000_h264.mp4` | 冻结 profile 的 validation |
| Take 03 holdout 视频 | `outputs/holdout_01_review/03_211139_Take_002_glove_vs_mocap_holdout_from_Take_000_h264.mp4` | 冻结 profile 的 test/stress |
| Profile | `outputs/holdout_01_strict25/01_210814_Take_000_registration_profile.json` | 可重放、带 provenance |
| 严格指标 | `outputs/holdout_01_strict25/*.metrics.json` | 25 ms、root-normalized |

严格 25 ms 结果：

| Take | 角色 | 左 joint / tip median | 右 joint / tip median |
|---|---|---:|---:|
| 01 | calibration/self-fit | 32.58 / 54.26 mm | 30.69 / 51.50 mm |
| 02 | validation holdout | 31.94 / 52.00 mm | 40.08 / 62.33 mm |
| 03 | test/stress holdout | 43.47 / 66.30 mm | 66.25 / 101.86 mm |

## 5. 为什么 v1 还不够好

1. Take 01 的 self-fit 仍有约 30 mm joint median 和约 51–53 mm fingertip median；一个全手统一尺度和固定 rotation 不能解释所有指骨长度、关节零位与非线性响应。
2. 三个 Take 使用相同 solver/config hash，但各自重新采集 session-neutral。跨 Take 退化同时包含 neutral/recenter 与佩戴重复性，不能只归因于 finger solver。
3. 现有 glove keypoint 只有 host monotonic timestamp；缺少传感器采样时刻，无法把动作相关误差可靠分解为时间延迟与姿态误差。
4. `Human.cma` 没有逐帧 confidence、occlusion、gap-fill 或 residual，当前无法剔除 MOCAP 求解器补点/低质量帧。
5. glove 没有独立 root translation；没有物理 wrist rigid body 到 glove root 的固定变换时，absolute wrist 不可辨识。

## 6. v2 可以安全尝试什么

以下参数必须只在独立 calibration split 上拟合，并冻结后在内部时间留出 + Take 02/03 上评价：

1. **硬件时间验证**：下一次采集使用共享 TTL；若接口不允许则使用 RGB 可见且 MOCAP 可追踪的同步 LED。动作相关性只做 QA，禁止逐 Take 搜索并回写视觉最优 \(\Delta t\)。
2. **稳健 palm orientation / mount offset**：从 wrist 与四个 MCP 构造逐帧 palm frame，在 SO(3) 上估固定偏置并做 outlier rejection；禁止 per-frame 对齐。
3. **正则化的骨长参数**：优先从 pose-invariant bone length 中估每指或每骨段尺度，设置合理范围并保存不确定度。fingertip 继续作为 holdout，先比较它是否真的改善。
4. **有限的 joint zero/gain**：只有在 dedicated calibration motions、原始 Marker quality 和足够 excitation 到位后再拟合；参数量必须受控，并用整段/动作类别留出防止过拟合。
5. **冻结 solver 输入**：保存并可重放 Take 01 的 `session_neutral.json`、skeleton、calibration、solver config 和原始 glove 解码表。当前只有 hash/path provenance，不足以复算同一个输出。

## 7. 不能用来“变好看”的做法

- 不在 Take 02/03 上重新拟合 rotation、scale、time offset 或 finger parameters 后再把它们称为 holdout。
- 不逐帧拟合刚体变换或直接复制 MOCAP 手指姿态。
- 不把 fingertip 放进拟合后再把 fingertip 误差称为独立验证。
- 不把同视频投影贴合、同段自拟合或共享 wrist 的误差称为绝对精度。
- 不在没有 Marker quality 的情况下静默删掉“看起来不对”的帧。

## 8. Production calibrated profile 的 Definition of Done

一个可以交付为 `calibration_profile.v2` 的结果至少需要：

- 明确的 calibration / internal holdout / cross-session test split；
- 所有输入、代码、solver/config、neutral 与输出的 SHA256；
- 参数、单位、坐标系、四元数顺序、active/passive 约定和估计不确定度；
- excitation/condition-number 质量门限，退化数据必须拒绝出 profile；
- 整体、逐手、逐指、静态/动态、慢/快动作和重戴后的指标；
- Marker/joint quality gating 前后的覆盖率与误差；
- 物理 rigid-body root link 到位后，才增加 absolute wrist translation / orientation 指标；
- 任何目标 Take 都不能参与其自身最终指标所使用的参数拟合。

## 9. 当前复现命令

```bash
cd /home/runyi/Project/hands_reloc/GT_calib

../hands_reloc/.venv/bin/python gt_calib_viz.py \
  --registration-segment 01 \
  --segment 01 --segment 02 --segment 03 \
  --max-interpolation-gap-ms 25 \
  --analyze-only \
  --output-dir outputs/holdout_01_strict25
```

先把 v1 作为可审计 baseline；v2 的每个新增自由度都必须用真正未参与拟合的帧、动作和 Take 证明它提升了泛化，而不只是让 calibration 视频更贴。
