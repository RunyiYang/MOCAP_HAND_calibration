# 跨 Take MOCAP/Glove 评估协议

## 目的

减少“在待评估 take 上使用 MOCAP 再拟合”的 GT 泄漏。当前协议只评估 MOCAP-wrist 条件下的 root-normalized 手形与姿态一致性，不评估绝对腕部平移。

## Split

| Take | 角色 | 是否用于固定 R+scale |
|---|---|---:|
| 01 / Take_000 | calibration | yes |
| 02 / Take_001 | validation holdout | no |
| 03 / Take_002 | test/stress holdout | no |

## 计算步骤

1. 从每行相机同步记录中减去 `capture_to_poll = thor_realtime - color_device_timestamp`，恢复曝光时刻的 Thor/CMAvatar target clocks。
2. 使用 `Human.cma.PtpTimeStamp` 的高分辨率相对 cadence，并由 Windows `TimeStamp` affine anchor；不使用视觉动作拟合 per-take offset。
3. 使用 01 的四个非拇指 MCP，分别为左右手拟合一个固定 canonical rotation 和统一尺度。
4. 将 registration profile 写入 JSON，并记录 Human CMA、glove keypoints、solver/config/session-neutral provenance 与 SHA256。
5. 冻结 profile 后应用到 02/03，不使用目标 take 重估 rotation、scale 或时间偏移。
6. Root-fusion 每帧使用曝光时刻 MOCAP wrist translation + orientation；所有定量结果显式为 MOCAP wrist SE(3)-conditioned root-normalized articulation。
7. 拒绝跨越超过 25 ms 源样本 bracket 的插值。
8. 拇指只绘制，不计入指标；四个 fingertip 不参与配准。

## 主指标

- `root_normalized_non_thumb_joint_epe_mm`：frame×16 joints pooled EPE。
- `root_normalized_frame_mpjpe_mm`：每帧 16 个非拇指关节的平均 EPE。
- `root_normalized_non_thumb_fingertip_epe_mm`：四个非拇指 fingertip pooled EPE。
- 覆盖率：质量/插值门限后保留帧数 ÷ 同步 MOCAP RGB 帧数。

## 复现

```bash
../hands_reloc/.venv/bin/python gt_calib_viz.py \
  --registration-segment 01 \
  --segment 01 --segment 02 --segment 03 \
  --max-interpolation-gap-ms 25 \
  --analyze-only \
  --output-dir outputs/holdout_01_strict25
```

复用已冻结的 profile：

```bash
../hands_reloc/.venv/bin/python gt_calib_viz.py \
  --registration-profile outputs/holdout_01_strict25/01_210814_Take_000_registration_profile.json \
  --segment 02 --segment 03 \
  --max-interpolation-gap-ms 25 \
  --analyze-only \
  --output-dir outputs/holdout_01_replay
```

## 解释边界

- 三段各自重新采集了 session-neutral；跨 take 结果包含 neutral/recenter 和重新佩戴/安装重复性影响，不能只归因于手指解算。
- MOCAP wrist 平移是共享条件，absolute wrist/6DoF 必须保持 `null/unavailable`。
- Root-fusion 的 MOCAP wrist orientation 也是共享条件，不能把结果解释为 glove wrist orientation 精度。
- 到最近 120 Hz MOCAP sample 的距离只是采样量化距离，不是同步 accuracy；没有共享 TTL 时不能宣称零时间误差。
- `Human.cma` 缺少逐帧质量字段；当前无法剔除动捕模型补点或低置信帧。
- 相机外参尚未经过完整 21 关节的独立像素真值验证。
