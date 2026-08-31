# Supervised edge articulation probe：研究记录，不可产品化

- 状态：`research_only_not_implemented`
- Parent mode：`mocap-root-fusion`
- Fit source：`01_210814_Take_000`
- Evaluation：Take 02 / Take 03 frozen holdout
- 结论：P95 有改善，但当前模型不可辨识、容量偏高、thumb 不完整且不是 Pareto improvement；不得替换 MOCAP Root Fusion v1。

## 1. 为什么做这个 probe

Root Fusion v1 已经消除了主要 wrist/palm 坐标组合错误，但 holdout pooled tip P95 仍约 105–114 mm。时间段诊断表明，高误差是屈曲 gesture plateau 中的 finger articulation mismatch，而不是同步 gap、wrist GlobalQ 跳变或片段边界。因此只读 probe 测试了“用 Take01 MOCAP fingers 监督标定 glove-local 骨边”的上界。

## 2. 精确模型

对每侧四个非拇指 finger chain 的 16 条 edge 独立拟合固定 similarity。Take01 的 glove wrist-local bone vector 为 \(x_{f,b}\)，MOCAP wrist-local target 为 \(y_{f,b}\)：

\[
\min_{R_b\in SO(3),\,s_b}\sum_f
\left\lVert s_b x_{f,b}R_b^T-y_{f,b}\right\rVert^2
\]

每条 edge 先全帧 Kabsch + scale，再按 frame residual 做一次 MAD 裁剪后 refit。应用时从 wrist 开始沿每根手指递推：

\[
P_c=P_p+s_b(L_c-L_p)R_b^T
\]

然后用目标帧 MOCAP wrist GlobalQ 和 translation 变到世界坐标。Take02/03 的 target finger joints 只用于 metrics，不进入 fit 或 apply。

模型规模是每侧 \(16\times(3+1)=64\) continuous DoF，两侧共 128 DoF，是当前每侧 canonical R+scale baseline 的 16 倍。它只能称为 fixed take-level moderate-capacity probe，不能称为低容量模型。

## 3. 冻结评估结果

下表为 pooled joint/tip EPE 的 `median / P95`，coverage 与 strict25 baseline 完全相同。

| Take | Side | Root Fusion v1 joint | Edge probe joint | Root Fusion v1 tip | Edge probe tip |
|---|---|---:|---:|---:|---:|
| 01 | Left | 13.33 / 82.65 | 9.28 / 44.89 | 22.04 / 102.36 | 22.14 / 67.26 |
| 01 | Right | 11.90 / 85.44 | 9.52 / 46.96 | 19.07 / 104.36 | 20.13 / 69.91 |
| 02 | Left | 11.58 / 86.83 | 10.29 / 49.45 | 18.33 / 104.92 | 23.75 / 69.38 |
| 02 | Right | 11.48 / 88.43 | 10.12 / 54.28 | 19.06 / 107.58 | 19.96 / 75.88 |
| 03 | Left | 12.86 / 95.64 | 12.05 / 58.78 | 23.18 / 113.95 | 26.88 / 70.10 |
| 03 | Right | 13.10 / 97.77 | 11.30 / 59.71 | 26.17 / 109.96 | 24.29 / 77.25 |

P95 明显降低，但 tip median 并非全面改善，例如 Take02 left 从 18.33 mm 变为 23.75 mm。它不是严格 Pareto improvement。

## 4. 为什么现在否决产品化

1. **可辨识性不足。** 16 条 edge 的 covariance 最小/最大奇异值比只有 0–0.0126；四条 wrist→MCP 静态 edge 为 0，完整 3D rotation 的 twist 不可辨识。
2. **记住了 rig palm。** MCP pooled median/P95 约 0.001/0.002 mm，说明模型逐指记住目标骨架的 palm geometry，不能把 pooled 指标改善都解释成 articulation 更准。
3. **使用全部 finger labels。** Take01 的 16 个非拇指 target joints、包括 4 tips，全部进入 fit；因此 `target_finger_joints_used_in_fit=true`、`fingertips_used_in_fit=true`。
4. **非物理 FK。** 点链位置连续，但每条 edge 都在共同 root-local frame 独立旋转；没有 parent joint-frame composition、joint limits、解剖耦合或 temporal regularization。
5. **thumb 实现缺失。** 原 probe 未拟合 thumb 且将其保持为零；因为当前指标排除 thumb，错误没有在数字中暴露。任何全手视频都必须明确使用 base-fusion passthrough 或实现独立 thumb contract。
6. **动作覆盖敏感。** Take01 first-half 和 second-half 分别拟合后，holdout tip P95 可相差约 6–13 mm，说明参数依赖 source gesture coverage。

## 5. 如果继续实验，必须满足

- 单独实现 `experimental` fitter 和 versioned profile；不要新增默认 pose mode，也不要用一个 boolean 隐藏 fit provenance。
- 建议未来接口为 `--pose-mode mocap-root-fusion --articulation-profile PATH`；默认无 profile 仍执行 target-finger-free v1。
- Profile schema 必须写：source/input SHA、base profile SHA、128 DoF、fit joint/index、tips/target fingers used、thumb policy、每 edge rotation/scale/singular values/MAD residual、无 regularization、无 joint limits及明确 claim boundary。
- Apply 泄漏测试：把 Take02/03 target finger points 设为 NaN 或随机值后，prediction 必须 bitwise 不变。
- 完成 synthetic recovery、outlier、degenerate rejection、profile roundtrip、exact topology、chain continuity、quaternion sign invariance 和 thumb policy 测试。
- 模型选择必须先做 Take01 blocked/gesture CV。由于本轮已经查看并用 Take02/03 讨论模型取舍，它们今后都只能作为 retrospective validation/stress；锁定 profile 和超参后必须新采 untouched Take04（最好包含重戴/re-neutral）作为 final test，跨人 claim 还需要新 subject。
- 始终分开报告 MCP/intermediate/tip、frame/pooled median/P95/max、coverage 和 bootstrap CI。
- 必做 ablation：global R+scale、edge-length-only、edge R+scale；若简单模型足够，不使用 128 DoF。

## 6. 当前决定

MOCAP Root Fusion v1 保持当前推荐 calibrated baseline：它不使用 Take01 target fingers 拟合 canonical 参数，并诚实保留高 P95。Supervised edge probe 只作为“现有数据上可能降低长尾的研究上界”，没有仓库实现、正式 profile、正式 metrics 或视频，不进入 evidence claim。
