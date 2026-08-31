# Depth Camera Pose Audit v1

- 状态：`depth_se3_reproducible_fixed_view_supported_color_pose_review`
- 运行：`run-20260831-depth-camera-pose-audit-v1`
- 主结果：`outputs/depth_calibration_audit/depth_camera_pose_audit.json`
- 候选而非正式结果：`outputs/depth_calibration_audit/d2c_rigid_candidate.json`

## 结论

数据里有 camera pose，但它是一个固定外参，不是逐帧 trajectory：

```text
T_world_from_depth =
[[-0.995519,  0.026543,  0.090763, -55.047]
 [-0.068895,  0.453880, -0.888396, 845.201]
 [-0.064776, -0.890668, -0.450017, 547.291]
 [ 0,         0,         0,          1     ]]
```

深度相机中心在 MOCAP/CS-400 world 中为 `[-55.047, 845.201, 547.291] mm`。其旋转行列式为 `1.0`，最大正交误差和正反矩阵闭合误差均为 `2.22e-16`，因此 depth-camera pose 是严格 SE(3)。

四个 BAG 没有 `/tf`、pose、odometry 或 trajectory topic。两路 IMU 的 orientation 均为零，只能用于重力/静止 sanity，不能提供 yaw 或平移。因此不能把数据解释为“每帧都有 camera pose”。

当前建议：

- depth-space 计算继续使用现有 `depth_camera_to_world / world_to_depth_camera`；
- 物理相机固定假设已有新的静态 RGB 背景证据支持；
- color-space 投影暂时沿用 delivered D2C forward affine 作为 review baseline，但不能把它叫作经过验证的 color-camera 6DoF pose；
- 不根据单一桌面平面给 01–03 做 per-Take pose correction。

## 标定是怎样得到的

00 的 109 帧 raw `mono16` 毫米深度先做时序中值，再拟合桌面平面。四个 CS-400 反光 marker 在 depth 中主要表现为无效深度孔，因此它们的 3D 中心不是直接深度量测，而是：

```text
manual depth pixel ray ∩ (table plane + 45 mm marker-center height)
```

桌面提供 roll、pitch 和法向平移；两条有语义的尺臂及其交点补足 yaw 和桌面内 x/y。每条尺臂只有两个 marker，且没有额外 marker holdout。

## 00 内部复算与重复性

完整 109 帧复算与 delivered JSON 的最大矩阵差为 `1.28e-13`，world origin 最大差为 `5.68e-14 mm`。

按连续时间分成 first/middle/last 三块，各自重新做 depth median、plane 和 pose：

| 指标 | 三块相对 109 帧全量的最大值 | 门限 | 结果 |
|---|---:|---:|---|
| depth pose rotation delta | 0.0573° | 0.1° | PASS |
| world origin in depth delta | 0.4743 mm | 1 mm | PASS |

这证明相同人工 marker 像素条件下的 depth 重复性，不是独立准确率。

原报告桌面 P50/P95 为 `0.767/3.912 mm`，但 P95 在计算前排除了所有 `>=30 mm` 的点。对全部 103,535 个候选点不截断时：

| P50 | P90 | P95 | P99 | 最大值 | `>=30 mm` 点数 |
|---:|---:|---:|---:|---:|---:|
| 0.781 mm | 2.572 mm | 6.176 mm | 35.454 mm | 126.320 mm | 1,440 |

后续报告必须保留全体分布；旧的 3.912 mm 只能标为 `residual < 30 mm` 条件分位数。

## 人工 marker 点选敏感度

以下是固定 plane、四点独立加入高斯像素扰动的 5,000 次情景分析。它是敏感度，不是对人工点选误差的实测置信区间。

| 每点假设噪声 | origin delta P50 / P95 | rotation delta P50 / P95 |
|---|---:|---:|
| sigma=0.5 px | 2.50 / 5.92 mm | 0.280 / 0.784° |
| sigma=1.0 px | 4.94 / 11.76 mm | 0.533 / 1.549° |
| sigma=2.0 px | 10.00 / 23.74 mm | 1.072 / 3.096° |

45 mm marker 高度先验每变化 1 mm，world origin 约变化 1.686 mm。现有数据没有实测 marker center 三维坐标或独立标记点，因此不能从重复性直接声称 absolute pose accuracy。

## 相机是否移动

原先 `camera_consistency` 只比较 serial number 和内参，不能证明相机没动。新审计在 00 与每个正式 Take 的 10%、50%、90% 三个时点上，对两侧静态背景做 SIFT、ratio test 和 partial-affine RANSAC：

| Take | 最大平移 | 最大旋转 | 最大尺度差 | 最少 RANSAC inliers | 结果 |
|---|---:|---:|---:|---:|---|
| 01 | 0.064 px | 0.0017° | 0.0000629 | 320 | PASS |
| 02 | 0.067 px | 0.0022° | 0.0000231 | 344 | PASS |
| 03 | 0.044 px | 0.0021° | 0.0000392 | 375 | PASS |

门限为 `0.5 px / 0.1° / 0.001 / >=100 inliers`。这为 fixed view 提供强证据，但仍是静态图像配准，不是独立公制 3D pose accuracy。

## raw-depth 跨 Take 警报

每个正式 Take 均匀取 61 帧 raw depth 做中值，并使用与 00 相同的桌面 ROI/RANSAC：

| Take | plane normal 相对 00 | plane d 相对 00 |
|---|---:|---:|
| 01 | 1.035° | -7.39 mm |
| 02 | 1.139° | -8.50 mm |
| 03 | 1.160° | -8.75 mm |

01–03 彼此接近，但与 00 不同；结果还会随桌面高度和左右 ROI 明显变化。结合 RGB 背景几乎完全不动，这更像 ToF 空间偏差、桌面局部、遮挡中值或场景条件差异的混合信号，不能直接解释为 camera moved，也不能把这些值直接 compose 成 pose correction。

当前将这项标为 `review`。下一步需要预定义静态非平面区域的 3D ICP、分区 plane map，以及独立锚定 world 的已知目标。

## 为什么 color camera pose 仍是 review

delivered depth-to-color 线性块来自 BAG 的 Orbbec stream profile；`camera_1_intrinsics.json` 同时声明 hardware D2C disabled，并把 rotation 数组留空。该 3x3 的指标为：

```text
det(A)                    = 0.9949142153
singular values           = [1.0000000092, 0.9999999723, 0.9949142337]
max |A^T A - I|           = 0.0101456425
```

所以它是 forward affine，不是旋转。将它做 polar decomposition 得到最近 SO(3) 后，在当前工作空间网格上的投影差为：

```text
P50 / P95 / max = 1.344 / 2.973 / 3.517 px
```

`d2c_rigid_candidate.json` 只为了后续 A/B 验证，绝不静默替换当前投影。必须拿官方 Orbbec D2C `R,t`，或采一组 RGB/depth 都可定位的 checkerboard/AprilTag/标定棒对应，使用独立 held-out pose 比较两者后才能升级。

## 复现

```bash
cd /home/runyi/Project/hands_reloc/GT_calib

python depth_calibration_audit.py \
  --formal-sample-count 61 \
  --monte-carlo-trials 5000
```

输出：

- `outputs/depth_calibration_audit/depth_camera_pose_audit.json`
- `outputs/depth_calibration_audit/depth_camera_pose_audit_summary.png`
- `outputs/depth_calibration_audit/rgb_fixed_camera_registration.jpg`
- `outputs/depth_calibration_audit/d2c_rigid_candidate.json`

单元测试：

```bash
python -m unittest tests.test_depth_calibration_audit -v
```

## 升级为 absolute GT 还需要什么

1. Femto Bolt 上固定至少 3 个、最好 4 个非共面 MOCAP marker，交付 rigid-body definition 与逐帧 raw marker/residual；这样每帧能直接得到 camera pose。
2. 或提供已知 3D 几何、同时被 MOCAP 和 RGB-D 观测的标定棒/板，不少于 30 个覆盖中心、边缘、近中远和不同朝向的 pose，至少 20% 整 pose 留出。
3. 官方 Orbbec depth-to-color 刚体 `R,t`、适用 profile、SDK/固件版本和导出方法。
4. 共享 TTL/曝光 timestamp contract，避免空间标定被时间偏差吸收。
5. held-out 2D/3D 对应点和 pre/post 标定物漂移记录；必须报告 coverage、未截断 residual、bootstrap pose CI 与 Jacobian rank。

在这些信息到位前，准确名称是 `depth-assisted fixed camera calibration estimate`，不是独立验证的 camera-to-MOCAP absolute GT。
