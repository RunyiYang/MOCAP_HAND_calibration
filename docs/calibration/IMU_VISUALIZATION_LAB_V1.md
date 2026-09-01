# IMU → MOCAP 可视化方案实验室 V1

## 当前状态

- 数据范围已固定：动作段只使用
  `thor_new4_20260831_processed` 的 `Take_005`、`Take_006` 和
  `Take_007`，不再混入旧 Take 01/02/03 BVH 结果。
- 发布矩阵：4 个新数据视频段 × 7 个方法 = 28 个 MP4；
  `Take_006` 被分成两个不重叠窗口。
- 输出目录：`imu_mocap_visualization_lab/`。
- 网页入口：`/downloads/imu-visual-lab/`，可切换视频段与方法，
  并与同一时间轴的 3D 手骨架同步。
- 默认方案：`guided_ik` / Anatomical IK；它用 fixed-phalanx
  constant-curvature endpoint IK 消除旧版 PIP/DIP 镜像分支。
- 状态：`verified_new_data_only_4x7_video_3d_web`。28 个 MP4 已全部
  full decode，checksum inventory 闭合，本地页和临时 Quick Tunnel 均由
  Chrome 实测加载 3D；旧混合数据包的 hash/指标没有沿用。

这是 canonical final-nine 和 `imu_mocap_comparison_delivery/` 之外的
离线视觉实验包，不覆盖原始差异证据。

## 新数据盘点与发布切分

| Recording / CMM | 原始盘点 | 发布窗口 | 发布 ID | 用途 |
|---|---|---|---|---|
| `161402 / Take_005` | RGB 1868 帧；20J solved pose 和 CMM 均存在 | RGB `[0,1867)`，1867 帧 | `take005` | 完整同步动作段 |
| `161610 / Take_006` | RGB 2620 帧、Depth 2619 帧；20J solved pose 和 CMM 均存在 | RGB `[0,1309)`，1309 帧 | `take006a` | 前半段 |
| `161610 / Take_006` | 同上 | RGB `[1309,2619)`，1310 帧 | `take006b` | 后半段，与 A 无重叠 |
| `161912 / Take_007` | RGB/Depth 1982 帧；20J solved pose 和 CMM 均存在 | RGB `[0,1981)`，1981 帧 | `take007` | 完整同步动作段 |
| `155410` | RGB/Depth 98 帧；`0` glove packets；无 solved pose；无 CMM | 不发布为动作段 | 无 | calibration-only evidence |

表中“完整”指各 recording 可用的 RGB↔solver↔CMM 共同发布区间，
不是强行使用容器中没有同步锚的尾帧。`155410` 只证明无手套
场景、RGB 内参与 CS-400 world calibration provenance；它不能生成
IMU→MOCAP 手部 demo，也不允许伪装成第四个动作 take。

## CMM marker 表是权威对应

三个 CMM take 都使用用户提供的编号表与实物照片。对应关系不是
根据空间距离猜测，也不是通过最近点逐帧重分配。编号语义固定为：

- `#1/#3/#5/#7/#9`：thumb/index/middle/ring/pinky tip。
- `#2/#4/#6/#8/#10`：五指 proximal/base surface marker。
- `#11`：手背黑色模块 reflector，只用来构造 virtual wrist，不当作
  第 21 个手部关节。

| # / 语义 | Take_005 L / R | Take_006 L / R | Take_007 L / R |
|---|---|---|---|
| 1 thumb tip | `11781` / `11769` | `11781` / `12058` | `11781→12503` / `12436` |
| 2 thumb base | `11777` / `11622` | `11978` / `12069` | `11978` / `12435` |
| 3 index tip | `11773` / `11619` | `12003` / `11619` | `12003` / `11619` |
| 4 index base | `11780` / `11614` | `12001` / `11614` | `12001` / `11614` |
| 5 middle tip | `11772` / `11616` | `12006` / `11616` | `12006` / `11616` |
| 6 middle base | `11779` / `11678` | `12013` / `11678` | `12013` / `12434` |
| 7 ring tip | `11609` / `11625` | `12045` / `11625` | `12045` / `11625` |
| 8 ring base | `11782` / `11685` | `12015` / `11685` | `12015` / `11685` |
| 9 pinky tip | `11787` / `11627` | `12076` / `11627` | `12076` / `11627` |
| 10 pinky base | `11786` / `11630` | `12051` / `11630` | `12051` / `11630` |
| 11 dorsum module | `11785` / `11629` | `12064` / `11629` | `12064` / `12433` |

Take_007 左手 #1 的 `11781→12503` 是同一物理 reflector 的 re-ID alias，
不是两个 marker。Loader 把它们合并成一条 logical track，只在两帧短缺口
内插值。CMM 点位于手套表面，不是 anatomical joint center。

## 统一空间显示合同

- 四个动作段都使用各自 recording 的 RGB intrinsics、RGB↔CMM
  timestamp 与同一 tabletop world calibration。
- operator translation 为 `X=0, Y=-44, Z=0 mm`，应用在每个动作
  recording 自己的 MOCAP world 坐标中。它是 display correction，不是
  独立标定出的物理外参，也不是 GT accuracy 证据。
- `155410` 没有手部层，因此不应用 `Y=-44 mm` 手部 display
  correction。
- #11 与四个 non-thumb base 构造 virtual wrist，并沿手背方向再向后
  `20 mm`；该点是可视化 root proxy，不是实测掌根关节。

前三段完整录制在前 20% 严格帧上做一次固定 palm-base similarity。该值
衡量 raw solver 与 CMM 掌部方向的一致性，不是 posterior/IK 的最终误差：

| Source Take | Left median / P95 | Right median / P95 |
|---|---:|---:|
| `Take_005` | 9.46 / 19.93 mm | 9.70 / 17.83 mm |
| `Take_006` | 11.69 / 18.78 mm | 12.10 / 18.85 mm |
| `Take_007` | 34.63 / 60.01 mm | 38.02 / 64.33 mm |

Take_007 是显著 raw IMU/配准质量异常。#1..#11 表、tip/base 距离、左右手
投影和 `11781→12503` 连续轨迹均已独立核对，不能为了降低该残差重排
权威 marker ID。最终页面保留 raw 三方案，也单独展示明确读取同帧 CMM 的
posterior/RBF/IK，避免把视觉贴合误称为 raw IMU 精度。

对“是否 Z 轴翻转”另做了 proper-SO(3) / det=-1 / 显式 Z reflection
交叉审计。当前 proper SO(3) 的 ordered palm-normal 误差在 Take_005/006
分别为 `9.2°/5.3°`，det=-1 会变成 `173.2°/174.1°`；直接翻 solver Z
会把 all-10 CMM median/P95 从 `35.9/91.3 mm` 恶化到
`80.0/179.3 mm`，RGB projection 从 `43.2/118.4 px` 恶化到
`130.4/302.0 px`。因此禁止做全局 Z reflection；旧画面中“像 Z 翻了”
的部分来自每指 PIP/DIP 独立选择镜像 IK branch。

## 七种方案

| ID | 监督范围 | 定义 | 用途 |
|---|---|---|---|
| `s2_continuous` | IMU articulation + MOCAP spatial calibration | 按真实 solver timestamp 对每根骨方向做 S² SLERP，骨长线性插值 | 无闪烁连续基线 |
| `s2_gaussian` | IMU articulation + MOCAP spatial calibration | 对骨方向做对称 9-tap S² Gaussian，`σ=1.8 frame`，骨长固定为序列 median | 普通零相位平滑 |
| `kalman_rts` | IMU articulation + MOCAP spatial calibration | constant-velocity Kalman forward pass + RTS backward pass，再投影到固定骨长 | 离线 Kalman 对照 |
| `posterior_75` | MOCAP-conditioned | MOCAP 骨方向 + 25% 平滑 IMU tangent residual | 软融合 |
| `posterior_98` | MOCAP-conditioned | MOCAP 骨方向 + 2% 平滑 IMU tangent residual | 强监督 posterior 对照 |
| `rbf_self_fit` | same-sequence teacher-fit | 用本段全部帧训练 IMU pose+velocity → 本段 MOCAP target 的 Gaussian RBF network | 同段 mapping 拟合上限 |
| `guided_ik` | MOCAP-conditioned oracle | CMM tip/base endpoint + fixed phalanx + constant curvature；`q_DIP=0.65 q_PIP`，PIP≤110°、DIP≤75°、thumb≤115° | 默认 CMM-guided 解剖约束视觉上界 |

前三种方法不读取每帧 CMM finger target，但仍使用 CMM 派生的固定
空间标定/root anchoring；因此最准确的名称是
`IMU articulation · MOCAP-calibrated`，不是 glove-only absolute 6DoF。
后四种方法直接或间接读取本段 MOCAP，不是独立 IMU 输出。

## 时间连续与流形处理

对 solver pose 先转成 19 根 parent→child 骨向量。目标 RGB 时间 `t` 落在
`t0,t1` 时，单位方向和骨长用：

```text
u(t) = Slerp_S2(u0, u1, (t-t0)/(t1-t0))
L(t) = (1-a)L0 + aL1
```

这里不再用 20/25 ms validity gate 把骨架隐藏。S² Gaussian 使用半径 4、
`σ=1.8` 的对称权重，是最大使用未来 4 个 RGB 帧的离线零相位
smoother，不能直接宣称为实时算法。

MOCAP-guided posterior 在 S² tangent space 计算：

```text
r_t = Log_S2(u_mocap_t -> u_imu_t)
r_bar_t = symmetric_gaussian(r_t),  ||r_bar_t|| <= 60 deg
u_out_t = Exp_S2(u_mocap_t, gain * r_bar_t)
```

`posterior_75` 使用 `gain=0.25`，`posterior_98` 使用 `gain=0.02`。
后者即 98% MOCAP conditioning / 2% IMU residual，只适合已有录制的最佳
展示，不是独立预测。

## CMM-guided anatomical IK 与 RBF 边界

CMM endpoint 先使用 `[1,2,1]/4` 对称平滑。旧 nested sphere IK 会让
PIP 与 DIP 独立选择圆交点镜像分支；真实四段审计发现 non-thumb
bend-normal 反号率为 `35%–69%`，所以会出现局部 S 形，看起来像单根手指
的 Z 被翻转。

当前 non-thumb 改为 exact constant-curvature solver：固定每段 phalanx
length，令 `q_DIP=0.65*q_PIP`，用一维二分求满足 measured base→tip span
的 `q_PIP`，整根手指只使用一个平面和一个 bend-normal hemisphere。
首帧由平滑 IMU guide 选择镜像分支，后续帧由 previous bend normal
硬锁时间连续；thumb 使用同样 branch lock 的解析 two-link IK。构造同时
硬限制 PIP≤110°、DIP≤75°、thumb≤115°。

真实 Take_005/006/007 六只手均通过：反号率全部为 `0`；PIP max
`84.27°–93.23°`，DIP max `54.78°–60.60°`，thumb max
`83.93°–113.40°`；fixed-bone drift max `6.8e-14 mm`；bend-plane
temporal delta P95 `2.23°–5.63°`。这仍使用同帧 CMM surface endpoint，
因而只是视觉 oracle，不是独立 IMU 或 anatomical-joint accuracy。

RBF 的输入是 root-relative 19×3 IMU position 和 finite-difference velocity。
按用户要求，全部视频帧同时是训练集和渲染集，`test_frames=0`；
输入中不放 frame/time index。它只能表示 same-sequence teacher-fit adherence，
不能表示新动作泛化。

## 2D + 3D 网页合同

- 2D 视频每帧都画 MOCAP 与当前 method pose；不闪烁、不因
  validity gate 消失、不画 joint/error connector，不用一根线标记差距。
- 3D Canvas 与 video currentTime 共用时间轴，展示 MOCAP left/right
  和当前 method left/right 四层骨架。
- 3D 支持 orbit、pan、zoom、reset view 和 timeline scrub；切换方法时
  保留当前 take 的帧位置与视角。
- 每个 method/take 的 motion JSON 与 MP4 都必须受 manifest hash 约束；
  加载失败时清空旧 3D 状态，避免把上一段姿态误当成当前数据。

## 复现与验收

```bash
cd /home/runyi/Project/hands_reloc/GT_calib

uv run gt-calib-delivery build-imu-visual-lab \
  --destination rebuilt_imu_mocap_visualization_lab \
  --manual-profile calibration_profiles/operator_y_minus_44_all_hand_overlays.v1.json

uv run gt-calib-delivery validate-imu-visual-lab \
  --destination rebuilt_imu_mocap_visualization_lab --full-decode

python web/build_site.py --source
```

最终验收结果：

- 28/28 MP4 full decode 通过，共 45,269 帧；全部为 H.264/yuv420p、
  960×540、30 FPS、faststart。
- 119 个 regular files、118 条 checksum、361,055,098 bytes；
  `sha256sum -c` 全部通过。Manifest SHA-256：
  `a0cbc268d94b91751c0aaf41b1a78f0aa7eec14cae6800d643b543ef912d79d6`。
- 45,269/45,269 帧左右 RESULT 骨架均可见；28 个 motion JSON 均为
  62 joints、0.1 mm 量化并与对应视频逐帧同长。
- 本地与 Quick Tunnel 页面均为 HTTP 200，MP4 Range 均为 HTTP 206；
  实际 Chrome 默认加载 `Take_007 / guided_ik` 后显示
  `3D ready · 62 joints · 1981 frames`。

四段平均的 CMM surface-marker reference 排名如下；监督范围不同，不能
把它当作公平的 IMU benchmark：

| Rank | Method | Median / P95 mm | Jitter P95 mm |
|---:|---|---:|---:|
| 1 | `rbf_self_fit` | 0.065 / 0.309 | 0.482 |
| 2 | `guided_ik` | 0.065 / 0.309 | 0.482 |
| 3 | `posterior_98` | 0.540 / 1.638 | 0.480 |
| 4 | `posterior_75` | 6.562 / 19.751 | 0.462 |
| 5 | `s2_gaussian` | 36.846 / 87.124 | 0.119 |
| 6 | `kalman_rts` | 36.859 / 87.149 | 0.101 |
| 7 | `s2_continuous` | 36.873 / 87.161 | 0.493 |

## Claim boundary

- 可以宣称发布为 4 个新数据段 × 7 个方法，且 `Take_006 A/B`
  为无重叠窗口；当前新包已完成 28/28 full decode、checksum、逐帧
  可见性、3D motion 和浏览器运行时验收。
- 可以宣称 marker ID 对应来自用户提供的编号表和照片；
  `11781→12503` 是 Take_007 左 thumb-tip logical alias。
- 不能把 `posterior_75` / `posterior_98` / `rbf_self_fit` / `guided_ik`
  称为 independent IMU accuracy。
- 不能把 `test_frames=0` 的同段拟合称为泛化或测试精度。
- 不能报告 glove-only absolute wrist translation / 6DoF accuracy。
- 不能把 CMM surface marker residual 直接解释为 anatomical joint error。
- 不能把 `Y=-44 mm` 解释为独立物理标定或 GT 精度。
- 不能把 `155410` 称为有手动作段。
