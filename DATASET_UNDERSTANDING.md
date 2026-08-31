# 01/02/03 数据理解与可视化口径

## 结论

01、02、03 都具备完成双手手套姿态与动捕手骨架同步比较的必要数据。`movementcap_worldcalib_pointcloud_package_20260830` 提供固定的 `mocap world -> RGB color camera` forward calibration，因此动捕 21 点可以投到 RGB。手套 20 点没有全局腕平移，不能单独直接投影；融合视图采用曝光时刻的 MOCAP wrist SE(3) 作为根节点条件输入。

2026-08-31 发现并修复了关键时间语义：`thor_monotonic_ns` 和 `cmavatar_windows_estimated_ns` 是 host poll/callback 时刻，比 `color_device_timestamp_us` 表示的相机采集晚约 254–255 ms。当前逐行减去这段延迟，再把 120 Hz MOCAP 插值到每个 RGB 采集时刻；旧 poll-time 产物不可继续用于同步结论。

00 不进入本次手姿态比较。它没有有效手套 IMU，但新增标定包用它的 CS-400 标尺和 RGB-D 求出了供 01/02/03 复用的相机外参。

## 三段正式数据

| 段 | RGB 帧 | Human.cma 帧 | 手套关键点帧/手 | MOCAP-only 发布帧 | capture→poll median/P95 | motion corr 旧→新 |
|---|---:|---:|---:|---:|---:|---:|
| 01 / Take_000 | 1815 | 7215 | 6943 / 6943 | 1748 | 255.103 / 270.295 ms | 0.626 → 0.876 |
| 02 / Take_001 | 1812 | 7482 | 6791 / 6793 | 1805 | 255.250 / 270.358 ms | 0.488 → 0.899 |
| 03 / Take_002 | 1810 | 7349 | 6688 / 6688 | 1800 | 254.008 / 269.719 ms | 0.395 → 0.882 |

发布片段分别裁剪到一个连续、无内部缺口、每帧都有 120 Hz MOCAP bracket 的共同区间。视觉相关峰残余为 +16/+10/+17 ms，均小于一帧，但只用于 QA，未作为逐段 offset 回写。

## 每段目录的角色

- `视频/RGB.mp4`：1920x1080、固定 30 FPS 的展示视频。
- `视频/Depth.mp4`：640x576 深度预览，不是可直接用于毫米计算的原始深度。
- `原始BAG与内参/camera_1_rgb_depth.bag`：原始 RGB-D 与真实设备时间；RGB topic 为 `/cam/sensor_2/frameType_2`。
- `同步校验/camera_cmavatar_alignment.csv`：相机设备时钟、Thor 单调时钟和 CMAvatar 时钟之间的桥。
- `同步校验/common_interval_sync_report.json`：交付的共同区间/clock uncertainty 报告；它不是硬件同步真值。
- `手套解算/solved/primary/*_hand_keypoints.csv`：每只手 20 点，单位 m；`world = wrist quaternion * local`，但 wrist xyz 始终为零。
- `手套解算/video_aligned/*_camera_aligned_pose.csv`：相机时间对齐后的姿态/角度，不含相机空间 xyz。
- `动捕/Take_xxx/Take_xxx_Human.cma`：需要使用的完整手骨架，单位 mm。
- `动捕/Take_xxx/Take_xxx_Body.cma`：手、前臂等刚体中心，不是完整手指骨架。
- `动捕/*.bvh`：120 Hz 导出，开头含额外全零 motion row；当前实现优先读取 CMA。

## Human.cma 结构

每只手使用 21 个点：

```text
Hand root
├── Thumb1 -> Thumb2 -> Thumb3 -> Thumb4
├── Index1 -> Index2 -> Index3 -> Index4
├── Middle1 -> Middle2 -> Middle3 -> Middle4
├── Ring1 -> Ring2 -> Ring3 -> Ring4
└── Pinky1 -> Pinky2 -> Pinky3 -> Pinky4
```

- `Skeleton_0_LeftHand*` 是有效左手；
- `Skeleton_1_RightHand*` 是有效右手；
- `Skeleton_0_RightHand*` 和 `Skeleton_1_LeftHand*` 为全零占位，必须过滤。

三段共 22,046 个动捕帧，所选 42 个点均为有限数且 FrameCounter 连续。全量点经新增外参投影后均为正深度并落在 1920x1080 画面范围内；这只是结构和坐标方向的 sanity check，不等于像素精度证明。

## 精确视频时间链

MP4 是固定 30 FPS，但 BAG 中 RGB 实际平均约 29.912 Hz。直接使用 `frame_index / 30` 作为绝对时间，到段末会累计约 176-178 ms 偏差。

交付关系是：

```text
MP4 frame i == BAG color message i
```

三段的 BAG color 消息数与 MP4 帧数完全相等；实现读取每条 BAG RGB message 的 device timestamp，不把 `camera_timecode_id` 当视频帧号，也不使用 `frame/30` 查询 MOCAP。index 身份必须再通过内容级 fail-closed 验收，不能只由“数量相等”推出。

正式发布用 `--frame-content-samples 0` 完成 BAG↔MP4 全帧内容级同帧 gate。对每个 BAG RGB message `i`，在 source MP4 的 `i-2...i+2` 中搜索最接近的解码画面，并在 240×135 灰度图上检查 MSE、PSNR 与感知 dHash：

| Take | raw best-offset=0（比例） | 可辨帧 best-offset=0 | MSE max | PSNR min | dHash max |
|---|---:|---:|---:|---:|---:|
| 01 | 1764/1815（0.9719008264） | 1715/1715 | 4.7364506721 | 41.3762734217 dB | 3 |
| 02 | 1768/1812（0.9757174393） | 1729/1729 | 4.6256175041 | 41.4791064309 dB | 3 |
| 03 | 1809/1810（0.9994475138） | 1794/1794 | 4.4396605492 | 41.6573059509 dB | 3 |

三段 global best offset 都是 0，content/timecode/temporal acceptance 均 PASS。raw best-offset=0 小于总帧数是因为静止或近重复画面在 ±2 帧窗口内不可区分；所有可辨帧都选择 offset 0。验收门限为 MSE max≤6、PSNR min≥39 dB、dHash max≤4、可辨帧≥24 且全部 offset=0、raw offset-0 fraction≥0.90。默认 31 帧模式只作快速回归，正式发布必须使用 `0=全帧`。

这里的“全帧”是每个 BAG/MP4 index 都完成 240×135 解码灰度与 dHash 内容比较，不是原始 MJPG/MP4 编码 bit-exact；它不验证 MOCAP 空间投影精度或硬件零时差。

正确时钟链为：

```text
latency(row) = thor_realtime_ns - color_device_timestamp_us * 1000
thor_acquisition(row) = thor_monotonic_ns - latency(row)
cmavatar_acquisition(row) = cmavatar_windows_estimated_ns - latency(row)
RGB device time -> corrected acquisition clock -> 120 Hz MOCAP bracket
```

`Human.cma.PtpTimeStamp` 用于高分辨率相对 cadence，并 affine anchor 到 Windows `TimeStamp`；absolute PTP epoch/Grandmaster 契约尚未交付。

## 新增空间标定

关键文件：

```text
movementcap_worldcalib_pointcloud_package_20260830/
  movementcap_ruler_worldcalib/results/manual_final/camera_to_world.json
```

坐标定义为右手系、单位 mm：原点在 CS-400 长短臂虚拟交点正下方的桌面，+X 沿长臂向外，+Y 沿短臂向外，+Z 为桌面法向向上。

标定包声明：

- 方法：人工识别 CS-400 标记 + 109 帧原始毫米深度中值；
- 桌面拟合残差 P50/P95：0.767 / 3.912 mm；
- 长短臂实测夹角：90.949°；
- 四个标记到对应 RGB 轴线最大垂距约 2.43 px；
- 复用于 01/02/03，依赖固定相机假设。

投影使用 `world_to_color_camera`、RGB 内参和 Orbbec rational Brown 8 项畸变。JSON 中畸变顺序为 `k1,k2,k3,k4,k5,k6,p1,p2`，不能当作普通 5 项 OpenCV 畸变直接使用。

随包三张 validation 图每段只用一个中间时刻的 `Body.cma` 左右手刚体中心；它们支持坐标方向和落点大体正确，但没有验证 `Human.cma` 的完整手关节像素误差。Human root 与 Body 手刚体中心也不是同一点，稳定相差约 69.5 mm。

## 当前手套配准与比较

推荐 root-fusion 只在 Take 01、每只手拟合一次 canonical rotation + scale，再冻结到 02/03：

```text
mocap_root_local_MCP ~= scale * canonical_R * glove_local_MCP
glove_in_world(t) = mocap_wrist_position(t)
                  + mocap_wrist_rotation(t) * scale * canonical_R * glove_local(t)
```

- 只用 index/middle/ring/pinky 四个 MCP 拟合固定 `R + scale`；
- 每帧严格使用曝光时刻动捕 wrist translation + orientation；
- 不逐帧重估 canonical rotation；
- 四个非拇指指尖是 holdout，不参与拟合；
- 拇指 20/21 点拓扑不同，只绘制、不进入指标。

当前固定配准后的差异如下。它们是共享 wrist 和同段配准后的 3D 一致性，不是独立物理精度：

| 段 | 左手非拇指关节 / 指尖 median | 右手非拇指关节 / 指尖 median |
|---|---:|---:|
| 01 | 13.44 / 22.29 mm | 11.94 / 19.42 mm |
| 02 | 11.49 / 18.05 mm | 11.39 / 18.88 mm |
| 03 | 12.94 / 23.27 mm | 13.19 / 26.38 mm |

这些数值是 MOCAP wrist SE(3)-conditioned articulation 一致性，不是独立 glove wrist/6DoF accuracy；Take 02/03 的 P95 仍高达 joint 86.06–97.44 mm、tip 104.87–113.80 mm。

## 当前不能宣称的内容

- 不能把上述 mm 差异称作手套的绝对精度；
- 不能把标定包的桌面拟合残差称作手关节重投影误差；
- 不能说手套自身提供了相机中的全局 6DoF 位姿；其全局平移来自动捕腕部；
- 不能用三张单帧 Body 圆圈图证明全序列 Human 手指像素准确；
- 不能把 240×135 解码灰度+dHash 的全帧内容 gate 写成原始编码 bit-exact 验证，也不能用它证明空间精度或硬件零时差；
- 不能在相机移动后继续无条件复用当前固定外参。
- 不能把曝光时刻到最近 MOCAP sample 的约 2.1/4.0 ms median/P95 称为端到端同步误差；它只是 120 Hz 采样量化距离。
- 没有共享 TTL/同步 LED 时，不能宣称硬件级零时间误差。
