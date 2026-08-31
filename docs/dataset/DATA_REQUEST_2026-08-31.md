# 2026-08-31 数据索取清单

状态：`ready_to_send`
跟踪表：[request_datalist_2026-08-31.csv](request_datalist_2026-08-31.csv)

## 先说明：不需要重复发送什么

我们已经有 Take_000/001/002 的 `.tak`、`Human.cma`、`Body.cma`、BVH、RGB-D BAG、相机内参和同步表。请不要重复打包这些文件。现有同步表已经足以发现并逐行移除约 254–255 ms 的 `camera capture → host poll` 延迟，因此不要再提供人工调出来的 per-take offset。当前最高优先级是从已有 TAK 导出原始 Marker 与质量信息，并补齐骨架、坐标和硬件时间语义。

2026-08-31 depth 审计还确认：现有 `T_world_from_depth` 是可复算的固定 SE(3)，但不存在逐帧 Femto Bolt camera pose；delivered D2C 3×3 不是合法旋转。明天如果只能额外拿一类空间数据，优先要 **Femto Bolt 相机刚体的原始 MOCAP marker trajectory + rigid-body definition**。如果历史数据没有相机刚体，则新采 MOCAP 与 RGB-D 共同可见的标定棒/板，不少于 30 个完整 pose，并整 pose 留出至少 20%。

同日 wrist 审计确认画面前方大块是手背板、后方黑块才是腕端模块；现有文件没有该后侧模块的可信 6DoF。若只能额外拿一类手部 mount 数据，优先要 **前/后两个硬件块分别对应哪个 rigid body 的书面说明 + 后侧模块 marker definition 或 `T_rear_module_from_hand_skeleton` 实测固定变换**。没有这项时，当前额外圆点只能叫 display proxy。

## P0：决定能否把 MOCAP 称为 GT

明日按跟踪表中的九个可验收交付包收件；下文 P0-1–P0-7 给出字段级契约，其中相机相关内容在跟踪表中进一步拆成独立的 P0-8/P0-9，避免“有一份旧 calibration 文件”被误当作已经交付逐帧 camera pose 和 held-out 标定板：

| Request ID | 交付包 |
|---|---|
| REQ-P0-01 | 三个 TAK 的 raw Marker 与逐帧质量 |
| REQ-P0-02 | 手部 Marker Set、21 点骨架与 solver 配置 |
| REQ-P0-03 | 三段 Human joint quality |
| REQ-P0-04 | 历史相机—MOCAP 同场标定、坐标系、held-out 对应与官方 D2C |
| REQ-P0-05 | 手背板、后侧腕模块、Body Hand 与 glove/skeleton root 的物理关系 |
| REQ-P0-06 | 时间戳契约、logger/PTP 与 shared TTL/同步 LED |
| REQ-P0-07 | Glove session-neutral、raw IMU、solver assets 与 replay |
| REQ-P0-08 | Femto Bolt 相机 MOCAP 刚体、raw trajectory、质量与 optical-frame mount |
| REQ-P0-09 | 新采 RGB-D×MOCAP 共同标定棒/板、至少 30 个完整 pose 与整 pose held-out split |

机器可跟踪状态、owner 和 due date 以 [request_datalist_2026-08-31.csv](request_datalist_2026-08-31.csv) 为准。

### P0-1 原始 Marker 与逐帧质量

请分别从三个 TAK 导出：

```text
Take_000_markers_raw.c3d
Take_000_markers_raw.csv
Take_001_markers_raw.c3d
Take_001_markers_raw.csv
Take_002_markers_raw.c3d
Take_002_markers_raw.csv
```

CSV 使用 long-table，一行一个 Marker/帧，必须包含：

```text
take_id, frame_counter, capture_timestamp_ns, ptp_timestamp_ns
marker_id, marker_label
x_mm, y_mm, z_mm
tracked, occluded, gap_filled, interpolated
residual_mm, ray_count, camera_mask
stream_kind  # raw_labeled / raw_unlabeled / solved
```

要求：120 Hz；与 `Human.cma.FrameCounter` 一一对应；缺失值使用 `NaN + tracked=false`，不能用 `(0,0,0)`；不能静默补点；同时附导出软件名称、版本与导出设置截图。

### P0-2 手部 Marker Set、骨架和求解器配置

请提供 native project/asset，并补充可读文件：

```text
mocap_project_or_asset_native.*
hand_joints.csv
hand_markers.csv
human_solver_config.json
marker_placement_left.jpg
marker_placement_right.jpg
```

`hand_joints.csv` 至少包含：

```text
skeleton_id, side, joint_id, joint_name, parent_joint_id
rest_offset_x_mm, rest_offset_y_mm, rest_offset_z_mm
bone_length_mm, position_semantics, local_axis_convention
```

`hand_markers.csv` 至少包含：

```text
marker_id, marker_label, side, anatomical_landmark, parent_joint_id
offset_x_mm, offset_y_mm, offset_z_mm
```

同时说明 Skeleton_0/1 与左右手的关系、21 点中哪些是实体 Marker/模型推算点、骨长来源、filter/smoothing/gap-fill 参数、最大补点时长、软件与 solver 精确版本，以及四元数顺序和 active/passive 约定。

### P0-3 Human 骨架质量表

若软件支持，请导出每个 take 的：

```text
Take_xxx_Human_quality.csv
```

字段：

```text
take_id, frame_counter, joint_id, valid, confidence
source_marker_count, occluded, gap_filled, interpolated, fit_residual_mm
```

如果软件不支持 joint-level quality，请书面确认，但 P0-1 的 Marker 级质量仍必须提供。

### P0-4 相机—MOCAP 同场标定原始证据

优先寻找 2026-08-29 当时同一布置的标定 take：

```text
mocap_camera_calibration.cal
coordinate_frames.json
camera_mocap_correspondences.csv
orbbec_factory_intrinsics.json
orbbec_depth_to_color_extrinsics.json
calibration_rgbd.bag
calibration_mocap.c3d 或 calibration_mocap.tak
calibration_report.json
FemtoBolt_rigid_body_definition.json
camera_marker_trajectory.c3d
camera_marker_quality.csv
camera_rigid_mount_photos/
```

`coordinate_frames.json` 必须明确原点、轴向、左右手系、单位、矩阵 row/column-major，并统一用 `T_A_from_B` 与 `p_A = R_A_from_B p_B + t_A_from_B`。

对应点 CSV 至少包含：

```text
sample_id, capture_timestamp_ns, target_point_id
mocap_x_mm, mocap_y_mm, mocap_z_mm
rgb_u_px, rgb_v_px, depth_mm
mocap_valid, marker_residual_mm
split  # calibration / held_out_validation
```

请同时提供 Orbbec SDK/固件导出的原始 depth→color `R,t`、畸变模型与序列号；不要只给当前复合后的 world→color 矩阵。现有交付里的 depth→color 3×3 轻微非正交，必须确认这是官方模型、数值舍入还是额外 affine correction。

如果没有同场数据，需要新采标定棒/刚性标定板：不少于 30 个姿态，覆盖画面中心/边缘和近/中/远深度，至少 20% held-out。

相机刚体至少 3 个、最好 4 个非共面 Marker；必须给 marker 到 raw-depth optical frame 和 RGB optical frame 的固定 mount transform、逐帧 valid/camera-mask/residual/occlusion/gap-fill。这样才能独立比较逐帧 `T_world_from_depth` 与当前固定外参，而不是再次用手动作或桌面拟合自己的答案。

新标定棒/板的 train/validation/test 必须按完整 target pose 或空间 cell 切分，不能把同一静态 pose 的相邻帧随机拆到不同 split。每个对应应同时保存 MOCAP 3D、RGB 2D、raw-depth pixel/mm 和可见性；只有这样才能在 held-out pose 上比较 delivered D2C affine 与 rigid candidate。

### P0-5 手背板、后侧腕模块与 glove/skeleton root 的物理关系

请先逐项确认画面前方大手背板与后方小黑块分别是什么硬件、是否带 MOCAP markers、对应 `Body.cma.LeftHand/RightHand`、`Human.cma` wrist 或其他哪个 rigid body。不要只回答“都是手部设备”。请提供：

```text
left_hand_rigid_body_definition.json
right_hand_rigid_body_definition.json
T_glove_root_from_body_hand.json
T_rear_module_from_hand_skeleton_left.json
T_rear_module_from_hand_skeleton_right.json
rear_module_marker_definition_left.json       # 若后侧模块被直接追踪
rear_module_marker_definition_right.json      # 若后侧模块被直接追踪
hardware_identity_and_frame_contract.md
rigid_body_mount_photos/
Take_xxx_body_quality.csv
```

`hardware_identity_and_frame_contract.md` 必须在照片上标注：front dorsum plate、rear wrist module、左右手、各 rigid-body origin/XYZ 轴、marker ID，以及 `Human.cma` wrist 的定义。若后侧模块没有被 MOCAP 直接追踪，也必须书面确认，不得用空的 Forearm BVH 当作有效 6DoF。

质量表字段：位置、四元数、valid、marker_count、mean_marker_residual_mm、occluded、interpolated。固定变换必须由 CAD/量具或独立 calibration take 求得并冻结，左右手分别交付；至少重复拆装 3 次报告 translation/rotation repeatability，不能在目标 take 上按画面重新拟合。

### P0-6 时间戳语义

请提供 `timestamp_contract.md`、采集端同步 logger 源码/commit 和一份原始 trigger log，明确：

- CMA `FrameCounter / TimeStamp / PtpTimeStamp` 的单位、epoch、timezone、clock domain；
- 时间代表曝光/采样、接收还是 solver 输出；
- C3D frame 与 CMA FrameCounter 的映射；
- RGB/depth device timestamp 的定义；
- `color_device_timestamp_us` 是曝光起点、中点还是结束，rolling/global shutter 语义；
- `thor_realtime_ns / thor_monotonic_ns` 是采集、poll、callback 还是写盘时刻；
- `PtpTimeStamp` 是否真实 PTP、grandmaster ID、锁定状态与绝对 epoch；
- 同步表每一列在哪里采样，以及 logger 引入的延迟和不确定度。

下一次采集必须加入可独立验证的共同事件：优先用相机与 MOCAP 都记录的共享 TTL；若硬件接口不允许，则使用 RGB 可见且由 MOCAP 刚体携带的同步 LED，并记录驱动脉冲。正式动作前后各不少于 10 个不同间隔脉冲，输出 `sync_trigger_events.csv`，字段至少为 `event_id, device, edge, device_timestamp_ns, ptp_timestamp_ns, sequence, lock_status`。验收时需能在未查看手部动作的情况下估计 offset/drift/uncertainty；目标 P95 小于 1 ms，否则如实报告。

后续手套日志也需要设备侧 `sample_counter + device_timestamp_ns`，不能只有 USB host receive completion；同时保留 packet sequence、host USB start/complete、IMU quaternion/gyro/acc 和 status。

### P0-7 Glove solver 可重放输入与 session-neutral

现有三个 Take 只有 solver 输出和 meta 中的 hash/Windows 路径，没有实际使用的 neutral 与配置文件。请提供：

```text
Take_000_session_neutral.json
Take_001_session_neutral.json
Take_002_session_neutral.json
Take_xxx_left_hand_imu_raw.csv
Take_xxx_right_hand_imu_raw.csv
glove_solver_assets/
  skeleton.*
  left_calibration.*
  right_calibration.*
  left_solver_config.*
  right_solver_config.*
glove_solver_replay.md
```

原始 IMU 表至少包含：

```text
take_id, side, sample_counter, device_timestamp_ns
packet_sequence, host_usb_start_ns, host_usb_complete_ns
sensor_id, quaternion_w, quaternion_x, quaternion_y, quaternion_z
gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z, status
```

同时说明 quaternion 的 frame、顺序、active/passive、reset/recenter 与 `session_neutral` 语义、IMU/board 到手掌或指骨的 mount transform，以及 raw BAG 到 pose/keypoints 的完整重放命令和精确软件/commit/container 版本。配置文件 SHA256 必须与 `pose.meta.json` 中记录一致；在同一输入上应能数值复现当前 pose/keypoints，不能只交付原机器绝对路径。

## P1：显著增强可解释性

1. 每段分层抽取至少 100 帧原始 1920×1080 RGB，双人标注 21 关节与 visibility。
2. 提供被试手长、掌宽、各指骨段长度、手套尺码、Marker/IMU 安装照片。
3. 在模型/profile/阈值锁定后新采独立 calibration take + untouched `Take_004` final：open、fist、spread、逐指弯曲、指尖触碰；手指保持不动时做 wrist-only 三个非共线轴，均含 slow/normal/fast。`Take_004` 在首次跑分前不得用于调参，且至少包含一次脱下重戴和重新 neutral；若要跨人 claim，再增加新 subject holdout。
4. 每段提供 manifest：设备序列号、固件、软件/asset版本、neutral ID、extrinsic ID、相机是否移动、是否重戴。
5. 正式动作前后各录一次固定标定物，报告外参平移/旋转漂移与 held-out 重投影误差。

## 建议统一交付结构

```text
GT_request_20260831/
├── Take_000/
├── Take_001/
├── Take_002/
├── definitions/
├── camera_mocap_calibration/
├── timestamp_contract.md
├── manifest.json
├── README_schema.md
└── SHA256SUMS.txt
```

`manifest.json` 必须包含源 TAK SHA256、导出工具/版本、生成时间、单位、坐标系、timestamp 语义和全部文件 SHA256。

## 可直接发送给对方的短消息

> 我们已有 Take_000/001/002 的 TAK、CMA、BVH、RGB-D BAG 和同步表，不需要重复发送，也不要提供人工调出来的 per-take 时间偏移。请优先从原 TAK 导出带 residual、camera mask、occlusion/gap-fill 标志的原始 Marker C3D/CSV，并提供手部 Marker Set、Human 骨架/solver 配置、原始相机—MOCAP 标定与 Orbbec depth→color 官方外参，以及 FrameCounter/TimeStamp/PtpTimeStamp、相机曝光时间戳和同步 logger 的精确定义。下一次采集请增加共享 TTL；如硬件不支持，用 RGB 可见且 MOCAP 可追踪的同步 LED，并交付原始 trigger log。另请在安装照片上分别标出前方手背板和后方腕部黑块，说明它们各自对应哪个 rigid body；交付后侧模块 marker definition，或左右手分别实测的 `T_rear_module_from_hand_skeleton`，并提供 Body Hand 到 glove root 的固定变换。Glove 侧提供 session_neutral、solver assets、设备时间戳原始 IMU 和可重放命令。全部文件请附 schema、软件版本和 SHA256SUMS。
