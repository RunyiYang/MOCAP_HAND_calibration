# 数据契约

## 正式数据角色

- `00_210652_三秒视频对齐_无手套`：CS-400 与 RGB-D 空间标定参考，不参与手套姿态评估。
- `01_210814_Take_000`：当前跨 take 协议中的 registration/calibration split。
- `02_210955_Take_001`：冻结 01 配准后的 validation split。
- `03_211139_Take_002`：冻结 01 配准后的 test/stress split。
- `thor_new4_20260831_processed/...161912/mocap/Take_007`：最终新增 CMM
  visualization split；照片编号 #1..#10 为双手各十个表面点，#11 只构造
  virtual wrist。它是 same-take alignment，不加入旧 01→02/03 holdout 指标。
- `thor_new4_20260831_processed/...155410`：最终 video 09 的无手套 RGB 与
  intrinsics 来源；不参与动态手 pose fit。

同一个 take 不得同时被标为“独立拟合”和“独立测试”。如果改变 split，必须在日报和证据索引中创建新的 run ID。

## 稳定数据约定

- MOCAP `Human.cma`：左右手各 21 点，位置单位 mm，约 120 Hz。
- 手套 keypoints：每手 20 点，位置文件单位 m；读取后统一转为 mm。
- 手套 `world_wrist_xyz` 始终为零，不提供独立全局腕部平移。
- RGB 使用原始 BAG 的 color message 时间戳；`MP4 frame i == BAG color message i` 必须同时通过总数、时间戳和内容级抽样证据，不能只因数量相等而假定。
- 不得用 `frame_index / 30` 代替设备时间；段末会产生约 176–178 ms 漂移。
- `thor_monotonic_ns` 与 `cmavatar_windows_estimated_ns` 是 host poll/callback 时刻；映射前必须逐行减去 `thor_realtime_ns - color_device_timestamp_us*1000`。
- `PtpTimeStamp` 当前只作为高分辨率相对 cadence，并 affine anchor 到 Windows `TimeStamp`；absolute epoch/Grandmaster 尚未交付。
- 空间变换命名为 `T_A_from_B`，点变换采用 `p_A = R_A_from_B p_B + t_A_from_B`。
- 当前相机外参依赖固定相机假设；相机移动后必须重新标定或验证。
- 公制 depth 只能读取 BAG topic `/cam/sensor_3/frameType_3` 的 640×576 little-endian `mono16`；`Depth.mp4` 是 8-bit 伪彩预览，禁止用于毫米几何。
- RGB/depth message 数为 00 `109/109`、01 `1815/1816`、02 `1812/1812`、03 `1810/1810`。01 的最后一个 depth message 是 unmatched，不能为了数量相等而删除事实或错配。
- 同 index depth 比 RGB 晚约 1.31 ms。depth-space MOCAP 查询必须使用 depth message 自己的 device timestamp，不能直接复用 RGB 时间。
- 当前 `T_world_from_depth` 是合法固定 SE(3)；D2C forward 3×3 非正交，只能标为 `affine_unverified`，不能当作 color-camera rigid pose。

## Take_007 CMM 约定

- CMM 只有 120 Hz、mm 单位的 reflector XYZ；没有 joint angle、hierarchy、
  wrist quaternion 或 anatomical center，因此“20 点”不能写成“20 个旋转 DoF”。
- 每手显示照片 marker #1..#10，顺序为五个 `tip/base` pair；#11 是 black
  dorsum module reflector，不作为第 21 joint。
- 左 #1 logical track 为 `11781 → 12503`；只允许最多四个 CMM 帧的内部短
  gap fill，禁止跨长缺口或首尾外推。
- Virtual wrist 固定为 `p11 + 20 mm * normalize(p11 -
  mean(p4,p6,p8,p10))`；20 mm 是逐帧 hand-local 后移，不是 world-axis offset。
- Take_007 source RGB 为 1982 帧，正式映射只覆盖 0..1980 共 1981 帧；最后
  frame 1981 未锚定并排除。
- Take_007 solved pose 的十点 residual 是 same-take fit consistency，不进入
  旧 Take 01/02/03 的 held-out accuracy 表。
- 手工 XYZ profile 必须使用 `gt_calib.manual_xyz_profile.v1`，并严格声明
  `source_recording=camera_glove_recording_20260831_161912`、
  `source_take=Take_007`、
  `coordinate_system=mocap_world_mm`、`units=mm`；`rear_offset_mm` 固定为 20，
  不能被人工 profile 改写。
- 使用 profile 重建时，原始 JSON 必须复制为
  `calibration-workbench/applied_manual_profile.json`，manifest 记录 SHA-256，
  validation 复算；operator adjustment 不是 independent accuracy evidence。

## 当前关节对应

- 定量评估排除拇指，因为 MOCAP 21 点与 glove 20 点拓扑不一致。
- 固定配准只使用 index/middle/ring/pinky MCP。
- fingertip 不参与配准，作为 holdout 观察。
- 指标统一使用 `mocap-wrist root-normalized` 坐标；absolute wrist/6DoF 指标为 unavailable。

## 质量门限

- 默认拒绝跨越超过 25 ms 源样本 bracket 的插值。
- 25 ms 是严格审阅门限，不是设备 nearest-sample age。
- `Human.cma` 当前没有逐帧 confidence、visibility、occlusion、gap-fill 或 residual 字段，因此只能做有限数、连续性和运动学 sanity check。

## datalist.csv 字段

- `role`：`calibration / evaluation / reference`
- `split`：`calib / validation / test / na`
- `status`：`raw / processed / verified / rejected / superseded`
- `source_path`：仓库相对路径；原始数据不得改写。
- `sha256`：单文件填实际值；目录可填 manifest 路径并在 notes 说明。
