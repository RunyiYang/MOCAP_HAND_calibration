# Changelog

## 2026-08-31

### Final Take_007 revision

- 最终九视频的新增动作从 `161610 / Take_006` 改为 `161912 / Take_007`；
  video 07/08 现在各为 1981 帧，source RGB frame 1981 因无 reconstructed
  camera/CMM anchor 明确排除。
- 按用户提供的 marker 表和实物照片固化每手 #1..#10 tip/base 对应；左 thumb
  tip logical track 支持 `11781 → 12503` re-ID。#11 只构造 virtual wrist，不再
  画作第 21 joint。
- Virtual wrist 改为 `p11 + 20 mm * normalize(p11 -
  mean(p4,p6,p8,p10))`，实现逐帧 hand-local 后移 2 cm，避免误用固定世界轴
  offset 或把 thumb-tip marker 当 root。
- Take_007 solved pose 现在用全部十个照片对应做 per-frame proper Kabsch，尺度
  按侧跨有效帧冻结；联合有效覆盖为 1367/1981（69.01%）。左/右十点
  same-take residual median 为 21.02/22.11 mm、P95 为 52.37/54.09 mm；该
  residual 不是 held-out 或 independent GT accuracy。
- Camera→CMM 的 980 个 anchor absolute median/P95/max 为
  2.238/3.988/4.577 ms，980/980 在 20 ms 内。上游总 `pass=false` 来自
  glove-to-CMAvatar 的 48 个 timing outlier，不再被错误解释成 Take_007 的
  camera/CMM 对齐失败。
- Video 09 强制绑定 `camera_glove_recording_20260831_155410` 的 98 帧无手套
  RGB、该 recording 自己的 intrinsics 与 CS-400 world calibration；不再复用
  错误的动作 recording calibration provenance。
- Video 09 新增 reference-image hard gate：archive `reference_rgb.png` 必须与
  155410 RGB frame 30 decoded BGR 逐像素一致，否则 build 失败；当前状态
  `pixel_identical`，并在 metrics 写入 `calibration_reference_identity`、PNG
  SHA-256 `e4a31ff9…3388` 与 decoded-BGR SHA-256 `cabc821f…bf9e`。
- 最终网页新增 Take_007 manual XYZ workbench，可叠加 clean RGB、CMM 和 solved
  trajectory，控制 global/left/right world-mm offset，并导出
  `gt_calib.manual_xyz_profile.v1`。`render-new` 与 `build` 新增
  `--manual-profile` 复跑入口；人工结果明确是 display calibration。
- Manual profile loader 现在对 `source_recording`、`source_take=Take_007`、
  `coordinate_system=mocap_world_mm`、`units=mm` 和三个 finite XYZ vector 做
  fail-closed 校验，并固定 `rear_offset_mm=20`。应用的 JSON 会原样复制到
  package、写入 SHA-256 并纳入 delivery validation；默认包显式记录 `null`。
- Take_007 正式九视频包已验证：40 files / 118.38 MiB，9/9 H.264 full decode，
  manifest/validation pass、`failures=[]`，39 项 SHA-256，全量测试 114 项
  通过（记录基线 `114 passed in 99.08s`）。
- 新增 `TAKE007_CMM_ALIGNMENT_V1.md`，并修订数据集说明、日报、根 README 与
  文档入口；旧 Take_006 选择保留为 superseded 历史，不再作为最终交付口径。

### Fixed

- 修复相机—MOCAP 时间语义：旧实现把 `color_device_timestamp_us` 直接映射到约 254–255 ms 之后的 Thor poll/callback 时刻，导致 MOCAP 动作在视频上提前约 7–8 帧。
- 现在逐行计算 `capture_to_poll = thor_realtime - color_device_timestamp`，并从 `thor_monotonic` 与 `cmavatar_windows_estimated` 中减去该延迟，恢复相机采集目标时刻；未加入 per-take 人工偏移。
- `Human.cma.PtpTimeStamp` 现在提供高分辨率相对节拍，并对毫秒级 Windows `TimeStamp` 做 affine anchor；绝对 PTP epoch 尚未交付，指标中明确保留这一边界。
- 修复 MP4↔BAG 身份验证只检查帧数/时间戳、不检查图像内容的证据缺口；正式发布现在用 `--frame-content-samples 0` 对全部 frame/message index 在 ±2 帧窗口中做 fail-closed 内容同帧搜索。默认 31 帧保留为快速回归。
- Raw-depth overlay 现在逐帧使用 depth message 自己的 `timestamp_usec` 查询 clock map 和 MOCAP，不再用对应 RGB timestamp 代替 depth acquisition time；01 多出的尾部 depth frame 保持 unmatched。

### Added

- 新增 rear-mount 显示开关 `--draw-rear-mount-proxy`：每手增加 `wrist - 0.8*(middleMCP-wrist)` 带白色中心的圆点与连线，原 21 点不修改，固定标注 `VIZ-only, not GT`。
- 新增显式诊断参数 `--mocap-world-z-offset-mm`；默认 0，非零值会写入 metrics 并强制标成 candidate，而不是静默改外参。
- 新增 `mocap_world_origin_audit.py` 与 24 帧 raw/rear-proxy/+Z37 三列 held-out RGB 总览；Take02 output 661 明确排除，source/CMA bracket/alpha 与正式 CSV 逐项断言一致。
- 新增 `mocap_wrist_depth_audit.py`：在三段各 301 个 raw-depth 帧上扫描 world Z 并单独评估 wrist、MCP 与 rear proxy，输出 `gt_calib.mocap_wrist_depth_audit.v1` JSON。
- 新增 `MOCAP_WRIST_SEMANTICS_V1.md`、三段 rear-module-aware H.264 与 `ev-20260831-026`–`033` 证据。
- 新增 `mocap_video_overlay.py`：生成 MOCAP-only 曝光时刻叠加、逐帧 alignment CSV、metrics、contact sheet 与 H.264 review。
- 新增 `depth_calibration_audit.py` 与 `DEPTH_CAMERA_POSE_AUDIT_V1.md`：直接读取 raw `mono16` 毫米深度，复算固定 depth-camera SE(3)，做 00 连续时间块重复性、人工点选敏感度、全残差、跨 Take 桌面与静态 RGB 背景检查，并单独输出未启用的 D2C rigid candidate。
- 新增 `depth_mocap_overlay.py` 与 `gt_calib.depth_mocap_overlay.v1` metrics：顺序解码 01/02/03 raw `mono16` 深度，以 depth 自身设备时间映射并插值 120 Hz MOCAP，使用 00 冻结的 `world_to_depth_camera` SE(3) 投影，输出 MP4V、H.264、contact sheet、逐帧 alignment CSV 和 metrics；不做逐 Take 空间 refit。
- 新增 `DEPTH_MOCAP_ALIGNMENT_V1.md`：固化 raw-depth frame/time/pose contract、1747/1804/1799 有效区间、逐帧 alignment CSV schema、复现/验收命令和 nonzero-coverage 非 accuracy 边界。
- 新增视觉动作相位 QA：固定 ROI 视频运动能量对投影 42 关节速度；残余峰值只验收，不回写到时钟模型。
- 新增 `MOCAP_VIDEO_ALIGNMENT_V1.md` 和 2026-08-31 日报；网页新增默认 `MOCAP action match` 模式、逐帧 CSV 下载和同步指标表。
- MOCAP 视频 metrics schema 升级为 `gt_calib.mocap_video_alignment.v2`，新增内容身份明细、可辨性、阈值和 acceptance；root-fusion profile/metrics 当前 schema 同步为 v2。
- 数据索取清单加入共享 TTL/同步 LED 原始事件、采集 logger、相机曝光语义、PTP grandmaster/lock 与官方 Orbbec depth→color 外参。
- 日报、README 和飞书草稿将明日最高优先级整理成九个可验收 P0 包：raw Marker、手部资产、Human quality、相机—MOCAP 对应、Femto Bolt 刚体、官方 D2C、硬件时间、Body Hand mount、glove replay。

### Results

- 全局 `+Z37 mm` 虽能改变二维覆盖，但 903 帧 raw-depth 上 wrist absolute residual median 从 12.94 mm 恶化到 67.00 mm，wrist+4 MCP 从 24.58 mm 恶化到 64.04 mm，因此不启用。
- hand-local rear proxy 的 absolute depth residual median 为 15.26 mm，接近原 wrist 12.94 mm，且原 joints 的 world/pixel 修改量均严格为 0。
- rear-proxy H.264 为 1748/1805/1800 帧；三份 timing CSV 与原版本 byte-identical。
- Take 01/02/03 的发布共同区间为 1748/1805/1800 帧，逐帧 MOCAP coverage 均为 100%。
- 旧 poll-time zero-lag correlation 为 0.6261/0.4876/0.3947；曝光修正后为 0.8761/0.8987/0.8819。
- 修正后的最佳视觉残余为 +16/+10/+17 ms，均小于一帧；三段该残余均未写回 profile。
- capture→poll median 为 255.103/255.250/254.008 ms；P95 为 270.295/270.358/269.719 ms。
- depth pose 109 帧复算与 delivered JSON 数值一致；00 连续三块最大 rotation/origin delta 为 0.057°/0.474 mm。00→01/02/03 静态 RGB 背景最大配准平移为 0.064/0.067/0.044 px，支持 fixed-view 假设。
- raw-depth 桌面跨 Take 相对 00 出现 1.035–1.160° 与 7.39–8.75 mm 的 ROI-sensitive 差异，保留为 depth-systematic review，不作为 per-Take pose correction。
- D2C forward 3×3 的 det 为 0.994914、最大正交误差为 0.010146；最近 SO(3) 在工作空间造成 P95 2.973 px 投影差，因此未静默替换。
- Raw-depth MOCAP 三段完整同步共同区间为 1747/1804/1799 帧；对应 depth source frame 为 60–1806、0–1803、3–1801。depth−RGB device timestamp median 为 1.3108/1.3113/1.3103 ms，且每帧直接使用 depth timestamp。
- 冻结 depth SE(3) 后，三段 projected joint 都保持正深度且在 640×576 frame 内；projected-pixel raw-depth nonzero coverage 为 joint 0.7643/0.5744/0.6455、wrist 0.6056/0.5690/0.6573。该覆盖率不是 skeleton-to-surface error 或 accuracy；当前没有 independent depth joint labels。
- corrected root-fusion Take 02 左/右 joint median 为 11.49/11.39 mm，Take 03 为 12.94/13.19 mm；holdout P95 长尾仍为 joint 86.06–97.44 mm、tip 104.87–113.80 mm。

### Validation

- 三段 rear-proxy H.264 已由 ffmpeg 完整解码，均 exit 0；BAG↔MP4 full-frame gate 仍为 1815/1812/1810，metrics 明确记录 proxy enabled、global translation disabled。
- RGB wrist-semantics 审计的 24/24 source/CMA mapping、24/24 exact source frame 与几何不变量全部通过；raw-depth 审计覆盖 903 帧、20 个 Z candidates。
- 最终全量单元/数据契约测试 `90/90` 通过（89.468 s，含腕端语义/depth 审计、BVH 导出、本地网页契约和 Range 服务）。Depth H.264 已完整解码 1747/1804/1799 帧，CSV data rows 与 audit/metrics/CSV/H.264 hashes 均已复核。
- 三段 MOCAP-only H.264 全帧解码为 1748/1805/1800 帧；corrected fusion 与 diagnostic review 均为 1815/1812/1810 帧并通过全片解码与 faststart 检查。
- 三段 timecode↔BAG timestamp 精确匹配为 875/875、878/878、876/876；逐帧 CSV 连续无断点。
- 全帧 BAG↔MP4 内容 gate：raw best-offset=0 为 1764/1815、1768/1812、1809/1810，对应 0.9719008264/0.9757174393/0.9994475138；可辨帧为 1715/1715、1729/1729、1794/1794 且全部 offset=0，三段 global best offset 均为 0。
- 全帧 MSE max 为 4.7364506721/4.6256175041/4.4396605492，PSNR min 为 41.3762734217/41.4791064309/41.6573059509 dB，dHash max 均为 3；三段 content/timecode/temporal acceptance 全部 PASS。
- 内容证据逐帧比较 240×135 解码灰度与 dHash，不是原始编码 bit-exact，也不验证空间精度或硬件零时差。
- 旧 poll-time 产物保留但在 `outputs/SUPERSEDED_POLL_TIME.md` 中标记为不可用于当前同步结论。

## 2026-08-30

### Added

- 新增 `docs/` 日常交付入口、日报模板、数据清单、证据索引和数据索取清单。
- 新增 registration profile 的保存与复用，记录输入 SHA256、solver/config 和 session-neutral provenance。
- 新增 01 calibration → 02/03 frozen holdout 协议。
- 新增 25 ms 源 bracket 插值缺口门限。
- 新增 pooled joint EPE、frame MPJPE、MCP/intermediate/tip 分项和覆盖率。
- 新增 `mocap-root-fusion` pose mode：解析 `Human.cma` wrist `GlobalQ_R/X/Y/Z` 为 wxyz，使用 sign-continuous SLERP 和 active wrist-local→MOCAP-world rotation。
- 新增 MOCAP Root Fusion v1 profile：只在 Take 01 wrist-local 非拇指 MCP 上拟合 canonical rotation + uniform scale，冻结到 Take 02/03。
- 新增三段 root-fusion H.264 review、严格 metrics 与 [`fusion_comparison_overview.svg`](../outputs/mocap_root_fusion_strict25/fusion_comparison_overview.svg)。
- 新增 supervised edge articulation 的 research-only 审计；记录其 128 DoF、target/tip fit、不可辨识 edge、thumb 缺口和不予产品化决定。

### Changed

- 指标 schema 从 `gt_calib.glove_mocap_rgb_preview.v1` 升级为 `v2`。
- 指标明确命名为 `root_normalized`；absolute wrist/6DoF 显式为 unavailable。
- 当前推荐 calibrated result 改为 **MOCAP Root Fusion v1**；旧 world-wrist 模式降级为 **diagnostic baseline**，不再代表当前推荐结果。
- 当日初版将 Root profile/metrics 分拆为 `gt_calib.mocap_root_conditioned_articulation_profile.v1` 与 `..._metrics.v1`，并写入不同 `artifact_type`；该历史版本已由 2026-08-31 的 v2 schema supersede，旧 profile 仅保留 loader 兼容。
- Root profile loader 对 pose mode、joint contract、root conditioning、quaternion semantics、左右手、数值范围和 rotation aliases 做 fail-closed 校验。

### Results

- 严格 25 ms 下，Take 02 holdout 左/右 joint median 为 11.58/11.48 mm，tip median 为 18.33/19.06 mm；joint P95 为 86.83/88.43 mm，tip P95 为 104.92/107.58 mm。
- 严格 25 ms 下，Take 03 holdout 左/右 joint median 为 12.86/13.10 mm，tip median 为 23.18/26.17 mm；joint P95 为 95.64/97.77 mm，tip P95 为 113.95/109.96 mm。
- Median 比 diagnostic baseline 明显下降，但 holdout P95 仍为 joint 86.83–97.77 mm、tip 104.92–113.95 mm；不把该改善解释成独立 glove wrist/absolute 6DoF accuracy。

### Validation

- 单元/数据契约测试扩展为 **15/15 passed**，覆盖 wxyz active rotation、sign-continuous SLERP、synthetic root fusion、robust outlier、profile freeze/portability、legacy compatibility 和语义篡改拒绝。
- 三段正式 H.264 均通过全帧解码：1815/1812/1810 帧，960×540、30 FPS、High/yuv420p。
- comparison SVG 通过 `xmllint`、浏览器渲染及 metrics 反向核数；12 组 median 和 12 个 fusion P95 均精确到 0.01 mm。

### Data

- 固定当前 split：Take_000 calibration、Take_001 validation、Take_002 test/stress。
- `docs/evidence/index.csv` 追加 `ev-20260830-007` 至 `ev-20260830-015`，覆盖 fusion profile、三份 strict metrics、三段 H.264、comparison SVG 和 Take 03 的 50 ms 灵敏度；旧证据行保留不变。
