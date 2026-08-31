# MOCAP × RGB / Raw Depth：采集时刻逐帧匹配（2026-08-31）

## 一句话结论

三段视频中的 MOCAP 动作已经按相机真实采集时刻逐帧重做。旧链路误用了约 254–255 ms 之后的 host poll/callback 时刻，因此骨架会领先画面约 7–8 帧；修正后，三段视频运动与 MOCAP 的 zero-lag correlation 从 0.395–0.626 提升到 0.876–0.899。现在同一条修正后的时钟链还直接覆盖 raw `mono16` 毫米深度：depth 使用自己的采集时间戳，固定 depth-camera SE(3) 不做逐 Take 重拟合。

## 这次具体修了什么

同步表每一行现在按下面的物理时间语义处理：

```text
capture_to_poll = thor_realtime_ns - color_device_timestamp_us * 1000
thor_acquisition = thor_monotonic_ns - capture_to_poll
cmavatar_acquisition = cmavatar_windows_estimated_ns - capture_to_poll
```

随后使用 `Human.cma.PtpTimeStamp` 的高分辨率相对节拍，在相邻 120 Hz `FrameCounter` 之间把 MOCAP 位置插值到每个 RGB acquisition timestamp。没有逐 Take 调一个“看起来更贴”的时间 offset。

## 三段动作匹配结果

| Take | 发布 source frames | 发布帧 | capture→poll median/P95 | zero corr 旧→新 | 修正后最佳残余 |
|---|---:|---:|---:|---:|---:|
| 01 | 60–1807 | 1748 | 255.103 / 270.295 ms | 0.626 → 0.876 | +16 ms |
| 02 | 0–1804 | 1805 | 255.250 / 270.358 ms | 0.488 → 0.899 | +10 ms |
| 03 | 3–1802 | 1800 | 254.008 / 269.719 ms | 0.395 → 0.882 | +17 ms |

三段发布片段内部均无缺口，每个 RGB 帧都有明确的 MOCAP low/high `FrameCounter`、目标时刻和插值 alpha；逐帧映射已保存为 CSV。+10/+16/+17 ms 相关峰残余均小于一个 30 FPS 帧，只用于 QA，**没有回写**。

同时增加了 BAG↔MP4 全帧内容级身份 gate，避免只凭“帧数相等”假设同一 index 就是同一画面。正式运行使用 `--frame-content-samples 0`，对每个 BAG color message 在 source MP4 的 ±2 帧窗口中搜索，并比较 240×135 解码灰度图与 dHash；结果如下：

| Take | raw best-offset=0（比例） | 可辨帧 best-offset=0 | MSE max | PSNR min | dHash max |
|---|---:|---:|---:|---:|---:|
| 01 | 1764/1815（0.9719008264） | 1715/1715 | 4.7364506721 | 41.3762734217 dB | 3 |
| 02 | 1768/1812（0.9757174393） | 1729/1729 | 4.6256175041 | 41.4791064309 dB | 3 |
| 03 | 1809/1810（0.9994475138） | 1794/1794 | 4.4396605492 | 41.6573059509 dB | 3 |

三段 global best offset 都是 0；所有可辨帧均为 offset 0，raw 中少量非零候选来自静止或近重复画面。门限为 MSE max≤6、PSNR min≥39 dB、dHash max≤4、可辨帧≥24 且全为 offset 0、raw offset-0 fraction≥0.90，三段 content/timecode/temporal acceptance 均通过。默认 31 帧模式仅用于快速回归；正式发布要求 `0=全帧`。

这里的“全帧内容证据”是每个 index 都做 240×135 解码灰度+dHash 比较，**不是原始编码 bit-exact**，也不证明空间投影精度或硬件零时差。

正式复跑命令：

```bash
/home/runyi/miniconda3/envs/viewer/bin/python mocap_video_overlay.py \
  --segment 01 --segment 02 --segment 03 \
  --output-dir outputs/mocap_video_alignment_review \
  --frame-content-samples 0 \
  --output-width 960
```

## Raw-depth × MOCAP 新交付

这次不是把彩色深度预览视频当作数据，而是直接顺序解码每段 `camera_1_rgb_depth.bag` 中 `/cam/sensor_3/frameType_3` 的 640×576 little-endian `mono16` 毫米深度。每一帧使用 depth message 自己的 `timestamp_usec` 查询 exposure-corrected clock map，再插值 120 Hz MOCAP；没有用对应 RGB timestamp 替换它。

| Take | raw depth / RGB messages | 发布 depth source frames | 发布帧 | depth−RGB median/P95 | nearest MOCAP P95 |
|---|---:|---:|---:|---:|---:|
| 01 | 1816 / 1815 | 60–1806 | 1747 | 1.3108 / 1.3173 ms | 3.9593 ms |
| 02 | 1812 / 1812 | 0–1803 | 1804 | 1.3113 / 1.3181 ms | 3.9579 ms |
| 03 | 1810 / 1810 | 3–1801 | 1799 | 1.3103 / 1.3170 ms | 3.9546 ms |

01 多出的最后一帧 depth 被明确保留为 unmatched tail，没有强制配给 RGB。三段发布帧全部有合法 MOCAP interpolation，最大 bracket 为 8.333922 ms；空间侧始终使用 00 冻结的 `world_to_depth_camera`，`det=1`、最大正交误差 `2.22e-16`，没有从 01/02/03 的手动作重新拟合外参。

正式程序还直接读取 BAG `/cam/streamProfileType_3`，逐项核对序列号 `CL8L563010D`、640×576、30 FPS、depth intrinsics 和 8 项 distortion 与每段内参 JSON/00 标定记录；三段全部 gates 通过。产物在同目录 staging 完成并哈希后原子替换，metrics 最后写入作为 commit marker；`--max-frames N` 使用 `_smokeN` stem，不会覆盖正式全片。

正式复跑命令：

```bash
/home/runyi/miniconda3/envs/viewer/bin/python depth_mocap_overlay.py \
  --segment 01 --segment 02 --segment 03 \
  --max-interpolation-gap-ms 25 \
  --min-depth-mm 350 --max-depth-mm 1800 \
  --snapshot-count 6 \
  --output-dir outputs/depth_mocap_alignment_review
```

建议在飞书为每个 Take 放一段 H.264 和一张 contact sheet：

```text
outputs/depth_mocap_alignment_review/
  01_..._depth_mocap_aligned_strict25_h264.mp4
  02_..._depth_mocap_aligned_strict25_h264.mp4
  03_..._depth_mocap_aligned_strict25_h264.mp4
  01/02/03_....contact.jpg
  01/02/03_....alignment.csv
  01/02/03_....metrics.json
```

`gt_calib.depth_mocap_overlay.v1` 还报告 projected joint/wrist 是否为正相机深度、是否落在 640×576 内、对应 raw depth pixel 是否非零。三段 joint nonzero coverage 为 0.7643/0.5744/0.6455，wrist 为 0.6056/0.5690/0.6573。**这些比例只验证 timing、projection 和传感器覆盖，不是 skeleton-to-surface error 或 joint accuracy**：MOCAP 点位于手内部，深度看到首表面，而且目前没有 independent depth joint labels/visibility。

逐帧 alignment CSV 的全部字段、frame contract、验收命令和 GT 边界已固化在 `docs/calibration/DEPTH_MOCAP_ALIGNMENT_V1.md`。

## 建议在飞书放的附件

每个 Take 放一段 H.264 视频和一张 contact sheet：

```text
outputs/mocap_video_alignment_review/
  01_..._mocap_video_aligned_strict25_h264.mp4
  02_..._mocap_video_aligned_strict25_h264.mp4
  03_..._mocap_video_aligned_strict25_h264.mp4
  01/02/03_....contact.jpg
  01/02/03_....alignment.csv
  01/02/03_....metrics.json
```

视频图例：洋红 = 左手 MOCAP 21 点；青色 = 右手 MOCAP 21 点；帧头显示 source RGB frame、匹配时间、MOCAP bracket、alpha 和 nearest sample delta。

网页已提供四个切换视图：

1. `MOCAP action match`：默认，先审动作相位；
2. `Metric depth projection`：raw 毫米深度 + 同时刻 MOCAP，检查时间、固定 SE(3) 和覆盖率；
3. `Calibrated fusion`：MOCAP wrist SE(3) + glove-local articulation；
4. `Wrist diagnostic`：保留 glove wrist orientation 的旧诊断基线。

Cloudflare/Wrangler dry-run 已通过，网站共 78 个文件、148.43 MiB，最大发布视频 14,241,449 bytes；本地审阅地址为 `http://127.0.0.1:8810/?mode=depth`。当前 Wrangler 未登录，且页面含可识别人物和实验室画面，因此没有创建匿名公网 URL；正式 URL 应先登录、绑定受控域名并加 Cloudflare Access，再贴到飞书。

## 当前 calibrated fusion

Take 01 只拟合每只手一个 wrist-local canonical rotation + uniform scale，随后冻结到 Take 02/03。严格 25 ms gate 下 non-thumb joint / fingertip median：

| Take | 左手 joint / tip | 右手 joint / tip |
|---|---:|---:|
| 01 calibration | 13.44 / 22.29 mm | 11.94 / 19.42 mm |
| 02 validation holdout | 11.49 / 18.05 mm | 11.39 / 18.88 mm |
| 03 test/stress holdout | 12.94 / 23.27 mm | 13.19 / 26.38 mm |

这些数字衡量的是 **MOCAP wrist SE(3)-conditioned finger articulation consistency**，不是手套独立 wrist/absolute 6DoF 精度。Take 02/03 的 P95 长尾仍为 joint 86.06–97.44 mm、tip 104.87–113.80 mm，不能只展示 median。

## Depth calibration 与 camera pose 新结论

数据里确实有一份 camera pose，但它是固定外参，不是逐帧 trajectory。四段 BAG 没有 `/tf`、odometry 或 pose/trajectory，IMU 也不能提供绝对相机 pose。当前 `T_world_from_depth` 的旋转是严格 SE(3)，深度相机中心在 world 中为 `[-55.047, 845.201, 547.291] mm`。用 00 的 109 帧 raw `mono16` 毫米深度重新计算，可以数值复现 delivered JSON；把 00 按连续时间分成三块后，最大 rotation/origin delta 为 `0.057° / 0.474 mm`。

我们还第一次真正检查了“相机没动”而不是只比 serial 和内参：00 与 01/02/03 在 10%、50%、90% 三个时点的静态 RGB 背景配准，最大平移分别只有 `0.064 / 0.067 / 0.044 px`，最大旋转约 `0.0017 / 0.0022 / 0.0021°`，支持 fixed-view 假设。

但仍有两个必须保留的 review：

1. raw-depth 桌面在正式 Take 相对 00 出现约 `1.04–1.16° / 7.4–8.7 mm` 的 ROI-sensitive 差异。结合 RGB 背景不动，它更像 ToF 空间偏差、局部桌面和遮挡中值的混合信号，不能直接当作 per-Take camera pose correction。
2. delivered depth→color 3×3 的 `det=0.994914`、最大正交误差 `0.010146`，不是旋转矩阵。最近 SO(3) 候选会带来 P95 `2.973 px` 的投影变化，因此只输出 candidate，未替换当前 projection。

建议飞书放两张图：

```text
outputs/depth_calibration_audit/depth_camera_pose_audit_summary.png
outputs/depth_calibration_audit/rgb_fixed_camera_registration.jpg
```

完整机器可读结果是 `outputs/depth_calibration_audit/depth_camera_pose_audit.json`，解释文档是 `docs/calibration/DEPTH_CAMERA_POSE_AUDIT_V1.md`。

## 当前 GT 边界

现在可以确认：

- 裁剪后的每个发布 RGB 帧都有可审计的曝光时刻 MOCAP bracket；
- BAG↔MP4 全帧内容 gate 已通过，三段所有可辨帧全部 best-offset=0；
- 时间错误根因是 capture→poll latency，而不是需要每段调一个 magic offset；
- 投影链、坐标方向和动作相位已可用于 review。

现在还不能宣称：

- 没有 shared exposure/OptiTrack TTL，不能说硬件级零时间误差；
- 没有独立逐帧 2D 手关节标签，不能说完整 21 点像素精度；
- `Human.cma` 没有 confidence/visibility/occlusion/gap-fill/residual，不能把全部 solved joints 当成可审计实体 Marker GT；
- `PtpTimeStamp` 的 absolute epoch/Grandmaster 契约未交付，目前只使用其稳定的相对 cadence；
- 当前 depth→color 线性块轻微非正交，需要官方 Orbbec 原始外参解释。
- Raw-depth overlay 的 nonzero coverage 不是骨架到手表面的距离；当前 metrics 明确为 `skeleton_to_surface_error_measured=false`、`independent_depth_joint_labels_available=false`。
- 固定 SE(3) 不是逐帧 camera trajectory；没有相机刚体 raw MOCAP trajectory 时，不能独立验证每一时刻的 camera pose。

## 明天最优先向数据方要

1. 三个 TAK 的 raw labeled/unlabeled Marker C3D/CSV，带 residual、camera mask、occlusion、gap-fill；
2. 手部 Marker Set、21 点骨架层级、Human solver/filter/gap-fill 配置和安装照片；
3. 三段 Human joint quality 表：valid/confidence/source marker/occlusion/gap-fill/residual；若软件不支持请书面确认；
4. 2026-08-29 同一布置的相机—MOCAP 原始标定、坐标系契约、held-out RGB/depth↔MOCAP correspondences，以及 Orbbec 官方 intrinsics、畸变与 depth→color 刚体 `R,t`；
5. Body Hand rigid-body definition、逐帧质量、安装照片与到 glove root 的固定变换；
6. timestamp contract、同步 logger 源码/commit、PTP grandmaster/lock；下一次采集加入 shared TTL，接口不允许时用 RGB 可见且 MOCAP 可追踪的同步 LED，并保留原始 trigger log；
7. 三段实际 session-neutral、solver assets、带 device timestamp 的原始 IMU 和端到端可重放命令；
8. Femto Bolt 相机刚体的 raw MOCAP marker trajectory、rigid-body definition、质量表、安装照片和 marker→depth/RGB optical frame 固定变换；
9. 如果历史共同标定不完整，新采 RGB-D 与 MOCAP 共同可见的刚性标定棒/板：不少于 30 个完整 pose，覆盖中心/边缘和近/中/远，整 pose 留出至少 20%，交付 target geometry、raw correspondences 和 split manifest。

完整可发送清单：`docs/dataset/DATA_REQUEST_2026-08-31.md` 与 `request_datalist_2026-08-31.csv`。

## 验收

- `gt_calib.mocap_video_alignment.v2` metrics 已记录内容身份、曝光时刻映射和 fail-closed acceptance；
- 最终全量单元与数据契约测试 `68/68` 通过（84.559 s，包含 BVH 导出、本地网页契约和 Range 服务器测试）；
- MOCAP-only H.264 全片解码 1748/1805/1800 帧；
- Raw-depth×MOCAP H.264 已完整解码 1747/1804/1799 帧，CSV rows 与 artifact hashes 均已复核；即使媒体验收通过，nonzero coverage 仍不能升级解释为 accuracy；
- fusion 与 diagnostic H.264 全片解码 1815/1812/1810 帧；
- RGB MOCAP/fusion/diagnostic H.264 为 960×540；raw-depth×MOCAP 为 640×576；全部为 30 FPS、High/yuv420p、faststart；
- 三段逐帧 CSV 连续，metrics 与交付 artifact SHA-256 已复核。
