# Raw-depth × MOCAP Alignment V1

- 日期：2026-08-31
- 状态：`verified_frame_complete_depth_projection_contract`
- Metrics schema：`gt_calib.depth_mocap_overlay.v1`
- Generator：`depth_mocap_overlay.py`
- 输出目录：`outputs/depth_mocap_alignment_review/`

## 1. 可宣称的结果

01、02、03 的 raw Orbbec `mono16` 毫米深度已经逐帧匹配到 CMAvatar/MOCAP：每个输出帧直接使用该 depth message 自己的设备采集时间戳，经 exposure-corrected camera clock mapping 得到 CMAvatar target time，再在相邻 120 Hz MOCAP samples 之间插值。空间变换只使用 00 交付并冻结的 `world_to_depth_camera` 严格 SE(3)，三段都没有根据手动作或 depth 画面重新拟合 pose。

三段完整同步共同区间分别为 1747、1804、1799 帧；逐帧 CSV、H.264、contact sheet 和 metrics 均已生成。这个交付验证：

- raw depth frame/message identity；
- depth acquisition timestamp 到 MOCAP 的查询与插值；
- 固定 world→raw-depth 投影链；
- projected joint/wrist 是否为正深度、在画面内、对应 raw pixel 是否非零。

它不验证 skeleton-to-surface error、逐关节 depth accuracy 或 absolute camera↔MOCAP GT。

## 2. Raw depth 与时间契约

每段输入为 `原始BAG与内参/camera_1_rgb_depth.bag`：

- Depth topic：`/cam/sensor_3/frameType_3`
- Encoding：little-endian `mono16`
- Resolution：640×576
- Units：millimetres
- Nominal rate：30 FPS

正式运行还直接解码 BAG 的 `/cam/streamProfileType_3`（`custom_msg/OBStreamProfileInfo`），并与每段 `camera_1_intrinsics.json` 及 00 标定记录逐项比较：设备序列号 `CL8L563010D`、640×576、30 FPS、`fx/fy/cx/cy`、8 项 Orbbec distortion 和 schema 任一不符即拒绝发布。三段 metrics 的 `recorded_camera_contract.pass` 与全部 gates 当前均为 `true`；这证明输入来自同一已登记 depth camera contract，但不能单凭 serial/内参证明相机物理位置没有移动。

全量消息核验显示，00/01/02/03 的 109/1816/1812/1810 个 depth messages 全部满足 BAG index timestamp、nested message record time 与 ROS Image header sec/nsec 在 ns 上一致；trailing `timestamp_usec` 与它们四舍五入到最近 µs 后一致。`timestamp_global_usec` 全部为 0，因此没有可宣称的 camera–MOCAP global hardware clock。

Depth 与同 index RGB 的时间差稳定在约 +1.31 ms，但这只是当前数据的经验同 clock-domain 证据，不是已交付的硬件同步契约。程序因此：

1. 直接读取每个 depth message 的 `timestamp_usec`；
2. 使用已修正 capture→poll latency 的 camera ClockMapping；
3. 不以 RGB timestamp、MP4 PTS 或 `frame/30` 代替；
4. 对 01 的额外尾部 depth frame 保持 unmatched，不强配到最后一帧 RGB。

01 的唯一额外 depth 是 source index 1815；它位于最后一个 RGB 之后，不是中间插帧。

## 3. 固定 camera pose

主空间变量定义为：

\[
p_D = T_{D\leftarrow W}p_W,
\]

其中 `W` 是当前 CS-400/MOCAP nominal world，`D` 是 raw-depth optical frame，`+X` 向图像右、`+Y` 向下、`+Z` 向相机前方，单位 mm。实际冻结矩阵来自：

```text
同步整理_20260829_三段/
  movementcap_worldcalib_pointcloud_package_20260830/
  movementcap_ruler_worldcalib/results/manual_final/camera_to_world.json
```

```text
T_D_from_W =
[[-0.9955187582, -0.0688948102, -0.0647758231,   38.8804839]
 [ 0.0265427160,  0.4538796694, -0.8906675754,  105.2963051]
 [ 0.0907628027, -0.8883956049, -0.4500170694, 1002.1591169]
 [ 0,             0,             0,               1          ]]
```

其旋转 `det=1`，最大正交误差 `2.22e-16`。Metrics 必须包含 `world_to_depth_camera_frozen_no_refit=true`。

这是固定外参，不是逐帧 Femto Bolt trajectory。四段 BAG 没有 `/tf`、odometry 或 pose/trajectory；IMU 只能辅助静止/重力方向检查，不能恢复绝对 yaw 或 translation。

## 4. 发布区间

区间采用 `[source_frame_first, source_frame_stop_exclusive)`；只有 camera clock map 内、MOCAP interpolation 合法且 bracket 不超过 25 ms 的连续完整区间才发布。

| Take | raw depth / RGB messages | depth source interval | 发布帧 | trim pre/suf | depth−RGB median/P95 | bracket max | nearest MOCAP P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 01 | 1816 / 1815 | `[60, 1807)` | 1747 | 60 / 9 | 1.3108 / 1.3173 ms | 8.333922 ms | 3.9593 ms |
| 02 | 1812 / 1812 | `[0, 1804)` | 1804 | 0 / 8 | 1.3113 / 1.3181 ms | 8.333922 ms | 3.9579 ms |
| 03 | 1810 / 1810 | `[3, 1802)` | 1799 | 3 / 8 | 1.3103 / 1.3170 ms | 8.333922 ms | 3.9546 ms |

所有发布帧都具有合法 interpolation；source frame、decoded frame number 和 device timestamp 严格递增。

## 5. 逐帧 alignment CSV

每段 `*.alignment.csv` 一行对应一个输出帧。Header 固定为：

```text
output_frame,source_depth_frame,bag_message_timestamp_ns,
depth_device_timestamp_us,depth_frame_number,source_bag_elapsed_s,
matched_clip_bag_elapsed_s,same_index_rgb_device_timestamp_us,
depth_minus_rgb_device_timestamp_ms,cmavatar_target_timestamp_ns,
mocap_low_frame_counter,mocap_high_frame_counter,
mocap_low_timestamp_ns,mocap_high_timestamp_ns,interpolation_alpha,
bracket_span_ms,nearest_mocap_sample_delta_ms,raw_valid_pixel_fraction,
projected_joint_inside_count,projected_joint_nonzero_depth_count,
left_wrist_depth_x,left_wrist_depth_y,left_wrist_inside_frame,
right_wrist_depth_x,right_wrist_depth_y,right_wrist_inside_frame
```

字段分为四组：

- Identity：`output_frame`、`source_depth_frame`、BAG/message timestamps、depth frame number；
- Clock：depth/RGB timestamp 差、CMAvatar target、MOCAP low/high frame/timestamp、alpha、bracket、nearest delta；
- Raw-depth coverage：整帧有效像素率、投影 joint inside/nonzero counts；
- Wrist projection：左右 wrist 在 raw-depth image 中的 `(x,y)` 与 inside flag。

CSV 行数不含 header 为 1747/1804/1799；`wc -l` 应分别得到 1748/1805/1800。

## 6. 产物契约

每段 stem 为：

```text
01_210814_Take_000_depth_mocap_aligned_strict25
02_210955_Take_001_depth_mocap_aligned_strict25
03_211139_Take_002_depth_mocap_aligned_strict25
```

每个 stem 必须生成：

```text
{stem}.mp4
{stem}_h264.mp4
{stem}.contact.jpg
{stem}.alignment.csv
{stem}.metrics.json
```

H.264 当前为 High profile、640×576、30 FPS、`yuv420p`。Metrics 的 `artifacts` 必须记录 MP4V、H.264、contact 和 frame-map CSV 的 path、bytes、SHA-256。

发布采用同目录 staging：先完成并哈希视频、contact 与 CSV，再以原子 rename 替换正式文件，最后写入 metrics 作为 commit marker；中断时不会把半成品标成正式结果。带 `--max-frames N` 的快速运行使用 `_smokeN` stem，不会覆盖上述正式 stem。

## 7. 复现

```bash
cd /home/runyi/Project/hands_reloc/GT_calib

/home/runyi/miniconda3/envs/viewer/bin/python depth_mocap_overlay.py \
  --segment 01 --segment 02 --segment 03 \
  --max-interpolation-gap-ms 25 \
  --min-depth-mm 350 --max-depth-mm 1800 \
  --snapshot-count 6 \
  --output-dir outputs/depth_mocap_alignment_review
```

350–1800 mm 只是固定 review color scale；raw validity 和 timestamp 统计始终来自原始 uint16 值。

## 8. 验收

单元与数据契约测试：

```bash
/home/runyi/miniconda3/envs/viewer/bin/python -m unittest discover -s tests -v
```

最新全量测试实测 `90/90` 通过（89.468 s），其中 depth overlay 专项仍为 `11/11`；新增覆盖包括腕端语义/depth 审计、BVH 导出、本地网页契约和 Range 服务。

逐帧 CSV：

```bash
wc -l outputs/depth_mocap_alignment_review/*.alignment.csv
```

预期含 header 为 1748/1805/1800 行。

H.264 frame count 和完整解码：

```bash
for video in outputs/depth_mocap_alignment_review/*_h264.mp4; do
  ffprobe -v error -count_frames -select_streams v:0 \
    -show_entries stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,nb_read_frames \
    -of compact=p=0:nk=1 "$video"
  ffmpeg -v error -i "$video" -map 0:v:0 -f null -
done
```

当前实测 decoded frames 为 1747/1804/1799，三段完整解码均 exit 0。最后用：

```bash
sha256sum \
  outputs/depth_calibration_audit/depth_camera_pose_audit.json \
  outputs/depth_mocap_alignment_review/*.metrics.json \
  outputs/depth_mocap_alignment_review/*.alignment.csv \
  outputs/depth_mocap_alignment_review/*_h264.mp4
```

核对 `docs/evidence/index.csv` 与 metrics 中的 artifact hashes。

## 9. Nonzero coverage 的解释边界

| Take | projected joint inside frame | projected joint nonzero depth | wrist inside frame | wrist nonzero depth |
|---|---:|---:|---:|---:|
| 01 | 1.0000 | 0.7643 | 1.0000 | 0.6056 |
| 02 | 1.0000 | 0.5744 | 1.0000 | 0.5690 |
| 03 | 1.0000 | 0.6455 | 1.0000 | 0.6573 |

这些数只说明投影点落在 raw-depth image 内以及对应 pixel 是否非零。它们受手套材料、ToF invalid holes、遮挡和骨架点位于可见表面之后影响。当前 metrics 必须保持：

```text
skeleton_to_surface_error_measured = false
independent_depth_joint_labels_available = false
```

因此不能从这些比例推导 mm joint error、surface distance、camera-pose accuracy 或 MOCAP GT accuracy。要升级为空间精度验证，至少需要独立 depth/RGB hand labels 或已知 body-root→glove surface geometry，并且不得用同一批手动作重新拟合 camera pose。

## 10. GT 边界

- 00 的 109 帧 raw depth 支持固定 depth SE(3) 的内部复现，但 00 没有同步 MOCAP marker/C3D 对应，不能独立证明 nominal CS-400 frame 与 Motive world 的绝对误差。
- 三段 raw-depth overlay 共用冻结外参，是 cross-modal projection evidence，不是逐帧 camera trajectory。
- Depth 与 RGB 的 +1.31 ms 稳定差支持当前数据中的经验配对，但 `timestamp_global_usec=0` 且没有 shared TTL，不能称硬件零时差。
- `Human.cma` 缺少 joint confidence、visibility、occlusion、gap-fill 和 residual；MOCAP solved joints 不能全部称为直接实体 Marker GT。
- 明日九项 P0 见 `docs/dataset/DATA_REQUEST_2026-08-31.md` 与 `request_datalist_2026-08-31.csv`。
