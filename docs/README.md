# GT Calib 文档入口

- 当前状态：`final_nine_video_delivery_in_progress + verified_new_world_projection`
- 最近更新：2026-08-31
- 当前数据版本：`20260829_take000-002 + thor_new4_take006 + 20260831_tabletop_worldcalib`
- 最新日报：[2026-08-31](daily/2026/2026-08-31.md)
- 数据契约：[dataset/README.md](dataset/README.md)
- 数据清单：[dataset/datalist.csv](dataset/datalist.csv)
- 明日索取清单：[DATA_REQUEST_2026-08-31.md](dataset/DATA_REQUEST_2026-08-31.md)
- 新采集与新世界坐标理解：[NEW_CAPTURE_2026-08-31.md](dataset/NEW_CAPTURE_2026-08-31.md)
- 评估协议：[CROSS_TAKE_EVALUATION.md](protocols/CROSS_TAKE_EVALUATION.md)
- 当前推荐 calibrated fusion：[MOCAP_ROOT_FUSION_V1.md](calibration/MOCAP_ROOT_FUSION_V1.md)
- 当前视频动作匹配协议：[MOCAP_VIDEO_ALIGNMENT_V1.md](calibration/MOCAP_VIDEO_ALIGNMENT_V1.md)
- 当前 wrist / 后侧模块显示协议：[MOCAP_WRIST_SEMANTICS_V1.md](calibration/MOCAP_WRIST_SEMANTICS_V1.md)
- 当前 depth camera pose 审计：[DEPTH_CAMERA_POSE_AUDIT_V1.md](calibration/DEPTH_CAMERA_POSE_AUDIT_V1.md)
- 当前 raw-depth×MOCAP 交付契约：[DEPTH_MOCAP_ALIGNMENT_V1.md](calibration/DEPTH_MOCAP_ALIGNMENT_V1.md)
- Skeleton_0/1 BVH 本地播放器：[BVH_LOCAL_VIEWER_V1.md](calibration/BVH_LOCAL_VIEWER_V1.md)
- 保留 wrist-orientation 诊断基线：[CALIBRATED_ONE_V1.md](calibration/CALIBRATED_ONE_V1.md)
- 被否决的监督式 articulation probe：[EXPERIMENTAL_ARTICULATION_PROBE.md](calibration/EXPERIMENTAL_ARTICULATION_PROBE.md)
- 证据索引：[evidence/index.csv](evidence/index.csv)
- 变更日志：[CHANGELOG.md](CHANGELOG.md)

## 当前一句话结论

当前已将 MOCAP 按 **RGB 曝光时刻**逐帧匹配到视频：旧链路误用了约 254–255 ms 之后的 host poll/callback 时刻；逐行移除该延迟后，三段 zero-lag motion correlation 从 0.395–0.626 提升到 0.876–0.899，残余峰值 +10/+16/+17 ms 均小于一帧且不回写为拟合偏移。正式 `gt_calib.mocap_video_alignment.v2` 还用 `--frame-content-samples 0` 对三段全部 BAG/MP4 frame index 完成 240×135 解码内容 gate。

空间侧已确认 delivered depth-camera transform 是严格 SE(3)，109 帧可复算，00 连续时间块最大漂移为 0.057°/0.474 mm；00→01/02/03 静态 RGB 背景最大平移仅 0.067 px，支持 fixed-view。color 侧 D2C 仍为非正交 affine，不能称为已验证 color-camera 6DoF pose；raw-depth 桌面跨 Take 的 ROI-sensitive 差异保留为 review。

现在还可直接把 MOCAP 投到 raw `mono16` 毫米深度：01/02/03 逐帧使用各自 depth message 的 `timestamp_usec`，不借用约 1.31 ms 更早的 RGB timestamp；固定 `world_to_depth_camera` SE(3) 从头到尾冻结，无 per-Take refit。三段完整同步区间为 1747/1804/1799 帧，产物位于 `outputs/depth_mocap_alignment_review/`。这是 timing、projection 和 raw-depth nonzero coverage 交付，不是 skeleton-to-surface accuracy。

Wrist 显示已经按硬件语义拆开：原 21 点保持不动，后方黑色腕部模块由带白色中心的 rear proxy 圆点单独表示。对 903 个 raw-depth 帧，全局 `+Z37 mm` 会把 wrist absolute residual median 从 12.94 mm 恶化到 67.00 mm，因此未启用；rear proxy 为 15.26 mm，并明确只作 `VIZ-only, not GT`。正式视频位于 `outputs/mocap_video_rear_mount_review/`。

## 当前推荐结果

严格 25 ms 下的 non-thumb joint / fingertip pooled EPE，表中均为 **median / P95**：

| Take | 角色 | 左 joint | 左 tip | 右 joint | 右 tip |
|---|---|---:|---:|---:|---:|
| 01 | calibration/self-fit | 13.44 / 83.53 mm | 22.29 / 102.43 mm | 11.94 / 85.83 mm | 19.42 / 104.74 mm |
| 02 | validation holdout | 11.49 / 86.06 mm | 18.05 / 104.87 mm | 11.39 / 88.08 mm | 18.88 / 107.28 mm |
| 03 | test/stress holdout | 12.94 / 95.44 mm | 23.27 / 113.80 mm | 13.19 / 97.44 mm | 26.38 / 110.23 mm |

Take 02/03 的 fusion P95 仍为 joint 86.06–97.44 mm、tip 104.87–113.80 mm。指标图见 [`fusion_comparison_overview.svg`](../outputs/mocap_root_fusion_strict25/fusion_comparison_overview.svg)。

旧 `glove wrist orientation + MOCAP wrist translation` world-wrist 结果仅保留为 **diagnostic baseline**，用于定位错误坐标组合、wrist orientation 和 session-neutral 影响；它不再是当前推荐 calibrated result。完整定义与旧数字见 [CALIBRATED_ONE_V1.md](calibration/CALIBRATED_ONE_V1.md)。

## 当前可宣称

- 01/02/03 发布的共同区间分别有 1748/1805/1800 个连续 RGB 帧；每帧都有小于 25 ms 的 120 Hz MOCAP bracket。
- 逐行移除 capture→poll 延迟后，视觉动作 zero-lag correlation 为 0.876/0.899/0.882；峰值残余均在一个 30 FPS 帧内。
- 曝光时刻到最近 MOCAP 样本 P95 约 3.95–3.96 ms；这是 120 Hz 采样量化距离，不是端到端同步精度。
- BAG↔MP4 全帧内容 gate 已通过：raw offset-0 为 1764/1815、1768/1812、1809/1810；所有可辨帧均为 offset 0（1715/1715、1729/1729、1794/1794），三段 global best offset 均为 0。
- MOCAP 21 点可以用固定外参投影到 RGB，适合做动作相位审阅可视化。
- 可在不修改原 21 点、相机外参或逐帧 timing CSV 的情况下，额外显示 hand-local 后侧腕模块 proxy。
- delivered depth camera pose 是严格 SE(3)，可用于 raw-depth optical frame 与 CS-400/MOCAP nominal world 之间的固定变换；00 内部重复性和跨 Take 静态 RGB 背景检查已通过。
- Raw-depth MOCAP overlay 直接使用 depth 自己的设备时间戳；01 的额外尾部 depth frame 不会被强配到 RGB，三段所有发布帧都有小于 25 ms 的合法 MOCAP bracket。
- 冻结 pose 后可报告 projected joint/wrist 的 positive-depth、inside-frame 与 raw-depth nonzero coverage，作为数据链和投影覆盖检查。
- 可用 Take 01 wrist-local MCP 拟合固定 canonical rotation 和统一尺度，并冻结到 Take 02/03。
- 在每帧使用 MOCAP wrist SE(3) 的条件下，可报告 glove local finger articulation 的 root-normalized median/P95 与覆盖率。
- 最终全量单元/数据契约测试 `90/90` 通过（89.468 s，含 15 项腕端语义/depth 审计、BVH 导出、本地网页契约和 Range 服务）；MOCAP-only、fusion、diagnostic 与新增 depth 媒体均已完成全帧解码。Depth H.264 为 1747/1804/1799 帧，CSV rows 和 evidence hashes 已复核。

## 当前不可宣称

- 不能称 MOCAP 21 个节点均为实体 Marker 的直接测量。
- 不能称当前投影为逐关节像素级绝对 GT。
- 不能把 delivered D2C affine 的 3×3 叫作 color-camera rotation，也不能把 polar rigid candidate 未经 held-out RGB-D 对应验证就投入使用。
- 不能把 00 的重复性解释为独立 absolute pose accuracy；00 没有同步的 MOCAP marker/C3D 对应来独立证明 nominal CS-400 frame 与 Motive world 的误差。
- 不能把固定 camera pose 解释为逐帧 Femto Bolt trajectory：四段 BAG 没有 `/tf`、odometry 或 pose/trajectory，IMU 也不给绝对 camera pose。
- 不能把 projected-pixel 的 nonzero depth coverage 解释为 joint depth error、surface distance 或 spatial accuracy；MOCAP wrist/joint 是内部骨架点，当前没有 independent depth joint labels/visibility。
- 全帧内容 gate 是 240×135 解码灰度+dHash 检查，不是原始编码 bit-exact，也不能证明空间精度或硬件零时差。
- 不能在缺少共享 TTL/同步 LED 的情况下把 +10–17 ms 视觉残余解释为零时间误差；该残余没有回写。
- 不能把 `PtpTimeStamp` 称为已验证的绝对 PTP epoch；当前仅使用其高分辨率相对节拍并由 Windows `TimeStamp` affine anchor。
- 不能报告手套绝对腕部平移、方向或完整 6DoF 精度；fusion 的 wrist translation 与 orientation 均来自 MOCAP。
- 不能把同段拟合结果称为独立测试精度。
- 不能把 rear-mount proxy 称为实测 module joint/6DoF，也不能启用已被 depth 反证的全局 `+Z37 mm` 作为 GT 修正。
- 不能再把已用于本轮模型选择和审查的 Take02/03 称为全盲 final test；新模型需在锁定后采集 untouched Take04。
- 不能只报告较低 median 而隐藏高 P95，也不能把同一 MOCAP asset 链内的 root/hand 一致性称为独立 GT 验证。

## 每日交付最小流程

1. 复制 `daily/_TEMPLATE.md` 到 `daily/YYYY/YYYY-MM-DD.md`。
2. 填入完整运行命令、数据 split、指标、边界和下一步。
3. 新数据追加到 `dataset/datalist.csv`，不覆盖旧行。
4. 新产物追加到 `evidence/index.csv`，同时记录 SHA256。
5. 影响复现或结论的变化写入 `CHANGELOG.md`。
6. 更新本页的“最近更新、最新日报、当前一句话结论”。

## 状态词

- `draft`：产物已生成，但命令、指标或证据尚未复核。
- `verified`：命令可复现，指标和证据已复核。
- `verified_conditional_fusion`：已完成复现与证据复核，但指标明确依赖外部 MOCAP wrist SE(3)，不可升级解释为独立 glove 6DoF。
- `verified_exposure_time_alignment`：已逐帧恢复相机采集时刻并完成独立视觉动作相位 QA；仍缺硬件共同事件和独立 2D GT。
- `blocked`：缺少必要输入或外部条件。
- `superseded`：已被新结果替代，保留记录但不再作为当前结论。

## 快速复现 MOCAP × 视频逐帧匹配

```bash
cd /home/runyi/Project/hands_reloc/GT_calib

/home/runyi/miniconda3/envs/viewer/bin/python mocap_video_overlay.py \
  --segment 01 --segment 02 --segment 03 \
  --draw-rear-mount-proxy \
  --frame-content-samples 0 \
  --output-width 960 \
  --output-dir outputs/mocap_video_rear_mount_review

/home/runyi/miniconda3/envs/viewer/bin/python mocap_world_origin_audit.py
/home/runyi/miniconda3/envs/viewer/bin/python mocap_wrist_depth_audit.py
```

## 快速复现 depth camera pose 审计

```bash
cd /home/runyi/Project/hands_reloc/GT_calib

python depth_calibration_audit.py \
  --formal-sample-count 61 \
  --monte-carlo-trials 5000
```

## 快速复现 raw-depth × MOCAP

```bash
cd /home/runyi/Project/hands_reloc/GT_calib

/home/runyi/miniconda3/envs/viewer/bin/python depth_mocap_overlay.py \
  --segment 01 --segment 02 --segment 03 \
  --max-interpolation-gap-ms 25 \
  --min-depth-mm 350 --max-depth-mm 1800 \
  --snapshot-count 6 \
  --output-dir outputs/depth_mocap_alignment_review
```

每段输出 `*_depth_mocap_aligned_strict25.mp4`、`*_h264.mp4`、`.contact.jpg`、`.alignment.csv` 和 `.metrics.json`；预期帧数为 1747/1804/1799。metrics 中的 canonical scope 是 `depth frame contract + temporal alignment + frozen spatial projection coverage`，不是 accuracy evaluation。

## 明日九项 P0

1. 三个 TAK 的 raw Marker C3D/CSV 与 residual、camera mask、occlusion/gap-fill。
2. 手部 Marker Set、21 点骨架和 solver/filter 配置。
3. 三段 Human joint quality 表。
4. 原始 camera—MOCAP 同场标定、坐标系契约、held-out correspondences 和官方 depth→color `R,t`。
5. Body Hand 刚体质量与到 glove root 的固定变换。
6. timestamp contract、logger/PTP 状态和 shared TTL 或同步 LED 原始事件。
7. 真实 session-neutral、glove solver assets、device-time raw IMU 和 replay 命令。
8. Femto Bolt 相机 MOCAP 刚体 marker trajectory/definition、mount 照片和 optical-frame 变换。
9. 不少于 30 个完整 pose 的 RGB-D×MOCAP 共同标定棒/板、raw correspondences 与整 pose held-out split。

字段级清单见 [DATA_REQUEST_2026-08-31.md](dataset/DATA_REQUEST_2026-08-31.md)。

## 快速复现当前 fusion 严格协议

```bash
cd /home/runyi/Project/hands_reloc/GT_calib

../hands_reloc/.venv/bin/python gt_calib_viz.py \
  --pose-mode mocap-root-fusion \
  --registration-segment 01 \
  --segment 01 --segment 02 --segment 03 \
  --max-interpolation-gap-ms 25 \
  --analyze-only \
  --output-dir outputs/mocap_root_fusion_strict25
```

Review 视频：

```bash
../hands_reloc/.venv/bin/python gt_calib_viz.py \
  --pose-mode mocap-root-fusion \
  --registration-segment 01 \
  --segment 01 --segment 02 --segment 03 \
  --max-interpolation-gap-ms 25 \
  --output-width 960 \
  --snapshot-count 6 \
  --output-dir outputs/mocap_root_fusion_review
```
