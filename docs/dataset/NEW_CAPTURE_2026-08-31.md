# 2026-08-31 新采集数据与世界坐标标定理解

## 选择与交付范围

`thor_new4_20260831_processed/` 包含一个无手套标定片段和三个动作片段，
不是单个 Take。最终九视频交付选择同步通过且时长最长的
`camera_glove_recording_20260831_161610 / Take_006`：

| Recording | RGB / Depth | Camera timecodes | Glove solved / hand | MOCAP | 角色 |
|---|---:|---:|---:|---:|---|
| 155410 | 98 / 98 | 46 | 无 | 无 | 无手套 CS-400 标定 |
| 161402 / Take_005 | 1868 / 1868 | 917 | 5793 | 8390 CMM | 备用动作 |
| 161610 / Take_006 | 2620 / 2619 | 1291 | 8206 | 11960 CMM | 最终新增动作 |
| 161912 / Take_007 | 1982 / 1982 | 980 | 6162 | 8552 CMM | sync report fail，未发布 |

最终视频编号为：旧 Take 01/02/03 各 MOCAP + solved pose 共六段；
Take_006 raw marker + solved pose 两段；155410 无手套标定一段。

## 新世界坐标变换

输入归档：`movementcap_20260831_worldcalib_tabletop_final.tar.gz`。
权威结果 member：

```text
movementcap_20260831_worldcalib/results/
20260831_processed_tabletop_origin/camera_to_world.json
```

该 JSON 明确适用于 155410、161402、161610、161912，固定相机序列号为
`CL8L563010D`。坐标为毫米右手系：桌面上的 CS-400 黑色顶点孔轴为原点，
长臂为 +X、短臂为 +Y、桌面法向向上为 +Z；反光球中心平面为 z=45 mm。

投影链使用列向量约定：

```text
p_color = T_color_from_world @ [p_world; 1]
```

随后应用 Take_006 RGB 内参及 OpenCV 五参数畸变。标定内部 RGB PnP
reprojection RMS/max 为 0.294/0.407 px；三个固定相机片段的静态 probe 最大
位移为 1.986 px。这些是标定重投影与固定相机一致性，不是动态手部独立 GT。

## Take_006 的时间合同

RGB 是 2620 帧 30 FPS，但 timecode 只有 1291 行。不能按 `2*i`，也不能把
MP4 PTS 当硬件曝光时间。相邻 `color_device_timestamp_us` 的步数为：

```text
k_i = max(1, round(delta_device_us / 33333.333333))
video_index_0 = 0
video_index_i = cumulative_sum(k_i)
```

Take_006 的步数分布为 1 帧 53 次、2 帧 1146 次、3 帧 91 次；稳健单帧
cadence 为 33428.5 us。累计末锚为 RGB index 2618，正好闭合 Depth 的
0..2618。RGB index 2619 没有同步锚，正式两段新增动作视频均将它排除。

每个相机锚到 CMM 的连续帧位置为：

```text
cmm_fractional_counter = cmavatar_frame_counter
                       + alignment_error_ms * 0.12
```

正号由 120 Hz 时间拟合验证；相机到 CMM 的绝对误差 p50/p95/max 为
1.966/3.938/4.493 ms。非锚 RGB 帧在相邻锚之间按 timestamp 插值，manifest
必须标为 `timestamp_inferred`，不能声称 MP4 内嵌逐帧硬件时间或已通过 raw
BAG 内容身份 gate。

## CMM marker 的真实能力

Take_006 CMM 有 133 个匿名字段，但在完整同步区间 10500 个 CMM 帧中只有
22 个 ID 100% 连续存在；其余为短时 transient/ghost。22 点稳定分成
`+world-X` 与 `-world-X` 两簇、各 11 点，投影抽检均落在两只物理手套的
反光球中心。因此第 7 段可以诚实画 raw marker clusters。

CMM 没有 anatomical side、joint name、parent edge、wrist quaternion 或 rigid
body definition，所以不能伪造成 2x21 手骨架，也不能把点云叫做逐关节 GT。

## Solved pose 的全局放置

Glove solved keypoints 是 20 点 root-local articulation，wrist XYZ 恒为零。
Take_006 没有 Human.cma/BVH wrist SE(3)，第 8 段采用明确受限的显示协议：

- 后方黑色腕块附近的一个 persistent marker 提供 wrist translation；
- 每侧固定四个 palm marker 只用于逐帧 root orientation；
- anatomical scale 冻结自旧 Take_000；
- 所有手指与 fingertip articulation 只来自 glove solver，绝不参与 root fit；
- camera/glove 20 ms validity gate 未通过时显示 unavailable，不 hold、不外推。

因此第 8 段是 `MOCAP-palm-root-conditioned glove articulation`，不是独立
glove wrist 6DoF，也不是 accuracy/GT 评估。

## 仍需向数据方索取

若要把新增数据升级成与旧三段同口径的 MOCAP 21-joint GT，至少还需：

1. Take_006 的 `Human.cma`，或 Skeleton_0/1 BVH 加完整 joint/side/root quaternion 合同；
2. raw RGB-D BAG 或逐 RGB frame 的 source index + exposure timestamp 表，用于内容同帧 gate；
3. raw marker definition：marker ID 到 side/anatomical landmark/rigid body 的映射；
4. marker residual、occlusion、gap-fill、camera mask 与 solved-joint quality；
5. 共享 TTL 或同步 LED 事件，用于独立估计时差、漂移和不确定度；
6. 后腕模块 rigid-body definition 或实测 `T_rear_module_from_hand_root`。
