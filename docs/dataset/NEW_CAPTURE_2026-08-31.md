# 2026-08-31 新采集数据与 Take_007 CMM 对齐

## 最终选择

本页替代此前以 `161610 / Take_006` 为新增动作的解释。用户补充的 marker
编号表和实物照片给出了最新一段数据的 reflector 语义，因此最终九视频改用
`camera_glove_recording_20260831_161912 / Take_007`：

| Recording / Take | RGB / Depth | Camera anchors | Glove solved / hand | CMM rows | 最终角色 |
|---|---:|---:|---:|---:|---|
| 155410 | 98 / 98 | 46 | 无 | 无 | video 09 无手套 CS-400 标定 |
| 161402 / Take_005 | 1868 / 1868 | 917 | 5793 | 8390 | 备用动作 |
| 161610 / Take_006 | 2620 / 2619 | 1291 | 8206 | 11960 | superseded，不再是 video 07/08 |
| 161912 / Take_007 | 1982 / 1982 | 980 | 6162 | 8552 | video 07/08 最终动作 |

最终编号固定为：旧 Take 01/02/03 各一段 MOCAP 与一段 solved pose，共六段；
Take_007 labeled CMM 与 aligned solved pose 两段；155410 无手套标定一段。

## 数据能力边界

Take_007 的 MOCAP 输入只有 `.cmm`。每一行包含 120 Hz `FrameCounter` 和每个
反光片的世界坐标 XYZ（毫米）；它没有 `Human.cma`、BVH hierarchy、关节角、
父子拓扑或 wrist quaternion。因此本交付所说的“20 点”是双手共 20 个实测
表面 landmark，不是 20 个独立旋转自由度，也不是完整的 MOCAP hand skeleton。

手套 solver 另行输出每手 20 个 anatomical keypoint。CMM 反光片贴在手套表面，
不位于解剖关节中心；二者即使时间与外参正确，也会保留系统性空间差异。

## 世界坐标与 RGB 投影

世界坐标来自 `movementcap_20260831_worldcalib_tabletop_final.tar.gz` 中的：

```text
movementcap_20260831_worldcalib/results/
20260831_processed_tabletop_origin/camera_to_world.json
```

该结果声明适用于 155410、161402、161610、161912，固定相机序列号为
`CL8L563010D`。坐标单位是毫米、右手系：桌面 CS-400 黑色顶点孔轴投影为
原点，长臂为 +X、短臂为 +Y、桌面法向向上为 +Z；reflector center plane 为
`z=45 mm`。

所有动态点先做 `p_color = T_color_from_world @ [p_world; 1]`，再使用目标
recording 自己的 RGB intrinsics 和 OpenCV 五参数畸变投影。Take_007 video
07/08 使用 161912 intrinsics；video 09 使用 155410 intrinsics，二者不能因数值
接近就复用错误 provenance。

## 照片编号、CMM track 与 solved keypoint 对应

下表由用户提供的编号表和实物照片固化。Solved index 采用
`wrist=0, thumb=1..3, index=4..7, middle=8..11, ring=12..15,
pinky=16..19`：

| # | 实物语义 | Left CMM ID | Right CMM ID | Solved index | 是否显示/拟合 |
|---:|---|---|---|---:|---|
| 1 | thumb tip | `11781`，随后 `12503` | `12436` | 3 | 是 |
| 2 | thumb base | `11978` | `12435` | 1 | 是 |
| 3 | index tip | `12003` | `11619` | 7 | 是 |
| 4 | index base | `12001` | `11614` | 4 | 是 |
| 5 | middle tip | `12006` | `11616` | 11 | 是 |
| 6 | middle base | `12013` | `12434` | 8 | 是 |
| 7 | ring tip | `12045` | `11625` | 15 | 是 |
| 8 | ring base | `12015` | `11685` | 12 | 是 |
| 9 | pinky tip | `12076` | `11627` | 19 | 是 |
| 10 | pinky base | `12051` | `11630` | 16 | 是 |
| 11 | dorsum black module | `12064` | `12433` | 不对应 joint | 仅构造 root |

左手 #1 在同一物理轨迹中由 `11781` re-ID 为 `12503`，中间只有两个缺失 CMM
帧。Loader 将二者作为同一个 logical marker，并只允许最多四个 CMM 帧的内部
短缺口做线性插值；它不会用任意“最近点”跨长缺口补轨迹。

Video 07 每手显示 #1..#10，并连接五条 `virtual wrist → base → tip` 链。
#11 不显示为第 21 点，它只参与 virtual wrist 的构造。

## Virtual wrist：掌根再向后 20 mm

对每只手、每一帧，先用食指到小指的四个 base reflector 求掌部中心：

```text
mcp_center = mean(p4, p6, p8, p10)
backward   = normalize(p11 - mcp_center)
virtual_wrist = p11 + 20 mm * backward
```

这里的“向后”是由该手当前姿态决定的 hand-local 方向：从四个 base 的中心指向
黑色手背模块 #11，再沿同一方向延长 2 cm。它不是固定 world X/Y/Z 平移。
默认 `rear_offset_mm=20`；#11 和这个 20 mm 先定义 root，浏览器中的 manual XYZ
随后才作为 operator offset 应用。

## RGB ↔ CMM 时间对齐

Take_007 有 1982 个 RGB 帧，但只有 980 行稀疏 camera timecode。相邻设备时间
按约 30 FPS cadence 恢复视频 index：

```text
k_i = max(1, round(delta_color_device_us / 33333.333333))
video_index_0 = 0
video_index_i = cumulative_sum(k_i)
```

实际步数分布为 `1:40, 2:878, 3:60, 4:1`，稳健 cadence 为 33427.0 us。
最后一个 camera anchor 对应 RGB index 1980，因此正式输出为 source RGB
`0..1980`，共 1981 帧；source frame 1981 没有 camera/CMM 锚，明确排除，不能
复制上一姿态或外推。

Camera anchor 到连续 CMM 位置采用：

```text
cmm_fractional_counter = cmavatar_frame_counter
                       + alignment_error_ms * 0.12
```

再在相邻 camera anchors 之间按设备 timestamp 插值。980 个 camera anchor 到
CMM 的绝对误差 median/P95/max 为 2.238/3.988/4.577 ms，且 980/980 在 20 ms
内。上游 `sync_quality_report.json` 的总 `pass=false` 来自 glove-to-CMAvatar
的 48/18505 个大于 20 ms 样本，不代表 camera-to-CMM 失败；因此视频 07 可
完整发布 1981 帧，而视频 08 仍按 camera + glove 联合 validity gate 显示有效性。

## Solved pose 对齐方法

对每侧独立处理。令 `g_j` 为 root-local solved keypoint，`m_j` 为照片 #1..#10
的 CMM 表面点，`r` 为上述 virtual wrist：

1. 使用十个固定对应，形成 `g_j - g_wrist` 与 `m_j - r`；
2. 在所有联合有效帧上估计尺度，并把每侧尺度冻结为中位数；
3. 每帧用十点 Kabsch 求 proper rotation（`det(R)=+1`）；
4. 输出 `q_k = r + manual_xyz + s * R * (g_k - g_wrist)`。

左/右冻结尺度分别为 0.831746 / 0.859253。联合有效帧为 1367/1981，覆盖率
69.01%；无效帧显示 unavailable，不 hold、不外推。十个表面对应的 same-take
残差如下：

| Side | correspondence count | median | P95 | max |
|---|---:|---:|---:|---:|
| left | 13670 | 21.02 mm | 52.37 mm | 85.58 mm |
| right | 13670 | 22.11 mm | 54.09 mm | 99.16 mm |

这些 residual 既参与同一段的 rotation/scale 拟合，又比较“手套表面 reflector”
与“anatomical solved keypoint”，所以只能描述可视化拟合一致性。它们不是
held-out residual，不是独立 GT accuracy，也不能证明 glove-only wrist 6DoF。

## 手动 XYZ 标定台

网页的“手动 XYZ 标定”区在 clean Take_007 RGB 上实时重投影两个 3D layer：

- CMM：`[1981, 2, 11, 3]`，每手为 synthetic root + #1..#10；
- solved：`[1981, 2, 20, 3]`；
- 全局 XYZ：同时平移双手，单位为 MOCAP world mm；
- 左/右 residual XYZ：可选、可联动，用于记录两侧不同的人工修正；
- layer、播放、逐帧和 reset 控件；浏览器本地保存当前值；
- 导出 schema：`gt_calib.manual_xyz_profile.v1`。

可接受的 profile 必须完整声明数据和单位，最小结构为：

```json
{
  "schema": "gt_calib.manual_xyz_profile.v1",
  "source_recording": "camera_glove_recording_20260831_161912",
  "source_take": "Take_007",
  "coordinate_system": "mocap_world_mm",
  "units": "mm",
  "global_world_xyz_mm": [0.0, 0.0, 0.0],
  "left_world_xyz_mm": [0.0, 0.0, 0.0],
  "right_world_xyz_mm": [0.0, 0.0, 0.0],
  "rear_offset_mm": 20.0
}
```

Loader 对缺字段、错误 recording/take、错误 coordinate system/units、非有限或
非三元素 XYZ 都 fail closed。`rear_offset_mm` 必须保持 20 mm；manual profile
只能调世界 XYZ，不能覆盖已确认的 hand-local root 定义。

启动：

```bash
uv run gt-calib-delivery serve --host 127.0.0.1 --port 8811
```

打开 `http://127.0.0.1:8811/final-nine/#manual-calibration`，调整后下载
`take007_manual_xyz_calibration.json`。将人工结果应用到视频 07/08：

```bash
uv run gt-calib-delivery render-new \
  --destination outputs/take007_manual_review \
  --manual-profile /path/to/take007_manual_xyz_calibration.json

uv run gt-calib-delivery build \
  --destination rebuilt_9_video_delivery \
  --manual-profile /path/to/take007_manual_xyz_calibration.json
```

Profile 中三个 XYZ vector 进入 metrics/provenance。若实际传入 profile，build
还会将原始 JSON 复制到
`calibration-workbench/applied_manual_profile.json`，在 manifest 记录路径与
SHA-256，并在 validation 中重新验 hash；默认零 offset build 明确记录
`applied_manual_profile=null`。人工调到“看起来重合”仍是 operator-selected
display calibration，不能升级为独立标定精度证据。

## Video 09 的对应关系

Video 09 只使用 `camera_glove_recording_20260831_155410`：98 帧无手套 RGB、
该 recording 自己的 `camera_1_intrinsics.json`，以及归档
`movementcap_20260831_worldcalib_tabletop_final.tar.gz` 中声明适用于该 recording
的 CS-400 world→color 结果。Calibration reference 是 155410 RGB frame 30。

它不复用 Take_006/Take_007 的 RGB 或 intrinsics，也不使用旧 `00` 片段。RGB
PnP reprojection RMS/max 0.294/0.407 px 是同一标定内部一致性，不是动态手部
精度。`Depth.mp4` 是 8-bit lossy preview，未作为公制标定输入。

## 仍缺少的独立验证数据

若要把该可视化升级成独立的动态手部 GT 评估，仍需：

1. Marker Set 的 surveyed reflector-center → anatomical joint-center / hand-root 固定变换及公差；
2. 独立 held-out calibration motion 或共同可见的标定物，不参与本轮十点 fit；
3. 每个 reflector 的 residual、visibility、occlusion、gap-fill 与 re-ID 事件表；
4. 共享 TTL/同步 LED 原始事件，用于独立估计 RGB、CMM、glove 的时差和漂移；
5. 若需要 rotational DoF/骨架 GT，提供对应 Take 的 BVH/CMA hierarchy、关节轴定义和 root pose，而不只是 CMM XYZ。

完整算法与验收口径见
[`TAKE007_CMM_ALIGNMENT_V1.md`](../calibration/TAKE007_CMM_ALIGNMENT_V1.md)。
