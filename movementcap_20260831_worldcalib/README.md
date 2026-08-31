# Movementcap CS-400 世界坐标系与点云

独立工程对应：

`/home/user/Project/intern/20260729-task/movementcap--同步整理_20260829_三段`

`00_210652_三秒视频对齐_无手套` 用于求一次相机到动捕世界系的固定变换；相机未移动，
结果复用于 Take 000、001、002，不修改此前的 `single_camera_worldcalib`。

## 最终坐标定义

- 原点：CS-400 顶点标记中心正下方的桌面点；应用 45 mm 标记中心高度偏移。
- `+X`：按本任务约定，沿长尺臂从顶点向外。
- `+Y`：按本任务约定，沿短尺臂从顶点向外。
- `+Z`：垂直桌面向上。
- 右手系；最终标定 JSON 和点云均使用毫米。

最终标定位于 `results/manual_final/`：

- `camera_to_world.json`：深度/彩色相机与世界系的双向矩阵；
- `reference_rgb_markers_and_world_axes.png`：反光标记、桌面世界轴和标记高度参考线；
- `reference_rgb_marker_vs_ground_axes_crop.png`：局部放大；
- `reference_depth_markers_and_axes.png`：原始深度中的标记和坐标轴；
- `validation_*.png`：三段正式 Take 的动捕手部重投影检查。

标定使用完整 109 帧原始毫米深度。桌面拟合 P50/P95 残差为 0.77/3.91 mm；四个标记到
等权正交轴的最大 RGB 垂距约 2.43 px。长短臂实测夹角为 90.949°，X/Y 各分担
0.474° 的正交修正。

## 复算标定

```bash
cd /home/user/Project/intern/20260729-task/movementcap_ruler_worldcalib

.venv/bin/python tools/calibrate_manual_cs400.py \
  --dataset-root '../movementcap--同步整理_20260829_三段' \
  --config configs/manual_cs400_points.json \
  --output-dir results/manual_final
```

该 BAG 是 ROS1 BAG，但图像消息采用 Orbbec 录制器的实际线布局；工具直接解析 BAG，
不依赖 OrbbecSDK 回放。

## 从视频生成世界系三维点云

下面的命令读取 Take 000 第 30 帧 RGB-D，使用最终世界变换输出毫米单位的彩色点云：

```bash
.venv/bin/python tools/export_world_point_cloud.py \
  --bag '../movementcap--同步整理_20260829_三段/01_210814_Take_000/原始BAG与内参/camera_1_rgb_depth.bag' \
  --calibration results/manual_final/camera_to_world.json \
  --output-dir results/world_point_cloud_take000_frame030 \
  --frame-index 30 \
  --stride 1 \
  --max-points 500000
```

主要输出：

- `selected_frame_depth_camera_coordinates.ply`：原始深度相机系；
- `selected_frame_world_coordinates.ply`：世界坐标系；
- `selected_frame_world_coordinates_with_axes.ply`：世界点云内附加红 X、绿 Y、蓝 Z；
- `world_axes_lines.ply`：独立坐标轴线；
- `selected_rgb_world_axes_overlay.png`：世界坐标轴与所选视频帧的叠加；
- `world_point_cloud_preview.png`：三维预览；
- `selected_rgb.png`、`selected_raw_depth_preview.png`；
- `point_cloud_metadata.json`：输入帧、时间差、矩阵、边界和桌面 Z 检查。

当前样例包含 177,485 个 RGB 着色点，RGB/深度设备时间差 1.306 ms。转换后桌面候选点
的世界 `Z` 中位数为 −1.50 mm，95% 绝对偏差为 5.42 mm。

自动标定设计见 `AUTOMATIC_CALIBRATION_PLAN.md`。
