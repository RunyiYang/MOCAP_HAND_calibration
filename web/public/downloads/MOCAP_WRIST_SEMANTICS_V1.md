# MOCAP wrist semantics 与后置模块可视化 v1

- 状态：`verified_display_proxy_with_depth_counterevidence_against_global_z37`
- 日期：2026-08-31
- 实现：`mocap_video_overlay.py`
- 正式视频：`outputs/mocap_video_rear_mount_review/`
- RGB 三列审计：`outputs/mocap_wrist_semantics_audit/`
- raw-depth 审计：`outputs/mocap_wrist_depth_audit/depth_audit.json`
- Claim：保留 `Human.cma` 的双手 2×21 个关节和原时间映射，在每只手的 wrist 后方额外画一个 hand-local rear-mount proxy。该点用于表达“前方大块是手背板、后方黑块是腕端模块”的显示语义；它不是测得的第 22 个 MOCAP 关节，也不是新的 GT。

## 1. 用户指出的语义错误

Take 02 的 output frame 661 中，原 21 点手骨架的 wrist 位于前方手背板区域；实际需要额外标出的手端是更靠近前臂的后侧黑色模块。两者不能被当成同一个物理点：

- 前方大块：手背上的 rigid plate / dorsum 区域；
- 后方小黑块：腕端模块，需要在可视化中显式表示；
- `Human.cma`：交付的是每手 21 个骨架点，不含一个有独立质量字段的 rear-module joint；
- `LeftForearm/RightForearm.bvh` 等文件的位置全零且旋转为常量占位，不能提供该黑块的实测 6DoF。

因此不能把原 wrist 静默改名为黑块，也不能无证据地移动全部 21 个关节来让二维画面“更贴”。

## 2. 对比过的三个方案

### 2.1 原始 21 点

完全使用 exposure-time 修正后的 timestamp chain，在相邻 120 Hz `Human.cma` 行之间插值，然后使用冻结的 world→RGB calibration 投影。它保留原数据语义和 raw-depth 一致性，但没有单独画出后侧模块。

### 2.2 hand-local rear-mount proxy（正式显示方案）

每只手取 wrist 索引 0 与 middle MCP 索引 9：

\[
P_{rear}=P_{wrist}-0.8\left(P_{middle\_mcp}-P_{wrist}\right)
\]

该计算在 MOCAP world 中完成，再使用同一个相机模型投影。实现满足：

- 原 21 点 world position 修改量严格为 0；
- 原 21 点 pixel position 修改量严格为 0；
- 每手只新增一个带白色中心的圆形显示点和 wrist→proxy 连线；
- 视频顶栏固定写 `rear mount proxy = VIZ-only, not GT`；
- metrics 固定写 `original_21_joint_positions_modified=false` 与 `additional_mocap_gt_joint=false`。

### 2.3 全局 world `+Z 37 mm`（保留为反例，不启用）

单帧二维观察曾提示，把全部 42 点统一加 `[0,0,37] mm` 会让根点和手指在 RGB 中整体上移。`37=45-8 mm` 也与本地 CS-400 marker-center height 和 TAK `groundAdjust.markerRadius` 的数值差相符，因此代码保留了显式诊断参数：

```text
--mocap-world-z-offset-mm 37
```

但这只是一个 origin-height 假设。它没有厂商 frame contract 支持，并被第 4 节的 raw-depth 结果明显反对，所以默认始终为 0；非零渲染会强制标为 `CANDIDATE ... / not GT`。

## 3. 24 帧 RGB held-out 对照

`mocap_world_origin_audit.py` 对 01/02/03 各取共同区间 6%、18%、30%、42%、54%、66%、78%、90% 的 8 帧，共 24 帧。用于提出假设的 Take 02 output 661 被明确排除。

每个时点并列三列：

1. 红色：raw 21 joints；
2. 青色 + 橙色菱形：raw 21 joints + rear proxy；
3. 绿色：global `+Z37` 2D counterexample。

自动验收结果：

| 检查 | 结果 |
|---|---:|
| source RGB / CMA low / CMA high / alpha 与正式 strict25 CSV 完全一致 | 24 / 24 |
| 解码到精确 source frame | 24 / 24 |
| rear proxy 对原 joints 的最大 world 修改 | 0 mm |
| rear proxy 对原 joints 的最大 pixel 修改 | 0 px |
| `+Z37` 刚性平移误差 | `2.84e-14 mm` |
| `+Z37` 最大骨长变化 | `4.26e-14 mm` |

三张总览：

- `outputs/mocap_wrist_semantics_audit/01_01_210814_Take_000_heldout_raw-rearproxy-z37_contact.png`
- `outputs/mocap_wrist_semantics_audit/02_02_210955_Take_001_heldout_raw-rearproxy-z37_contact.png`
- `outputs/mocap_wrist_semantics_audit/03_03_211139_Take_002_heldout_raw-rearproxy-z37_contact.png`

这些图说明三种画法的二维差异，并使人能检查 proxy 是否指向后侧模块。它们没有独立标注的 module center，因此不能单独选择一个世界外参或报告 pixel GT error。

## 4. 903 帧 raw-depth 反证

`mocap_wrist_depth_audit.py` 在每段完整同步区间均匀取 301 个 raw `mono16` depth frame，共 903 帧。相机 pose、depth 自身 timestamp、CMAvatar 插值和所有内外参都冻结；扫描 world Z `-20..70 mm`（5 mm 步长并额外包含 37 mm）。每个投影点读取 5 px 内最近非零 raw depth，残差定义为：

\[
e_z=Z_{joint}^{camera}-D_{nearest\ visible\ surface}
\]

关键 aggregate 结果：

| 对象 | 样本 | exact / 5 px depth coverage | signed median | absolute median | `abs<50 mm` / 全部投影 |
|---|---:|---:|---:|---:|---:|
| 原 wrist，Z=0 | 1,806 | 61.35% / 95.13% | +6.87 mm | 12.94 mm | 86.54% |
| 原 wrist，全局 Z=+37 | 1,806 | 97.07% / 99.11% | -66.80 mm | 67.00 mm | 1.61% |
| wrist + 4 MCP，Z=0 | 9,030 | 75.75% / 97.66% | -21.34 mm | 24.58 mm | 88.28% |
| wrist + 4 MCP，全局 Z=+37 | 9,030 | 61.96% / 91.10% | -63.98 mm | 64.04 mm | 12.21% |
| rear proxy，原 21 点不动 | 1,806 | 94.57% / 99.22% | -8.46 mm | 15.26 mm | 78.79% |

三段 wrist 的最低 median absolute residual 都在约 `+5 mm`，而不是 `+37 mm`。Take 02 output 661 上，全局 +37 后左右 wrist 的 signed residual 分别恶化到约 -85.3 / -74.7 mm。二维上移虽然更容易撞到某个前景表面，却把腕/掌预测到了可见表面前方；因此不能激活为全局 GT 或默认 overlay。

rear proxy 的 15.26 mm 与原 wrist 的 12.94 mm 属于同一量级，同时不改任何原关节。它支持当前“保留 21 点 + 单独表示后方模块”的显示决策，但 raw depth 测的是首个可见表面，仍不能把 proxy 变成物理 GT。

## 5. output 661 的结果

在 960×540 review 坐标中：

| 手 | raw wrist `(u,v)` | rear proxy `(u,v)` | 与 raw wrist 距离 |
|---|---:|---:|---:|
| Left（画面右） | `(692.26, 276.23)` | `(679.36, 251.53)` | 27.9 px |
| Right（画面左） | `(352.01, 284.15)` | `(356.60, 260.22)` | 24.4 px |

proxy 落到用户指出的后侧黑块区域；原 wrist 与全部指关节仍保持原投影。正式 Take 02 视频：

```text
/home/runyi/Project/hands_reloc/GT_calib/outputs/mocap_video_rear_mount_review/02_210955_Take_001_mocap_video_aligned_strict25_h264.mp4
```

## 6. 三段正式视频与验收

| Take | H.264 frames | full decode | full BAG↔MP4 gate | frame-map 与原版本 | H.264 SHA-256 |
|---|---:|---:|---:|---:|---|
| 01 | 1748 | PASS | 1815/1815 | byte-identical | `431a0b8849311de98e60e6dc5ee679763ab4f5955c5cf25509142ccbfededf1b` |
| 02 | 1805 | PASS | 1812/1812 | byte-identical | `7a141794217dd773c1a7b4c51d21980210c078047edee0e260013c9acc968093` |
| 03 | 1800 | PASS | 1810/1810 | byte-identical | `c3d9dc519a99efe56d52047b49120b25ab432efb2f8fdf3c1d96d69aec7c12da` |

全部视频为 H.264、960×540、30 FPS。三份 alignment CSV 与原正式版本 SHA-256 完全相同，证明 proxy 没有改帧选择、timestamp、CMA bracket、alpha 或原 wrist pixel 字段。

## 7. 复现

```bash
cd /home/runyi/Project/hands_reloc/GT_calib

/home/runyi/miniconda3/envs/viewer/bin/python mocap_video_overlay.py \
  --segment 01 --segment 02 --segment 03 \
  --output-dir outputs/mocap_video_rear_mount_review \
  --draw-rear-mount-proxy \
  --frame-content-samples 0 \
  --output-width 960

/home/runyi/miniconda3/envs/viewer/bin/python mocap_world_origin_audit.py
/home/runyi/miniconda3/envs/viewer/bin/python mocap_wrist_depth_audit.py
```

`--mocap-world-z-offset-mm` 只用于显式反证实验，不属于上面的正式命令。

## 8. 还缺什么才能把后侧黑块作为 GT

至少需要以下独立数据之一，优先前两项：

1. 把后侧模块定义成 MOCAP rigid body，交付 marker layout、逐帧 6DoF、residual、visible marker count、occlusion/gap-fill 标记；
2. 实测固定变换 `T_rear_module_from_hand_skeleton`，包括 CAD/量具尺寸、坐标轴照片、左右手分别测量和重复装配误差；
3. RGB-D 与 MOCAP 同时可见的刚性标定棒或 fiducial，并给出 held-out pose；
4. 独立标注的后侧模块 2D center / depth surface，且标注帧不参与 proxy 尺度选择；
5. `Human.cma` joint 与 `Body Hand` rigid-body 的厂商 frame/solver contract。

在这些数据到齐前，当前结果应叫“rear-module-aware visualization”，不能叫“rear module calibrated GT”。

## 9. 证据边界

- raw depth 是相机射线上的首个可见表面，不是关节中心；黑色/反光模块也可能产生洞。
- 5 px 最近有效搜索会填补深度孔洞，也可能选中邻近表面。
- `Human.cma` 没有 per-joint visibility、confidence、residual 或 gap-fill 字段。
- 三段来自现有记录，没有 untouched 第四段或独立 module label。
- proxy 的 0.8 是 hand-local 显示尺度，不是测量得到的 rigid transform。
- 当前修正不改变时序、相机外参、原 21 点、BVH 动作或任何数值 GT。
