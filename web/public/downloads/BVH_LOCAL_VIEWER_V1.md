# Skeleton_0/1 BVH 本地播放器 V1

- 日期：2026-08-31
- 动作源：仅 `Take_NNN_Skeleton_0.bvh` 与 `Take_NNN_Skeleton_1.bvh`
- 页面：`http://127.0.0.1:8811/bvh/?take=02`
- 导出 schema：`gt-calib-bvh-web-v1`
- Generator：`bvh_web_export.py`

## 1. 可视化范围

`Skeleton_0` 是左手，`Skeleton_1` 是右手。每份文件包含 21 个有 channel 的节点：一个名为 `Hips` 的手根和五根手指各四个关节；这里的 `Hips` 不是人体髋部。每只手有 20 条骨骼连接、66 个 motion channels，原始帧率约 120 Hz。

页面不读取 `LeftArm/Forearm/LeftHand/RightArm/Forearm/RightHand/Torso.bvh`。这些文件各自只有一个 6DoF ROOT，没有细化手指 hierarchy。

Canvas 中左手为洋红色、右手为青色。它使用 BVH forward kinematics 后的双手 42 点，支持 Take 切换、播放/暂停、逐帧拖动、倍速、鼠标旋转、滚轮缩放和双击复位。所有 HTML、CSS、JavaScript、视频与数据均从 localhost 提供，不使用 CDN。

## 2. 真实 BVH 边界

| Take | BVH frames | strict source frames | 原始 Frame Time | 网页动作帧 |
|---|---:|---:|---:|---:|
| 01 / Take_000 | 7216 | 1…7214 | 0.00833333 s | 3608 |
| 02 / Take_001 | 7483 | 1…7481 | 0.00833333 s | 3741 |
| 03 / Take_002 | 7350 | 1…7348 | 0.00833333 s | 3675 |

MovementCap 导出的 BVH 存在两个特殊边界：

- frame 0 是全零 dummy；
- frame `N-1` 是没有已交付 CMA counter/timestamp 对应项的额外尾帧。

正式网页只发布 `1…N-2`。动作位置从约 120 Hz 抽样到约 60 Hz，并以 `0.001 BVH unit` 量化；最大绝对量化误差不超过 `0.0005 BVH unit`。原始 BVH 文件仍可直接下载。

## 3. 与视频的逐帧映射

参考 H.264 是 30 FPS CFR 播放主时钟。每个 output frame 从既有 alignment CSV 读取 `mocap_low_frame_counter`、`mocap_high_frame_counter` 和 `interpolation_alpha`。实测正确的源索引公式为：

```text
bvh_source_frame = mocap_frame_counter - Human.cma.first_frame_counter
```

不能加一。真实数据逐帧验证显示 BVH frame 1 对应 CMA 第二行，BVH frame `N-2` 对应 CMA 最后一行。

网页使用：

```text
video_sync.bvh_fractional_source_frame_by_output_frame[output_frame]
```

取得 fractional BVH frame，再在相邻已导出 world-position frames 之间线性插值。Take 01 和 Take 03 的视频 output frame 0 bracket 会碰到 dummy frame 0，因此明确写为 `null` 并从 output frame 1 开始；Take 02 全区间有效。页面不会把首帧静默挪成下一帧。

## 4. 复现

```bash
cd /home/runyi/Project/hands_reloc/GT_calib

python bvh_web_export.py --segment 01 --segment 02 --segment 03
python web/build_site.py
python -m unittest tests.test_bvh_web_export -v
python -m unittest discover -s tests -v
```

启动支持 MP4 byte-range seek 的只读 localhost 服务：

```bash
python local_review_server.py
```

然后直接打开：

```text
http://127.0.0.1:8811/bvh/?take=01
http://127.0.0.1:8811/bvh/?take=02
http://127.0.0.1:8811/bvh/?take=03
```

## 5. 解释边界

- Canvas 动作位置来自 Skeleton_0/1 BVH；Human.cma 只用于证明 vendor ordinal contract，alignment CSV 只提供视频时钟映射。
- BVH 动作页使用 BVH 世界坐标做独立正交视图，没有投影到 camera optical frame。
- “逐帧同步”表示每个可用视频输出帧有明确的 BVH source bracket；没有 shared TTL 时仍不能宣称硬件零时间误差。
- 该页是动作审阅器，不是 camera calibration、skeleton-to-surface error 或独立空间 GT 评估。

最终验收：BVH 导出专项 `10/10`、Range 服务专项 `5/5`、全量单元与数据/网页契约测试 `90/90` 通过（89.468 s）；真实 Chrome 已验证 Take 01 的无效首帧跳过、Take 01/02 切换、播放、视频 ready state、非空 Canvas，以及仅从 localhost 加载资源。真实 MP4 Range 请求返回 `206 Partial Content`；Take 02 可从 frame 0 直接 seek 到 20.000 s / output frame 600，并同步更新到 BVH 2492.533，期间 runtime error 为 0。
