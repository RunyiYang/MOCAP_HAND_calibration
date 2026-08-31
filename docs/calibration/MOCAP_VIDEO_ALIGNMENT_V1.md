# MOCAP ↔ RGB Video Alignment v1

- 状态：`verified_frame_complete_timestamp_match_with_full_frame_content_identity_gate`
- Schema：`gt_calib.mocap_video_alignment.v2`
- 数据：`01_210814_Take_000`、`02_210955_Take_001`、`03_211139_Take_002`
- 实现：`mocap_video_overlay.py`
- 正式产物：`outputs/mocap_video_alignment_review/`
- Claim：发布片段中的每一帧都由该帧的 RGB device acquisition timestamp 映射到 CMAvatar 时间，在相邻 120 Hz `FrameCounter` 之间插值，再用已交付的 camera/world calibration 投影；BAG message i ↔ MP4 frame i 的身份另由正式全帧内容级 gate 验收。

> 2026-08-31 wrist 语义补充：本页保留未增加显示点的 21-joint baseline。当前面向人工 review 的版本位于 `outputs/mocap_video_rear_mount_review/`，只额外绘制后侧腕部模块 proxy，不改原 21 点或本页的逐帧时间映射。选择 proxy、拒绝全局 `+Z37 mm` 的 RGB/depth 证据见 [MOCAP_WRIST_SEMANTICS_V1.md](MOCAP_WRIST_SEMANTICS_V1.md)。

该产品只画 MOCAP，不加载 glove。这里的 **frame-complete** 表示裁剪后片段的每一帧都有可审计的时间映射和有效插值；source identity 则对 source MP4/BAG 的 1815/1812/1810 个 index 全部完成 240×135 解码灰度+dHash 比较。它不是原始 MJPG/MP4 编码 bit-exact，也不表示逐关节像素误差为零、空间精度已验证或硬件同步为零误差。

## 1. 修复的旧时间错误：poll/callback 不是曝光时刻

`camera_cmavatar_alignment.csv` 中以下两列记录 Thor 从缓冲队列取到帧时的 host poll/callback 时刻：

- `thor_monotonic_ns`
- `cmavatar_windows_estimated_ns`

旧实现把这些时刻直接配给 `color_device_timestamp_us` 对应的 RGB 图像。实际图像在 callback 前已经完成采集并进入缓冲区，因此旧映射把约 237–303 ms 的 queue/callback latency 错当成传感器时间，令投影的 MOCAP 动作领先画面约 7–8 个 30 FPS 帧。

对 clock-map 的每一行 (r)，定义：

- (d_r=1000\cdot\texttt{color\_device\_timestamp\_us}_r)：RGB acquisition timestamp，ns；
- (q^R_r=\texttt{thor\_realtime\_ns}_r)：Thor realtime poll 时刻；
- (q^M_r=\texttt{thor\_monotonic\_ns}_r)：Thor monotonic poll 时刻；
- (q^C_r=\texttt{cmavatar\_windows\_estimated\_ns}_r)：映射到 CMAvatar/Windows 的 poll 时刻。

逐行测得的 capture-to-poll 延迟是：

\[
\ell_r=q^R_r-d_r
\]

然后把两个 host clock 都退回到 acquisition instant：

\[
h_r=q^M_r-\ell_r,\qquad c_r=q^C_r-\ell_r
\]

旧错误相当于使用 (q^C_r)；当前实现使用修正后的 (c_r)。对任意 RGB/BAG frame (i)，先取其 device timestamp (d_i)，再对 clock-map 行做分段线性插值：

\[
t^{C}_{i}=\operatorname{lerp}_{d_r\rightarrow c_r}(d_i)
\]

同样可得到修正后的 host monotonic acquisition time。映射只允许在 clock-map device timestamp 范围内插值；0.5 µs inclusive tolerance 只用于避免 epoch 浮点舍入把端点误删，±1 µs 已判为越界并返回 invalid，而不是被 `np.interp` 静默夹到端点。

三段逐行 latency 分布如下，单位 ms：

| Take | min | median | P95 | max |
|---|---:|---:|---:|---:|
| 01 | 237.915261 | 255.102949 | 270.295412 | 303.013267 |
| 02 | 237.213445 | 255.250336 | 270.358207 | 303.193256 |
| 03 | 236.498212 | 254.007910 | 269.718802 | 297.037131 |

## 2. CMAvatar 的 PTP affine anchor

`Human.cma` 同时提供低精度的 Windows display timestamp、连续 `FrameCounter` 和高分辨率 `PtpTimeStamp`。当前没有交付 PTP absolute epoch/Grandmaster 契约，因此不能把 `PtpTimeStamp` 直接称为硬件绝对时间。当前实现用它提供高分辨率相对节拍，用 Windows timestamp 锚定绝对时间。

令第 (k) 行 PTP 秒数为 (p_k)，解析后的 Windows 秒数为 (w_k)。最小二乘 slope 为：

\[
a=\frac{\sum_k(p_k-\bar p)(w_k-\bar w)}{\sum_k(p_k-\bar p)^2}
\]

最终用于插值的 MOCAP 时间为：

\[
m_k=\bar w+a(p_k-\bar p)
\]

等价截距为 (b=\bar w-a\bar p)，即 (m_k=a p_k+b)。与 120 Hz `FrameCounter` 的独立 cadence 检查是：

\[
e^{120}_k=\left[(p_k-p_0)-\frac{F_k-F_0}{120}\right]\cdot10^6\ \mu s
\]

| Take | PTP→Windows slope (a) | display fit abs residual median/P95/max (ms) | PTP vs 120 Hz counter abs residual median/P95/max (µs) |
|---|---:|---:|---:|
| 01 | 0.9999783092542173 | 0.362635 / 1.098704 / 2.208948 | 1.666667 / 3.000000 / 3.666667 |
| 02 | 0.9999788738662096 | 0.333548 / 1.037097 / 2.649307 | 1.333334 / 2.666667 / 3.333334 |
| 03 | 0.9999772778400298 | 0.335217 / 0.981283 / 2.196312 | 1.000000 / 2.000001 / 2.666667 |

该拟合证明高分辨率 cadence 稳定；它不补足缺失的 absolute PTP epoch contract。

## 3. 每个发布帧的完整计算

### 3.1 MP4 frame 与 BAG timestamp

交付契约规定 source MP4 frame (i) 对应 RGB BAG color message (i)。脚本先要求两者总数严格相等；否则 fail closed。三段分别为 1815/1815、1812/1812、1810/1810。

稀疏 clock-map 中落入 BAG 区间的 device timestamps 也必须在 BAG timestamp 集合中精确出现：01 为 875/875，02 为 878/878，03 为 876/876。随后所有视频帧均使用各自 BAG message 的 `rgb_device_timestamp_us`，不使用 `frame_index / 30` 做 MOCAP 查询。

帧数相等和 timestamp 命中都不能单独证明 message i 与 frame i 是同一画面。正式 v2 运行传入 `--frame-content-samples 0`：从 ROS1 chunk 索引解码每个 Orbbec MJPG message，再顺序解码 MP4 的每一帧，在 240×135 灰度图上比较同 index，并搜索 `-2/-1/0/+1/+2` 五个候选 offset。默认值 31 使用 `unique rint(linspace(0, N-1, 31))`，只用于快速回归，不是正式发布证据。

内容 gate 的 fail-closed 门限是：同 index `luma MSE max≤6`、`PSNR min≥39 dB`、`dHash64 Hamming max≤4`；全局最佳 offset 必须为 0；raw sample 中 offset 0 最优比例≥0.90；至少 24 帧的 best/runner-up MSE ratio≥1.05，且所有这些“可辨帧”都必须选择 offset 0。

| Take | raw best-offset=0（比例） | 可辨帧 best-offset=0 | MSE max | PSNR min | dHash max |
|---|---:|---:|---:|---:|---:|
| 01 | 1764/1815（0.9719008264） | 1715/1715 | 4.7364506721 | 41.3762734217 dB | 3 |
| 02 | 1768/1812（0.9757174393） | 1729/1729 | 4.6256175041 | 41.4791064309 dB | 3 |
| 03 | 1809/1810（0.9994475138） | 1794/1794 | 4.4396605492 | 41.6573059509 dB | 3 |

三段 global best offset 均为 0，所有可辨帧都选择 offset 0。raw best-offset=0 小于总帧数，是因为静止或近重复画面在 ±2 帧窗口中可能把相邻候选排到最前；这些帧的 best/runner-up ratio 小于 1.05，因此不具备 offset 判别力，不能解读为运动帧错位。三段 content/timecode/temporal acceptance 全部 PASS，正式 metrics 记录 `full_frame_content_comparison=true`。

“全帧内容证据”仅表示每个 BAG/MP4 index 都完成上述解码视觉内容检查。由于两种容器和压缩链不同，它不是原始编码字节或压缩码流 bit-exact；它也不验证第 7 节的空间投影精度，更不能替代 shared exposure/OptiTrack TTL 来证明硬件零时差。

### 3.2 120 Hz bracket 与位置插值

对修正后的目标时间 (t^C_i)，寻找相邻 MOCAP 行 (k,k+1)：

\[
m_k\le t^C_i\le m_{k+1},\qquad \Delta m_i=m_{k+1}-m_k\le25\text{ ms}
\]

\[
\alpha_i=\frac{t^C_i-m_k}{m_{k+1}-m_k}
\]

对左右手每个关节 (j) 的世界坐标逐分量插值：

\[
P^W_{i,h,j}=(1-\alpha_i)P^W_{k,h,j}+\alpha_iP^W_{k+1,h,j}
\]

不做 nearest-frame rounding。有效帧还必须同时满足：clock-map 内部、common interval 内部、
(0\le\alpha_i\le1)、finite points，并且 bracket span 不超过 25 ms。

三段的 bracket P95/max 都是 8.333921/8.333921 ms；nearest-sample delta P95 分别为 3.960204、3.957748、3.954959 ms，max 分别为 4.163265、4.165173、4.164696 ms。25 ms 是两侧源样本 bracket gate，不是 nearest-sample age。

### 3.3 MOCAP world 到 RGB pixel

对每个插值后的世界点 (P^W=[X,Y,Z]^T)，使用交付的 forward transform：

\[
P^C=A_{CW}P^W+t_{CW}=[X_C,Y_C,Z_C]^T
\]

这里刻意写作 `A_CW`，而不是声称它是纯旋转 `R`：交付 3×3 block 的 determinant 为 0.994914，最大正交误差为 0.010052。只有 (Z_C>0) 的点进入投影。

\[
x=X_C/Z_C,\qquad y=Y_C/Z_C,\qquad r^2=x^2+y^2
\]

Orbbec distortion 系数顺序是 (k_1,k_2,k_3,k_4,k_5,k_6,p_1,p_2)：

\[
g(r)=\frac{1+k_1r^2+k_2r^4+k_3r^6}{1+k_4r^2+k_5r^4+k_6r^6}
\]

\[
x_d=xg(r)+2p_1xy+p_2(r^2+2x^2)
\]

\[
y_d=yg(r)+p_1(r^2+2y^2)+2p_2xy
\]

\[
u=f_xx_d+c_x,\qquad v=f_yy_d+c_y
\]

正式 metrics 保存了实际 (A_{CW},t_{CW},f_x,f_y,c_x,c_y) 和 distortion 数组，避免文档手抄参数成为第二事实源。

### 3.4 无缺口发布区间

脚本取全部有效帧的首尾，若首尾之间存在任何 invalid frame 就直接报错，而不是静默删帧。发布帧 (o) 与 source frame 的关系为：

\[
i=o+i_{first},\qquad o=0,\ldots,N-1
\]

`.alignment.csv` 为每个 (o) 保存 source frame、device/CMAvatar timestamp、MOCAP low/high `FrameCounter`、
(\alpha)、bracket、nearest delta 和左右 wrist pixel。

## 4. 三段 frame-complete 结果

| Take | source MP4/BAG | 发布 source frame（inclusive） | output frame（inclusive） | 发布帧/CSV rows | trim prefix/suffix | coverage | clock uncertainty (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 01 | 1815 / 1815 | 60–1807 | 0–1747 | 1748 / 1748 | 60 / 7 | 100% | 0.650287 |
| 02 | 1812 / 1812 | 0–1804 | 0–1804 | 1805 / 1805 | 0 / 7 | 100% | 0.662216 |
| 03 | 1810 / 1810 | 3–1802 | 0–1799 | 1800 / 1800 | 3 / 7 | 100% | 0.641302 |

`coverage=100%` 的分母是表中的发布片段，不是未经裁剪的 source MP4。三段共投影 73,416、75,810、75,600 个 joint samples；positive-depth、inside-frame 和 wrist-inside-frame fraction 都为 1.0。这证明产物可看且没有投影飞出画面，不是独立像素精度。

## 5. Visual motion QA 与“不回写”规则

QA 信号定义：固定 240×135 图像中 `x=40:120,y=35:115` ROI 的相邻帧灰度绝对差均值，对比双手 42 joints 的投影 pixel speed 中位数；两者都做 7 帧 centered smoothing。

相关定义为：

\[
\operatorname{corr}(E_{video}[t],V_{mocap}(t_{camera}+\delta))
\]

| Take | legacy zero corr | legacy best shift / corr | corrected zero corr | corrected best residual / corr | zero-corr gain | QA |
|---|---:|---:|---:|---:|---:|---:|
| 01 | 0.626070 | -233 ms / 0.875550 | 0.876144 | +16 ms / 0.878987 | 0.250074 | PASS |
| 02 | 0.487623 | -252 ms / 0.894254 | 0.898664 | +10 ms / 0.901295 | 0.411041 | PASS |
| 03 | 0.394691 | -243 ms / 0.884211 | 0.881863 | +17 ms / 0.887205 | 0.487172 | PASS |

旧时刻的最佳负 shift 与逐行测得的约 255 ms callback latency 一致。修正后的 zero-shift correlation 均大于 0.80，gain 均大于 0.20，最佳残余 10–17 ms 均小于一个 30 FPS 帧的 33.333 ms。

**这 10–17 ms 只用于 QA，绝不写回时间映射，也不作为逐 Take 拟合的 offset。** 正式 overlay 始终使用第 1–3 节的独立 timestamp chain。原因是 30 FPS 图像、7 帧平滑、固定 ROI、压缩和投影误差都可能移动相关峰；当前又没有 shared exposure/OptiTrack TTL。逐 Take 回写视觉最优值会把目标动作泄漏到同步参数中，并制造不可审计的“看起来更贴”。

## 6. 30 FPS CFR 与 BAG 实际节拍边界

source metadata 和交付 H.264 都报告 30/1 FPS。渲染器逐帧写入 nominal 30 FPS MP4V，再转码为 30 FPS H.264 CFR；片段内部不丢帧、不补帧、不重采样。

但是 BAG device timestamps 的实际相邻间隔约 33.43 ms，而不是严格 33.333 ms：

| Take | BAG delta min/median/P95/max (ms) | effective BAG FPS | BAG span − `(N−1)/30` |
|---|---:|---:|---:|
| 01 | 33.420 / 33.430 / 33.438 / 33.444 | 29.912220560 | +170.889667 ms |
| 02 | 33.417 / 33.430 / 33.438 / 33.441 | 29.911733856 | +177.446667 ms |
| 03 | 33.419 / 33.430 / 33.438 / 33.444 | 29.912440849 | +175.533333 ms |

因此 H.264 的 CFR PTS 只用于方便播放；它不是 acquisition wall-clock。该限制不影响当前逐帧 overlay，因为每一帧查询 MOCAP 时使用的是 BAG device timestamp。任何下游数值对齐都必须使用 `.alignment.csv` 中的 `rgb_device_timestamp_us` / `cmavatar_target_timestamp_ns`，不能用 `output_frame/30` 反推绝对时间。

## 7. 空间标定与 GT 边界

当前空间链路来自 `movementcap.camera_to_mocap_world.v1`，方法为 `manual_visual_CS400_markers_plus_109_frame_raw_depth_median`；假设相机固定，camera-consistency check 已通过，桌面深度平面 residual P50/P95 为 0.767/3.912 mm。

当前可以宣称：

- 双手 MOCAP 21 点可按明确的 timestamp chain 投影到对应 RGB 帧；
- 裁剪后每个发布帧都有完整 bracket 和逐帧 CSV provenance；
- BAG↔MP4 全帧内容身份 gate 通过，所有可辨帧均选择 offset 0；
- 三段 contact sheet 中骨架均落在对应手部区域，适合人工 review。

当前不能宣称：

- 没有独立逐帧 2D hand landmarks，未测量 21 关节 pixel reprojection error；
- CS-400 marker identity 是人工选择，外参加入了 raw depth median，不是独立 held-out 全手标定；
- forward 3×3 block 轻微非正交，不能把它包装成严格 (SE(3))；
- `Human.cma` 没有 per-joint visibility、confidence、occlusion、gap-fill 或 residual，21 个节点也不能都称为直接实体 Marker 测量；
- 没有共享曝光/OptiTrack TTL，所以软件 clock uncertainty 和 visual QA 不能替代硬件同步真值；
- 同一 MOCAP 数据既生成 3D skeleton 又用于投影，不能把“落在手上”称为独立 GT accuracy；本产品也不评价 glove。
- 全帧内容 gate 不是原始编码 bit-exact，也不验证空间投影精度或硬件零时差。

## 8. 产物与哈希

每段生成：

```text
*_mocap_video_aligned_strict25.mp4
*_mocap_video_aligned_strict25_h264.mp4
*_mocap_video_aligned_strict25.contact.jpg
*_mocap_video_aligned_strict25.alignment.csv
*_mocap_video_aligned_strict25.metrics.json
```

三个 H.264 SHA-256：

| Take | SHA-256 |
|---|---|
| 01 | `c711286edd5c6e7f3188e44e5178a5ca6780428c8eed115f54c186436195e049` |
| 02 | `f32ca5aed08092da2ad9cb15243a87f87c810483c6f7b24fe66895a3e37c0a6e` |
| 03 | `de8298664ad3d5de1b3513be742655a8557b3055200798b463632971503e5611` |

三份 metrics 中声明的 MP4V、CSV、contact sheet、H.264 共 12 个 artifact hash 已独立重算，12/12 匹配。

## 9. 复现与验收

从空输出目录完整重建的复现命令：

```bash
cd /home/runyi/Project/hands_reloc/GT_calib

/home/runyi/miniconda3/envs/viewer/bin/python mocap_video_overlay.py \
  --segment 01 --segment 02 --segment 03 \
  --output-dir outputs/mocap_video_alignment_review \
  --frame-content-samples 0 \
  --output-width 960
```

正式发布显式使用 `--frame-content-samples 0`；不传该参数时默认 31 帧，仅适合快速回归。本次最终全帧审计是在已验收媒体上追加 `--analyze-only` 执行，保留并复核既有 MP4V、contact、H.264 artifact hash；wall time 为 4:58.38，峰值 RSS 360732 KB。未显式填写但由脚本默认冻结的其他参数是 `--max-interpolation-gap-ms 25`、`--snapshot-count 6`，并启用 H.264。

H.264 验收结果：

| Take | codec/profile | size/pixel format | FPS | frames | duration |
|---|---|---|---:|---:|---:|
| 01 | H.264 High | 960×540 / yuv420p | 30 | 1748 | 58.266667 s |
| 02 | H.264 High | 960×540 / yuv420p | 30 | 1805 | 60.166667 s |
| 03 | H.264 High | 960×540 / yuv420p | 30 | 1800 | 60.000000 s |

独立全帧 decode 使用：

```bash
for f in outputs/mocap_video_alignment_review/*_h264.mp4; do
  set -o pipefail
  ffmpeg -v error -i "$f" -map 0:v:0 -f framemd5 - \
    | awk 'BEGIN { n=0 } /^[0-9]/ { n++ } END { print FILENAME, n }'
done
```

三段均 exit 0，decoded frames 恰为 1748/1805/1800，并与 ffprobe、metrics 和 CSV data rows 一致。最终 temporal acceptance 六类条件全部通过：发布帧全 valid、external clock uncertainty ≤1 ms、bracket max ≤25 ms、timecode rows 精确命中 BAG timestamps、全帧内容身份 gate pass、visual motion QA pass。
