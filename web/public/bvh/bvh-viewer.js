"use strict";

const MANIFEST_URL = "/bvh/data/manifest.json";
const MANIFEST_SCHEMA = "gt-calib-bvh-web-manifest-v1";
const EXPORT_SCHEMA = "gt-calib-bvh-web-v1";

const TAKE_ASSETS = Object.freeze({
  "01": {
    id: "take-000",
    video: "/media/take-000-mocap.mp4",
    poster: "/media/take-000-mocap-contact.jpg",
    left: "/bvh/data/take-000-skeleton-0.bvh",
    right: "/bvh/data/take-000-skeleton-1.bvh",
  },
  "02": {
    id: "take-001",
    video: "/media/take-001-mocap.mp4",
    poster: "/media/take-001-mocap-contact.jpg",
    left: "/bvh/data/take-001-skeleton-0.bvh",
    right: "/bvh/data/take-001-skeleton-1.bvh",
  },
  "03": {
    id: "take-002",
    video: "/media/take-002-mocap.mp4",
    poster: "/media/take-002-mocap-contact.jpg",
    left: "/bvh/data/take-002-skeleton-0.bvh",
    right: "/bvh/data/take-002-skeleton-1.bvh",
  },
});

const COLORS = Object.freeze({
  left: "#f06be6",
  right: "#42deef",
  grid: "rgba(129, 174, 208, 0.105)",
  gridStrong: "rgba(142, 191, 225, 0.22)",
  outline: "rgba(2, 7, 13, 0.78)",
});

const DEFAULT_VIEW = Object.freeze({ yaw: -0.34, pitch: -0.62, zoom: 1 });

const state = {
  manifest: null,
  entry: null,
  motion: null,
  takeKey: null,
  abortController: null,
  generation: 0,
  outputFrame: 0,
  sourceTarget: null,
  sourceBracket: null,
  positionBuffer: new Float32Array(126),
  view: { ...DEFAULT_VIEW },
  drag: null,
  fallbackRaf: 0,
};

const dom = {
  takeButtons: Array.from(document.querySelectorAll("[data-take]")),
  loadState: document.querySelector("#load-state"),
  loadStateText: document.querySelector("#load-state-text"),
  canvas: document.querySelector("#motion-canvas"),
  canvasWrap: document.querySelector("#canvas-wrap"),
  canvasLoading: document.querySelector("#canvas-loading"),
  resetView: document.querySelector("#reset-view"),
  video: document.querySelector("#master-video"),
  videoOverlay: document.querySelector("#video-play-overlay"),
  playToggle: document.querySelector("#play-toggle"),
  playIcon: document.querySelector("#play-icon"),
  playLabel: document.querySelector("#play-label"),
  timeline: document.querySelector("#timeline"),
  timeReadout: document.querySelector("#time-readout"),
  frameReadout: document.querySelector("#frame-readout"),
  playbackRate: document.querySelector("#playback-rate"),
  hudOutputFrame: document.querySelector("#hud-output-frame"),
  hudSourceFrame: document.querySelector("#hud-source-frame"),
  metricVideoFrame: document.querySelector("#metric-video-frame"),
  metricBvhFrame: document.querySelector("#metric-bvh-frame"),
  metricBracket: document.querySelector("#metric-bracket"),
  metricAlpha: document.querySelector("#metric-alpha"),
  boundaryCopy: document.querySelector("#boundary-copy"),
  strictRange: document.querySelector("#strict-range"),
  downloadLeft: document.querySelector("#download-left"),
  downloadRight: document.querySelector("#download-right"),
  downloadMotion: document.querySelector("#download-motion"),
};

const context = dom.canvas.getContext("2d", { alpha: false });

function invariant(condition, message) {
  if (!condition) throw new Error(message);
}

function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

function setLoadState(kind, message) {
  dom.loadState.dataset.kind = kind;
  dom.loadStateText.textContent = message;
}

function setControlsEnabled(enabled) {
  dom.playToggle.disabled = !enabled;
  dom.timeline.disabled = !enabled;
  dom.playbackRate.disabled = !enabled;
}

function normaliseTakeKey(value) {
  const number = Number(value);
  if (Number.isInteger(number) && number >= 1 && number <= 3) {
    return String(number).padStart(2, "0");
  }
  return "01";
}

function hexFromBuffer(buffer) {
  return Array.from(new Uint8Array(buffer), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function fetchJson(url, { signal, expectedSha256 } = {}) {
  const response = await fetch(url, { cache: "no-store", signal });
  if (!response.ok) throw new Error(`${url} 返回 HTTP ${response.status}`);
  const bytes = await response.arrayBuffer();
  if (expectedSha256 && globalThis.crypto?.subtle) {
    const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
    invariant(hexFromBuffer(digest) === expectedSha256, `${url} SHA-256 不匹配`);
  }
  return JSON.parse(new TextDecoder().decode(bytes));
}

function validateManifest(manifest) {
  invariant(manifest?.schema === MANIFEST_SCHEMA, "BVH manifest schema 不匹配");
  invariant(manifest.export_schema === EXPORT_SCHEMA, "BVH export schema 不匹配");
  invariant(manifest.validation?.status === "pass", "BVH manifest validation 未通过");
  invariant(manifest.validation.all_requested_segments_exported === true, "BVH 数据段不完整");
  invariant(Array.isArray(manifest.takes) && manifest.takes.length === 3, "BVH manifest 必须包含三个 Take");
  const keys = manifest.takes.map((take) => take.segment_key);
  invariant(new Set(keys).size === 3 && keys.every((key) => TAKE_ASSETS[key]), "BVH Take key 非法");
}

function validateMotion(payload, entry) {
  invariant(payload?.schema === EXPORT_SCHEMA, "动作 JSON schema 不匹配");
  invariant(payload.segment_key === entry.segment_key, "动作 JSON segment_key 与 manifest 不一致");
  invariant(payload.segment === entry.segment, "动作 JSON segment 与 manifest 不一致");
  invariant(payload.validation?.status === "pass", "动作 JSON validation 未通过");
  invariant(payload.validation.finite_motion === true, "动作数据包含非有限值");
  invariant(payload.validation.left_right_frame_contract === true, "左右手帧契约未通过");

  const skeletons = payload.skeletons;
  invariant(Array.isArray(skeletons) && skeletons.length === 2, "必须恰好包含左右两只手");
  invariant(skeletons[0].side === "left" && skeletons[0].source_skeleton === "Skeleton_0", "Skeleton_0 必须是左手");
  invariant(skeletons[1].side === "right" && skeletons[1].source_skeleton === "Skeleton_1", "Skeleton_1 必须是右手");
  for (const skeleton of skeletons) {
    invariant(skeleton.joint_count === 21, `${skeleton.side} 必须有 21 joints`);
    invariant(Array.isArray(skeleton.joints) && skeleton.joints.length === 21, `${skeleton.side} joint metadata 不完整`);
    invariant(Array.isArray(skeleton.edges) && skeleton.edges.length === 20, `${skeleton.side} 必须有 20 bones`);
  }

  const motion = payload.motion;
  const positions = payload.positions;
  const sync = payload.video_sync;
  invariant(Array.isArray(motion.source_frame_indices), "缺少 source_frame_indices");
  invariant(motion.source_frame_indices.length === motion.exported_frame_count, "source_frame_indices 数量错误");
  invariant(motion.source_frame_indices.length === positions.frames.length, "位置帧数量与 source 索引不一致");
  invariant(motion.source_frame_indices.length >= 2, "动作帧不足");
  invariant(motion.source_frame_indices.every((value, index, values) => Number.isInteger(value) && (index === 0 || value > values[index - 1])), "source_frame_indices 必须严格递增");
  invariant(motion.source_frame_indices[0] === motion.strict_source_frame_first, "strict 首帧未保留");
  invariant(motion.source_frame_indices.at(-1) === motion.strict_source_frame_last, "strict 尾帧未保留");

  invariant(positions.encoding === "frame-major-flat-int32-json", "位置编码不支持");
  invariant(positions.components === "XYZ" && positions.total_joint_count === 42, "位置拓扑必须为 42×XYZ");
  invariant(positions.values_per_frame === 126, "每帧必须包含 126 个数值");
  invariant(Number.isFinite(positions.quantum_bvh_units) && positions.quantum_bvh_units > 0, "quantum 非法");
  invariant(Array.isArray(positions.origin_bvh_units) && positions.origin_bvh_units.length === 3, "origin 非法");
  invariant(positions.frames.every((frame) => Array.isArray(frame) && frame.length === 126), "量化位置帧长度错误");
  invariant(Array.isArray(positions.bounds_bvh_units?.min) && Array.isArray(positions.bounds_bvh_units?.max), "位置 bounds 缺失");

  const mapping = sync.bvh_fractional_source_frame_by_output_frame;
  invariant(sync.master === "mocap_h264_overlay_video", "视频 master 契约不支持");
  invariant(Array.isArray(mapping) && mapping.length === sync.overlay_output_frame_count, "视频映射长度错误");
  invariant(sync.overlay_output_frame_count === entry.overlay_output_frame_count, "视频输出帧数与 manifest 不一致");
  invariant(sync.validation?.status === "pass", "视频同步 validation 未通过");
  invariant(sync.validation.alignment_rows_match_overlay_output_frames === true, "视频映射行数错误");
  invariant(sync.validation.dummy_seed_mappings_explicitly_null === true, "dummy seed 必须显式为 null");
  invariant(sync.validation.fractional_bvh_frames_strictly_increasing === true, "BVH fractional mapping 必须严格递增");
  invariant(Number.isInteger(sync.first_valid_output_frame), "缺少 first_valid_output_frame");
  invariant(mapping[sync.first_valid_output_frame] !== null, "first_valid_output_frame 仍是 null");
  invariant(mapping.every((value, index) => value === null || (Number.isFinite(value) && (index === 0 || mapping[index - 1] === null || value > mapping[index - 1]))), "视频映射值非法");
}

function selectTakeButton(takeKey) {
  for (const button of dom.takeButtons) {
    const selected = button.dataset.take === takeKey;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
  }
}

function configureDownloads(takeKey, entry) {
  const assets = TAKE_ASSETS[takeKey];
  dom.downloadLeft.href = assets.left;
  dom.downloadLeft.download = `${assets.id}-skeleton-0.bvh`;
  dom.downloadRight.href = assets.right;
  dom.downloadRight.download = `${assets.id}-skeleton-1.bvh`;
  dom.downloadMotion.href = `/bvh/data/${entry.file}`;
  dom.downloadMotion.download = entry.file;
}

function updateBoundaryCopy(payload) {
  const motion = payload.motion;
  const sync = payload.video_sync;
  const unavailable = sync.unavailable_output_frames;
  const skipped = unavailable.length
    ? `视频输出帧 ${unavailable.join("、")} 的 bracket 碰到 dummy，因此播放器自动从 ${sync.first_valid_output_frame} 开始。`
    : "当前视频从第一帧起就具有严格 BVH pose。";
  dom.boundaryCopy.textContent = `BVH source frame 0 是厂商全零 seed，source frame ${motion.source_frame_count - 1} 没有 Human.cma 对应项；两者都未进入导出位置。${skipped}`;
  dom.strictRange.textContent = `${motion.strict_source_frame_first} … ${motion.strict_source_frame_last}`;
}

function resetMetrics() {
  dom.metricVideoFrame.textContent = "—";
  dom.metricBvhFrame.textContent = "—";
  dom.metricBracket.textContent = "—";
  dom.metricAlpha.textContent = "—";
  dom.hudOutputFrame.textContent = "VIDEO —";
  dom.hudSourceFrame.textContent = "BVH —";
  dom.frameReadout.textContent = "frame —";
  dom.timeReadout.textContent = "00:00.000 / 00:00.000";
}

async function loadTake(takeKey) {
  if (!state.manifest) return;
  const entry = state.manifest.takes.find((candidate) => candidate.segment_key === takeKey);
  if (!entry) return;

  state.abortController?.abort();
  state.abortController = new AbortController();
  const generation = ++state.generation;
  state.takeKey = takeKey;
  state.entry = entry;
  state.motion = null;
  state.sourceTarget = null;
  state.sourceBracket = null;
  dom.video.pause();
  dom.video.removeAttribute("src");
  dom.video.load();
  setControlsEnabled(false);
  selectTakeButton(takeKey);
  configureDownloads(takeKey, entry);
  resetMetrics();
  dom.canvasLoading.hidden = false;
  dom.canvasLoading.querySelector("p").textContent = `正在载入 Take ${takeKey} 动作数据`;
  setLoadState("loading", `Take ${takeKey} · 正在校验 ${(entry.byte_size / 1048576).toFixed(2)} MiB 动作数据…`);

  const url = new URL(window.location.href);
  url.searchParams.set("take", takeKey);
  history.replaceState(null, "", url);

  try {
    const payload = await fetchJson(`/bvh/data/${entry.file}`, {
      signal: state.abortController.signal,
      expectedSha256: entry.sha256,
    });
    if (generation !== state.generation) return;
    validateMotion(payload, entry);
    state.motion = payload;
    state.outputFrame = payload.video_sync.first_valid_output_frame;
    resetView();
    updateBoundaryCopy(payload);

    const first = payload.video_sync.first_valid_output_frame;
    const last = payload.video_sync.overlay_output_frame_count - 1;
    dom.timeline.min = String(first);
    dom.timeline.max = String(last);
    dom.timeline.value = String(first);
    dom.playbackRate.value = "1";
    dom.video.playbackRate = 1;

    const assets = TAKE_ASSETS[takeKey];
    dom.video.poster = assets.poster;
    dom.video.src = assets.video;
    dom.video.load();
    setControlsEnabled(true);
    dom.canvasLoading.hidden = true;
    renderOutputFrame(first);
    setLoadState(
      "ready",
      `Take ${takeKey} 已校验 · ${payload.motion.exported_frame_count.toLocaleString()} samples · 42 joints / 40 bones`,
    );
    armVideoFrameCallback(generation);
  } catch (error) {
    if (error.name === "AbortError" || generation !== state.generation) return;
    console.error(error);
    setLoadState("error", `Take ${takeKey} 加载失败：${error.message}`);
    dom.canvasLoading.hidden = false;
    dom.canvasLoading.querySelector("p").textContent = "动作数据加载失败";
    setControlsEnabled(false);
  }
}

// Find the exported position samples bracketing a fractional raw BVH source
// frame. This deliberately searches motion.source_frame_indices instead of
// assuming the web export has a perfectly uniform stride at its final sample.
function sourceSampleBracket(sourceTarget) {
  const indices = state.motion.motion.source_frame_indices;
  if (sourceTarget <= indices[0]) return { low: 0, high: 0, alpha: 0 };
  if (sourceTarget >= indices.at(-1)) {
    const end = indices.length - 1;
    return { low: end, high: end, alpha: 0 };
  }

  let left = 0;
  let right = indices.length - 1;
  while (left + 1 < right) {
    const middle = (left + right) >> 1;
    if (indices[middle] <= sourceTarget) left = middle;
    else right = middle;
  }
  const span = indices[right] - indices[left];
  return {
    low: left,
    high: right,
    alpha: span > 0 ? (sourceTarget - indices[left]) / span : 0,
  };
}

function interpolatePositions(sourceTarget) {
  const payload = state.motion;
  const bracket = sourceSampleBracket(sourceTarget);
  const lowValues = payload.positions.frames[bracket.low];
  const highValues = payload.positions.frames[bracket.high];
  const quantum = payload.positions.quantum_bvh_units;
  const origin = payload.positions.origin_bvh_units;
  const alpha = bracket.alpha;
  const output = state.positionBuffer;

  for (let index = 0; index < output.length; index += 1) {
    const quantized = lowValues[index] + (highValues[index] - lowValues[index]) * alpha;
    output[index] = origin[index % 3] + quantized * quantum;
  }
  state.sourceBracket = bracket;
  return output;
}

function nearestValidOutputFrame(requested) {
  const mapping = state.motion.video_sync.bvh_fractional_source_frame_by_output_frame;
  let frame = clamp(Math.round(requested), 0, mapping.length - 1);
  if (mapping[frame] !== null) return frame;
  for (let offset = 1; offset < mapping.length; offset += 1) {
    if (frame + offset < mapping.length && mapping[frame + offset] !== null) return frame + offset;
    if (frame - offset >= 0 && mapping[frame - offset] !== null) return frame - offset;
  }
  throw new Error("视频映射没有有效 BVH pose");
}

function renderOutputFrame(requestedFrame) {
  if (!state.motion) return;
  const frame = nearestValidOutputFrame(requestedFrame);
  const sync = state.motion.video_sync;
  const sourceTarget = sync.bvh_fractional_source_frame_by_output_frame[frame];
  invariant(sourceTarget !== null, "尝试渲染 dummy BVH pose");
  state.outputFrame = frame;
  state.sourceTarget = sourceTarget;
  interpolatePositions(sourceTarget);
  drawScene();
  updateReadouts();
}

function updateReadouts() {
  const sync = state.motion.video_sync;
  const indices = state.motion.motion.source_frame_indices;
  const bracket = state.sourceBracket;
  const sourceLow = indices[bracket.low];
  const sourceHigh = indices[bracket.high];
  const fps = sync.overlay_nominal_fps;
  const totalSeconds = sync.overlay_output_frame_count / fps;
  const currentSeconds = state.outputFrame / fps;

  dom.timeline.value = String(state.outputFrame);
  dom.timeReadout.textContent = `${formatTime(currentSeconds)} / ${formatTime(totalSeconds)}`;
  dom.frameReadout.textContent = `frame ${state.outputFrame} / ${sync.overlay_output_frame_count - 1}`;
  dom.hudOutputFrame.textContent = `VIDEO ${String(state.outputFrame).padStart(4, "0")}`;
  dom.hudSourceFrame.textContent = `BVH ${state.sourceTarget.toFixed(3)}`;
  dom.metricVideoFrame.textContent = String(state.outputFrame);
  dom.metricBvhFrame.textContent = state.sourceTarget.toFixed(3);
  dom.metricBracket.textContent = sourceLow === sourceHigh ? String(sourceLow) : `${sourceLow} → ${sourceHigh}`;
  dom.metricAlpha.textContent = bracket.alpha.toFixed(4);
}

function formatTime(seconds) {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const minutes = Math.floor(safe / 60);
  const remainder = safe - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(3).padStart(6, "0")}`;
}

function outputFrameAtMediaTime(mediaTime) {
  const fps = state.motion.video_sync.overlay_nominal_fps;
  return nearestValidOutputFrame(Math.round(mediaTime * fps));
}

function renderAtMediaTime(mediaTime) {
  if (!state.motion || !Number.isFinite(mediaTime)) return;
  renderOutputFrame(outputFrameAtMediaTime(mediaTime));
}

function armVideoFrameCallback(generation) {
  if (!("requestVideoFrameCallback" in dom.video)) return;
  dom.video.requestVideoFrameCallback((_now, metadata) => {
    if (generation !== state.generation) return;
    renderAtMediaTime(metadata.mediaTime);
    armVideoFrameCallback(generation);
  });
}

function startFallbackLoop() {
  cancelAnimationFrame(state.fallbackRaf);
  if ("requestVideoFrameCallback" in dom.video) return;
  const tick = () => {
    renderAtMediaTime(dom.video.currentTime);
    if (!dom.video.paused && !dom.video.ended) state.fallbackRaf = requestAnimationFrame(tick);
  };
  state.fallbackRaf = requestAnimationFrame(tick);
}

function seekToOutputFrame(requestedFrame) {
  if (!state.motion) return;
  const frame = nearestValidOutputFrame(requestedFrame);
  const fps = state.motion.video_sync.overlay_nominal_fps;
  renderOutputFrame(frame);
  if (Number.isFinite(dom.video.duration)) dom.video.currentTime = frame / fps;
}

async function togglePlayback() {
  if (!state.motion) return;
  if (dom.video.paused || dom.video.ended) {
    if (dom.video.ended || state.outputFrame >= state.motion.video_sync.overlay_output_frame_count - 1) {
      seekToOutputFrame(state.motion.video_sync.first_valid_output_frame);
    }
    try {
      await dom.video.play();
    } catch (error) {
      setLoadState("error", `浏览器阻止播放：${error.message}`);
    }
  } else {
    dom.video.pause();
  }
}

function syncPlaybackUi() {
  const playing = !dom.video.paused && !dom.video.ended;
  dom.playIcon.textContent = playing ? "Ⅱ" : "▶";
  dom.playLabel.textContent = playing ? "暂停" : "播放";
  dom.videoOverlay.hidden = playing;
  dom.videoOverlay.setAttribute("aria-label", playing ? "暂停" : "播放");
  if (playing) startFallbackLoop();
}

function resetView() {
  state.view = { ...DEFAULT_VIEW };
  drawScene();
}

function ensureCanvasResolution() {
  const rect = dom.canvas.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (dom.canvas.width !== width || dom.canvas.height !== height) {
    dom.canvas.width = width;
    dom.canvas.height = height;
  }
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { width: rect.width, height: rect.height, dpr };
}

function viewGeometry(width, height) {
  const bounds = state.motion.positions.bounds_bvh_units;
  const min = bounds.min;
  const max = bounds.max;
  const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  const diagonal = Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
  const scale = (Math.min(width, height) * 0.79 * state.view.zoom) / Math.max(diagonal, 1);
  return { bounds, center, scale };
}

function projectPoint(x, y, z, width, height, geometry) {
  const px = x - geometry.center[0];
  const py = y - geometry.center[1];
  const pz = z - geometry.center[2];
  const cy = Math.cos(state.view.yaw);
  const sy = Math.sin(state.view.yaw);
  const cp = Math.cos(state.view.pitch);
  const sp = Math.sin(state.view.pitch);
  const yawX = cy * px + sy * pz;
  const yawZ = -sy * px + cy * pz;
  const pitchY = cp * py - sp * yawZ;
  const pitchZ = sp * py + cp * yawZ;
  return {
    x: width / 2 + yawX * geometry.scale,
    y: height / 2 - pitchY * geometry.scale,
    z: pitchZ,
  };
}

function drawGrid(width, height, geometry) {
  const min = geometry.bounds.min;
  const max = geometry.bounds.max;
  const y = min[1] - (max[1] - min[1]) * 0.08;
  context.save();
  context.lineWidth = 1;
  for (let step = 0; step <= 10; step += 1) {
    const ratio = step / 10;
    const x = min[0] + (max[0] - min[0]) * ratio;
    const z = min[2] + (max[2] - min[2]) * ratio;
    const x0 = projectPoint(x, y, min[2], width, height, geometry);
    const x1 = projectPoint(x, y, max[2], width, height, geometry);
    const z0 = projectPoint(min[0], y, z, width, height, geometry);
    const z1 = projectPoint(max[0], y, z, width, height, geometry);
    context.strokeStyle = step === 5 ? COLORS.gridStrong : COLORS.grid;
    context.beginPath();
    context.moveTo(x0.x, x0.y);
    context.lineTo(x1.x, x1.y);
    context.stroke();
    context.beginPath();
    context.moveTo(z0.x, z0.y);
    context.lineTo(z1.x, z1.y);
    context.stroke();
  }
  context.restore();
}

function drawScene() {
  const { width, height } = ensureCanvasResolution();
  context.clearRect(0, 0, width, height);
  const gradient = context.createRadialGradient(width * 0.5, height * 0.43, 0, width * 0.5, height * 0.5, Math.max(width, height) * 0.72);
  gradient.addColorStop(0, "#10263a");
  gradient.addColorStop(0.48, "#081624");
  gradient.addColorStop(1, "#040a12");
  context.fillStyle = gradient;
  context.fillRect(0, 0, width, height);
  if (!state.motion || state.sourceTarget === null) return;

  const geometry = viewGeometry(width, height);
  drawGrid(width, height, geometry);
  const projected = [];
  for (let joint = 0; joint < 42; joint += 1) {
    const offset = joint * 3;
    projected.push(projectPoint(
      state.positionBuffer[offset],
      state.positionBuffer[offset + 1],
      state.positionBuffer[offset + 2],
      width,
      height,
      geometry,
    ));
  }

  const bones = [];
  for (const skeleton of state.motion.skeletons) {
    const jointOffset = skeleton.position_joint_offset;
    for (const [parent, child] of skeleton.edges) {
      const a = jointOffset + parent;
      const b = jointOffset + child;
      bones.push({ a, b, side: skeleton.side, depth: (projected[a].z + projected[b].z) / 2 });
    }
  }
  bones.sort((a, b) => a.depth - b.depth);

  context.lineCap = "round";
  context.lineJoin = "round";
  for (const bone of bones) {
    const a = projected[bone.a];
    const b = projected[bone.b];
    context.beginPath();
    context.moveTo(a.x, a.y);
    context.lineTo(b.x, b.y);
    context.lineWidth = 7;
    context.strokeStyle = COLORS.outline;
    context.stroke();
    context.beginPath();
    context.moveTo(a.x, a.y);
    context.lineTo(b.x, b.y);
    context.lineWidth = 3.25;
    context.strokeStyle = COLORS[bone.side];
    context.shadowColor = COLORS[bone.side];
    context.shadowBlur = 5;
    context.stroke();
    context.shadowBlur = 0;
  }

  const nodes = projected.map((point, index) => ({ point, index })).sort((a, b) => a.point.z - b.point.z);
  for (const { point, index } of nodes) {
    const local = index % 21;
    const side = index < 21 ? "left" : "right";
    const tip = [4, 8, 12, 16, 20].includes(local);
    const radius = local === 0 ? 5.2 : tip ? 4 : 3;
    context.beginPath();
    context.arc(point.x, point.y, radius + 1.8, 0, Math.PI * 2);
    context.fillStyle = COLORS.outline;
    context.fill();
    context.beginPath();
    context.arc(point.x, point.y, radius, 0, Math.PI * 2);
    context.fillStyle = COLORS[side];
    context.shadowColor = COLORS[side];
    context.shadowBlur = tip || local === 0 ? 10 : 5;
    context.fill();
    context.shadowBlur = 0;
  }

  drawAxisWidget(width, height);
}

function drawAxisWidget(width, height) {
  const originX = width - 42;
  const originY = height - 42;
  const length = 23;
  const axes = [
    { label: "X", vector: [1, 0, 0], color: "#ff7e89" },
    { label: "Y", vector: [0, 1, 0], color: "#7de7ad" },
    { label: "Z", vector: [0, 0, 1], color: "#79a9ff" },
  ];
  const cy = Math.cos(state.view.yaw);
  const sy = Math.sin(state.view.yaw);
  const cp = Math.cos(state.view.pitch);
  const sp = Math.sin(state.view.pitch);
  context.save();
  context.font = "600 9px ui-monospace, monospace";
  context.lineWidth = 1.5;
  for (const axis of axes) {
    const [x, y, z] = axis.vector;
    const yawX = cy * x + sy * z;
    const yawZ = -sy * x + cy * z;
    const pitchY = cp * y - sp * yawZ;
    const endX = originX + yawX * length;
    const endY = originY - pitchY * length;
    context.beginPath();
    context.moveTo(originX, originY);
    context.lineTo(endX, endY);
    context.strokeStyle = axis.color;
    context.stroke();
    context.fillStyle = axis.color;
    context.fillText(axis.label, endX + 3, endY + 3);
  }
  context.restore();
}

function wireInteraction() {
  for (const button of dom.takeButtons) {
    button.addEventListener("click", () => loadTake(button.dataset.take));
    button.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const index = dom.takeButtons.indexOf(button);
      let next = index;
      if (event.key === "ArrowLeft") next = (index + dom.takeButtons.length - 1) % dom.takeButtons.length;
      if (event.key === "ArrowRight") next = (index + 1) % dom.takeButtons.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = dom.takeButtons.length - 1;
      dom.takeButtons[next].focus();
      loadTake(dom.takeButtons[next].dataset.take);
    });
  }

  dom.playToggle.addEventListener("click", togglePlayback);
  dom.videoOverlay.addEventListener("click", togglePlayback);
  dom.video.addEventListener("click", togglePlayback);
  dom.video.addEventListener("play", syncPlaybackUi);
  dom.video.addEventListener("pause", syncPlaybackUi);
  dom.video.addEventListener("ended", syncPlaybackUi);
  dom.video.addEventListener("loadedmetadata", () => {
    if (!state.motion) return;
    // Preserve a scrub made while the MP4 metadata was still loading. On a
    // fresh Take state.outputFrame already equals first_valid_output_frame.
    seekToOutputFrame(state.outputFrame);
  });
  dom.video.addEventListener("loadeddata", () => renderAtMediaTime(dom.video.currentTime));
  dom.video.addEventListener("seeked", () => renderAtMediaTime(dom.video.currentTime));
  dom.video.addEventListener("timeupdate", () => renderAtMediaTime(dom.video.currentTime));
  dom.video.addEventListener("error", () => {
    if (!state.motion) return;
    const mediaError = dom.video.error;
    setLoadState("error", `参考视频加载失败${mediaError ? `（code ${mediaError.code}）` : ""}；BVH 仍可拖动时间轴审阅。`);
  });

  dom.timeline.addEventListener("input", () => seekToOutputFrame(Number(dom.timeline.value)));
  dom.playbackRate.addEventListener("change", () => {
    dom.video.playbackRate = Number(dom.playbackRate.value);
  });
  dom.resetView.addEventListener("click", resetView);

  dom.canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    dom.canvas.setPointerCapture(event.pointerId);
    dom.canvas.dataset.dragging = "true";
    state.drag = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
  });
  dom.canvas.addEventListener("pointermove", (event) => {
    if (!state.drag || state.drag.pointerId !== event.pointerId) return;
    const dx = event.clientX - state.drag.x;
    const dy = event.clientY - state.drag.y;
    state.drag.x = event.clientX;
    state.drag.y = event.clientY;
    state.view.yaw += dx * 0.008;
    state.view.pitch = clamp(state.view.pitch + dy * 0.008, -1.45, 1.45);
    drawScene();
  });
  const finishDrag = (event) => {
    if (!state.drag || state.drag.pointerId !== event.pointerId) return;
    state.drag = null;
    dom.canvas.dataset.dragging = "false";
  };
  dom.canvas.addEventListener("pointerup", finishDrag);
  dom.canvas.addEventListener("pointercancel", finishDrag);
  dom.canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    state.view.zoom = clamp(state.view.zoom * Math.exp(-event.deltaY * 0.0012), 0.45, 4);
    drawScene();
  }, { passive: false });
  dom.canvas.addEventListener("dblclick", resetView);

  new ResizeObserver(drawScene).observe(dom.canvasWrap);
  window.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement || event.target instanceof HTMLButtonElement) return;
    if (event.code === "Space") {
      event.preventDefault();
      togglePlayback();
    } else if (event.key === "ArrowLeft" && state.motion) {
      seekToOutputFrame(state.outputFrame - 1);
    } else if (event.key === "ArrowRight" && state.motion) {
      seekToOutputFrame(state.outputFrame + 1);
    }
  });
}

async function main() {
  invariant(context, "Canvas 2D context 不可用");
  wireInteraction();
  drawScene();
  try {
    const manifest = await fetchJson(MANIFEST_URL);
    validateManifest(manifest);
    state.manifest = manifest;
    const requested = normaliseTakeKey(new URL(window.location.href).searchParams.get("take"));
    await loadTake(requested);
  } catch (error) {
    console.error(error);
    setLoadState("error", `Manifest 加载失败：${error.message}`);
    dom.canvasLoading.querySelector("p").textContent = "本地 BVH manifest 不可用";
  }
}

main();
