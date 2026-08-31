"use strict";

const DELIVERY_ROOT = "/downloads/final-nine";
const EXPECTED_SCHEMA = "gt_calib.final_nine_video_delivery.v1";
const cards = Array.from(document.querySelectorAll("[data-video-id]"));

function safeRelativePath(value) {
  if (typeof value !== "string" || !value || value.startsWith("/") || value.includes("..") || value.includes(":")) {
    throw new Error(`不安全的 manifest 路径: ${String(value)}`);
  }
  return `${DELIVERY_ROOT}/${value}`;
}

function formatBytes(bytes) {
  const units = ["B", "KB", "MB", "GB"];
  let value = Number(bytes);
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatDuration(seconds) {
  const rounded = Math.round(Number(seconds));
  const minutes = Math.floor(rounded / 60);
  const remaining = rounded % 60;
  return `${minutes}:${String(remaining).padStart(2, "0")}`;
}

function setManifestState(kind, message) {
  const state = document.querySelector("#manifest-status");
  if (!state) return;
  state.classList.remove("verified", "failed");
  if (kind) state.classList.add(kind);
  const label = state.querySelector("span:last-child");
  if (label) label.textContent = message;
}

function hydrateCard(card, entry) {
  const video = card.querySelector("video");
  const source = video?.querySelector("source");
  const download = card.querySelector("[data-download]");
  const metrics = card.querySelector("[data-metrics]");
  const videoPath = safeRelativePath(`videos/${entry.filename}`);
  const posterPath = safeRelativePath(entry.poster);

  if (video) video.poster = posterPath;
  if (source && source.getAttribute("src") !== videoPath) {
    source.setAttribute("src", videoPath);
    video?.load();
  }
  if (download) download.href = videoPath;
  if (metrics) metrics.href = safeRelativePath(entry.metrics);

  const frames = card.querySelector('[data-field="frames"]');
  const duration = card.querySelector('[data-field="duration"]');
  const bytes = card.querySelector('[data-field="bytes"]');
  const sha = card.querySelector('[data-field="sha"]');
  if (frames) frames.textContent = `${entry.frame_count} frames`;
  if (duration) duration.textContent = `${Number(entry.duration_s).toFixed(2)} s`;
  if (bytes) bytes.textContent = formatBytes(entry.bytes);
  if (sha) {
    sha.textContent = `SHA ${entry.sha256.slice(0, 10)}…`;
    sha.title = entry.sha256;
  }
}

function validateManifest(manifest) {
  if (manifest?.schema !== EXPECTED_SCHEMA) throw new Error("manifest schema 不匹配");
  if (manifest?.status !== "pass") throw new Error("manifest 状态不是 pass");
  if (manifest?.video_count !== 9 || !Array.isArray(manifest?.videos) || manifest.videos.length !== 9) {
    throw new Error("manifest 必须恰好包含 9 段视频");
  }
  const manifestIds = new Set(manifest.videos.map((entry) => entry.id));
  const authoredIds = new Set(cards.map((card) => card.dataset.videoId));
  if (manifestIds.size !== 9 || authoredIds.size !== 9 || [...authoredIds].some((id) => !manifestIds.has(id))) {
    throw new Error("页面卡片与 manifest 视频 ID 不一致");
  }
}

async function loadManifest() {
  try {
    const response = await fetch(`${DELIVERY_ROOT}/manifest.json`, { cache: "no-store" });
    if (!response.ok) throw new Error(`manifest HTTP ${response.status}`);
    const manifest = await response.json();
    validateManifest(manifest);
    const byId = new Map(manifest.videos.map((entry) => [entry.id, entry]));
    for (const card of cards) hydrateCard(card, byId.get(card.dataset.videoId));

    const totalDuration = manifest.videos.reduce((sum, entry) => sum + Number(entry.duration_s), 0);
    const totalBytes = manifest.videos.reduce((sum, entry) => sum + Number(entry.bytes), 0);
    const videos = document.querySelector("#stat-videos");
    const duration = document.querySelector("#stat-duration");
    if (videos) videos.textContent = String(manifest.video_count);
    if (duration) duration.textContent = formatDuration(totalDuration);
    setManifestState("verified", `manifest 已验证 · 9/9 · ${formatBytes(totalBytes)}`);
  } catch (error) {
    setManifestState("failed", `manifest 未加载 · ${error.message}`);
  }
}

function videosForPair(pair) {
  return Array.from(document.querySelectorAll(`[data-pair="${pair}"] video`));
}

function wirePairControls() {
  for (const button of document.querySelectorAll("[data-sync-play]")) {
    button.addEventListener("click", async () => {
      const videos = videosForPair(button.dataset.syncPlay);
      if (videos.length !== 2) return;
      const shouldPlay = videos.some((video) => video.paused);
      if (shouldPlay) {
        const targetTime = Math.min(...videos.map((video) => Number.isFinite(video.currentTime) ? video.currentTime : 0));
        for (const video of videos) video.currentTime = targetTime;
        await Promise.allSettled(videos.map((video) => video.play()));
        button.textContent = "同步暂停";
      } else {
        for (const video of videos) video.pause();
        button.textContent = "同步播放";
      }
    });
  }

  for (const button of document.querySelectorAll("[data-sync-reset]")) {
    button.addEventListener("click", () => {
      for (const video of videosForPair(button.dataset.syncReset)) {
        video.pause();
        video.currentTime = 0;
      }
      const play = document.querySelector(`[data-sync-play="${button.dataset.syncReset}"]`);
      if (play) play.textContent = "同步播放";
    });
  }
}

wirePairControls();
loadManifest();

const WORKBENCH_ROOT = `${DELIVERY_ROOT}/calibration-workbench`;
const WORKBENCH_INDEX = `${WORKBENCH_ROOT}/take007_alignment.json`;
const WORKBENCH_SCHEMA = "gt_calib.manual_xyz_workbench.v1";
const ADJUSTMENT_SCHEMA = "gt_calib.manual_xyz_profile.v1";
const WORKBENCH_STORAGE_KEY = "gt_calib.take007.manual_xyz.v1";
const HAND_NAMES = ["left", "right"];
const AXES = ["x", "y", "z"];

const workbenchElements = {
  canvas: document.querySelector("#alignment-canvas"),
  loading: document.querySelector("#alignment-loading"),
  play: document.querySelector("#alignment-play"),
  previous: document.querySelector("#alignment-prev"),
  next: document.querySelector("#alignment-next"),
  frame: document.querySelector("#alignment-frame"),
  time: document.querySelector("#alignment-time"),
  linkHands: document.querySelector("#link-hands"),
  enableResidual: document.querySelector("#enable-residual"),
  residualControls: document.querySelector("#residual-controls"),
  showMocap: document.querySelector("#show-mocap"),
  showSolved: document.querySelector("#show-solved"),
  rearOffset: document.querySelector("#rear-offset-value"),
  effectiveLeft: document.querySelector("#effective-left"),
  effectiveRight: document.querySelector("#effective-right"),
  reset: document.querySelector("#alignment-reset"),
  export: document.querySelector("#alignment-export"),
  saveState: document.querySelector("#alignment-save-state"),
};

const sourceVideo = document.createElement("video");
sourceVideo.preload = "auto";
sourceVideo.muted = true;
sourceVideo.playsInline = true;

function zeroVector() {
  return { x: 0, y: 0, z: 0 };
}

function defaultWorkbenchAdjustment() {
  return {
    global: zeroVector(),
    left: zeroVector(),
    right: zeroVector(),
    residualEnabled: false,
    linkHands: true,
    showMocap: true,
    showSolved: true,
  };
}

const workbench = {
  metadata: null,
  camera: null,
  layers: {},
  adjustment: defaultWorkbenchAdjustment(),
  ready: false,
  videoReady: false,
  frameCallbackPending: false,
};

function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function sanitizeVector(value) {
  return Object.fromEntries(AXES.map((axis) => [axis, finiteNumber(value?.[axis])]));
}

function sanitizeAdjustment(value) {
  const defaults = defaultWorkbenchAdjustment();
  return {
    global: sanitizeVector(value?.global),
    left: sanitizeVector(value?.left),
    right: sanitizeVector(value?.right),
    residualEnabled: typeof value?.residualEnabled === "boolean" ? value.residualEnabled : defaults.residualEnabled,
    linkHands: typeof value?.linkHands === "boolean" ? value.linkHands : defaults.linkHands,
    showMocap: typeof value?.showMocap === "boolean" ? value.showMocap : defaults.showMocap,
    showSolved: typeof value?.showSolved === "boolean" ? value.showSolved : defaults.showSolved,
  };
}

function loadSavedAdjustment() {
  try {
    const saved = localStorage.getItem(WORKBENCH_STORAGE_KEY);
    if (!saved) return defaultWorkbenchAdjustment();
    const parsed = JSON.parse(saved);
    return sanitizeAdjustment(parsed?.adjustment ?? parsed);
  } catch (_) {
    return defaultWorkbenchAdjustment();
  }
}

function setSaveState(message, kind = "") {
  const element = workbenchElements.saveState;
  if (!element) return;
  element.textContent = message;
  element.classList.remove("saved", "error");
  if (kind) element.classList.add(kind);
}

function saveAdjustment() {
  try {
    localStorage.setItem(WORKBENCH_STORAGE_KEY, JSON.stringify({
      schema: ADJUSTMENT_SCHEMA,
      adjustment: workbench.adjustment,
      updated_at: new Date().toISOString(),
    }));
    setSaveState("已自动保存到当前浏览器。", "saved");
  } catch (_) {
    setSaveState("浏览器禁止 localStorage；请使用“导出标定 JSON”保存。", "error");
  }
}

function workbenchAssetPath(path) {
  if (typeof path !== "string" || !path || path.startsWith("/") || path.includes("..") || path.includes(":") || path.includes("\\")) {
    throw new Error(`不安全的标定资源路径: ${String(path)}`);
  }
  return `${WORKBENCH_ROOT}/${path.replace(/^\.\//, "")}`;
}

function flattenMatrix(value, rows, columns, label) {
  let flat;
  if (Array.isArray(value) && value.length === rows && value.every((row) => Array.isArray(row) && row.length === columns)) {
    flat = value.flat();
  } else if (Array.isArray(value) && value.length === rows * columns) {
    flat = value.slice();
  } else {
    throw new Error(`${label} 必须是 ${rows}×${columns} 数组`);
  }
  if (flat.some((item) => !Number.isFinite(Number(item)))) throw new Error(`${label} 含非数值项`);
  return flat.map(Number);
}

function positiveInteger(value, label) {
  const number = Number(value);
  if (!Number.isInteger(number) || number <= 0) throw new Error(`${label} 必须是正整数`);
  return number;
}

function validateLayerMetadata(layer, name, frameCount) {
  if (!layer || layer.dtype !== "float32-le") throw new Error(`${name}.dtype 必须是 float32-le`);
  if (!Array.isArray(layer.shape) || layer.shape.length !== 4) throw new Error(`${name}.shape 必须有 4 维`);
  const shape = layer.shape.map((value, index) => positiveInteger(value, `${name}.shape[${index}]`));
  if (shape[0] !== frameCount || shape[1] !== 2 || shape[3] !== 3) {
    throw new Error(`${name}.shape 必须是 [${frameCount}, 2, joints, 3]`);
  }
  if (!Array.isArray(layer.chains) || layer.chains.length === 0) throw new Error(`${name}.chains 缺失`);
  const chains = layer.chains.map((chain, chainIndex) => {
    if (!Array.isArray(chain) || chain.length < 2) throw new Error(`${name}.chains[${chainIndex}] 无效`);
    return chain.map((joint) => {
      const index = Number(joint);
      if (!Number.isInteger(index) || index < 0 || index >= shape[2]) throw new Error(`${name}.chains 含越界 joint`);
      return index;
    });
  });
  return { ...layer, shape, chains };
}

function validateWorkbenchMetadata(metadata) {
  if (metadata?.schema !== WORKBENCH_SCHEMA) throw new Error(`标定 schema 不匹配: ${metadata?.schema ?? "missing"}`);
  const frameCount = positiveInteger(metadata.video?.frame_count, "video.frame_count");
  const video = {
    ...metadata.video,
    width: positiveInteger(metadata.video?.width, "video.width"),
    height: positiveInteger(metadata.video?.height, "video.height"),
    fps: finiteNumber(metadata.video?.fps),
    frame_count: frameCount,
  };
  if (video.fps <= 0) throw new Error("video.fps 必须大于 0");
  workbenchAssetPath(video.path);

  const camera = {
    sourceWidth: positiveInteger(metadata.camera?.source_width ?? metadata.video?.source_width, "camera/video.source_width"),
    sourceHeight: positiveInteger(metadata.camera?.source_height ?? metadata.video?.source_height, "camera/video.source_height"),
    matrix: flattenMatrix(metadata.camera?.matrix, 3, 3, "camera.matrix"),
    worldToColor: flattenMatrix(metadata.camera?.world_to_color, 4, 4, "camera.world_to_color"),
    distortion: Array.isArray(metadata.camera?.distortion) ? metadata.camera.distortion.slice(0, 5).map(Number) : [],
  };
  while (camera.distortion.length < 5) camera.distortion.push(0);
  if (camera.distortion.some((item) => !Number.isFinite(item))) throw new Error("camera.distortion 含非数值项");

  return {
    ...metadata,
    video,
    rear_offset_mm: finiteNumber(metadata.rear_offset_mm, 20),
    layers: {
      mocap: validateLayerMetadata(metadata.layers?.mocap, "layers.mocap", frameCount),
      solved: validateLayerMetadata(metadata.layers?.solved, "layers.solved", frameCount),
    },
    normalizedCamera: camera,
  };
}

function float32LittleEndian(buffer) {
  if (buffer.byteLength % 4 !== 0) throw new Error("Float32 文件大小不是 4 bytes 的整数倍");
  const endianProbe = new Uint16Array([0x00ff]);
  const littleEndianHost = new Uint8Array(endianProbe.buffer)[0] === 0xff;
  if (littleEndianHost) return new Float32Array(buffer);
  const view = new DataView(buffer);
  const values = new Float32Array(buffer.byteLength / 4);
  for (let index = 0; index < values.length; index += 1) values[index] = view.getFloat32(index * 4, true);
  return values;
}

async function loadBinaryLayer(layer, name) {
  const response = await fetch(workbenchAssetPath(layer.path), { cache: "no-store" });
  if (!response.ok) throw new Error(`${name} HTTP ${response.status}`);
  const buffer = await response.arrayBuffer();
  const expectedValues = layer.shape.reduce((product, size) => product * size, 1);
  if (buffer.byteLength !== expectedValues * 4) {
    throw new Error(`${name} 大小错误: ${buffer.byteLength} B，预期 ${expectedValues * 4} B`);
  }
  return { ...layer, values: float32LittleEndian(buffer) };
}

function setWorkbenchLoading(kind, message) {
  const element = workbenchElements.loading;
  if (!element) return;
  element.classList.remove("ready", "failed");
  if (kind) element.classList.add(kind);
  const label = element.querySelector("span:last-child");
  if (label) label.textContent = message;
}

function paintEmptyCanvas(message = "等待 Take_007 数据") {
  const canvas = workbenchElements.canvas;
  const context = canvas?.getContext("2d");
  if (!canvas || !context) return;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#02070d";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = "rgba(66, 217, 238, 0.055)";
  context.lineWidth = 1;
  for (let x = 0; x <= canvas.width; x += 40) {
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, canvas.height);
    context.stroke();
  }
  for (let y = 0; y <= canvas.height; y += 40) {
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(canvas.width, y);
    context.stroke();
  }
  context.fillStyle = "#60758d";
  context.font = "700 13px ui-monospace, monospace";
  context.textAlign = "center";
  context.fillText(message, canvas.width / 2, canvas.height / 2);
}

function currentFrameIndex() {
  const frameCount = workbench.metadata?.video.frame_count ?? 1;
  const fps = workbench.metadata?.video.fps ?? 30;
  return Math.max(0, Math.min(frameCount - 1, Math.round(finiteNumber(sourceVideo.currentTime) * fps)));
}

function effectiveOffset(handName) {
  const global = workbench.adjustment.global;
  const residual = workbench.adjustment.residualEnabled ? workbench.adjustment[handName] : zeroVector();
  return AXES.map((axis) => global[axis] + residual[axis]);
}

function layerPoint(layer, frame, hand, joint) {
  const [, handCount, jointCount, coordinateCount] = layer.shape;
  const start = (((frame * handCount) + hand) * jointCount + joint) * coordinateCount;
  const point = [layer.values[start], layer.values[start + 1], layer.values[start + 2]];
  return point.every(Number.isFinite) ? point : null;
}

function projectWorldPoint(point, offset) {
  const canvas = workbenchElements.canvas;
  const camera = workbench.camera;
  if (!canvas || !camera || !point) return null;
  const xw = point[0] + offset[0];
  const yw = point[1] + offset[1];
  const zw = point[2] + offset[2];
  const transform = camera.worldToColor;
  const xc = transform[0] * xw + transform[1] * yw + transform[2] * zw + transform[3];
  const yc = transform[4] * xw + transform[5] * yw + transform[6] * zw + transform[7];
  const zc = transform[8] * xw + transform[9] * yw + transform[10] * zw + transform[11];
  const wc = transform[12] * xw + transform[13] * yw + transform[14] * zw + transform[15];
  const perspectiveScale = Number.isFinite(wc) && Math.abs(wc) > 1e-12 ? wc : 1;
  const xCamera = xc / perspectiveScale;
  const yCamera = yc / perspectiveScale;
  const zCamera = zc / perspectiveScale;
  if (![xCamera, yCamera, zCamera].every(Number.isFinite) || zCamera <= 1e-6) return null;

  const x = xCamera / zCamera;
  const y = yCamera / zCamera;
  const [k1, k2, p1, p2, k3] = camera.distortion;
  const r2 = x * x + y * y;
  const radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
  const distortedX = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
  const distortedY = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
  const matrix = camera.matrix;
  const sourceU = matrix[0] * distortedX + matrix[1] * distortedY + matrix[2];
  const sourceV = matrix[3] * distortedX + matrix[4] * distortedY + matrix[5];
  const u = sourceU * canvas.width / camera.sourceWidth;
  const v = sourceV * canvas.height / camera.sourceHeight;
  if (![u, v].every(Number.isFinite) || Math.abs(u) > 1e7 || Math.abs(v) > 1e7) return null;
  return [u, v];
}

const LAYER_COLORS = {
  mocap: ["#42d9ee", "#ff50c8"],
  solved: ["#4ce69a", "#ffd45b"],
};

function drawProjectedLayer(context, name, frame) {
  const layer = workbench.layers[name];
  if (!layer) return;
  const isMocap = name === "mocap";
  const jointCount = layer.shape[2];

  for (let hand = 0; hand < 2; hand += 1) {
    const color = LAYER_COLORS[name][hand];
    const offset = effectiveOffset(HAND_NAMES[hand]);
    const projected = Array.from({ length: jointCount }, (_, joint) => projectWorldPoint(layerPoint(layer, frame, hand, joint), offset));

    context.save();
    context.strokeStyle = color;
    context.fillStyle = color;
    context.lineWidth = isMocap ? 2.7 : 2;
    context.lineCap = "round";
    context.lineJoin = "round";
    context.shadowColor = "rgba(0, 0, 0, 0.75)";
    context.shadowBlur = 3;
    for (const chain of layer.chains) {
      let drawing = false;
      context.beginPath();
      for (const joint of chain) {
        const pixel = projected[joint];
        if (!pixel) {
          drawing = false;
          continue;
        }
        if (!drawing) context.moveTo(pixel[0], pixel[1]);
        else context.lineTo(pixel[0], pixel[1]);
        drawing = true;
      }
      context.stroke();
    }

    const usedJoints = new Set(layer.chains.flat());
    for (const joint of usedJoints) {
      const pixel = projected[joint];
      if (!pixel) continue;
      context.beginPath();
      context.arc(pixel[0], pixel[1], isMocap ? 4.2 : 3.1, 0, Math.PI * 2);
      context.fill();
      context.strokeStyle = "rgba(3, 10, 18, 0.9)";
      context.lineWidth = 1.2;
      context.stroke();
      context.strokeStyle = color;
    }

    if (isMocap && projected[0]) {
      const [rootX, rootY] = projected[0];
      context.fillStyle = "#f5fbff";
      context.fillRect(rootX - 2, rootY - 2, 4, 4);
      context.fillStyle = color;
      context.font = "800 10px ui-monospace, monospace";
      context.textAlign = "left";
      context.fillText(`${hand === 0 ? "L" : "R"} ROOT`, rootX + 8, rootY - 7);
    }
    context.restore();
  }
}

function updateTransport(frame) {
  if (workbenchElements.frame) workbenchElements.frame.value = String(frame);
  if (workbenchElements.time && workbench.metadata) {
    const fps = workbench.metadata.video.fps;
    workbenchElements.time.textContent = `${frame} / ${workbench.metadata.video.frame_count - 1} · ${(frame / fps).toFixed(3)} s`;
  }
  if (workbenchElements.play) workbenchElements.play.textContent = sourceVideo.paused ? "播放" : "暂停";
}

function drawAlignmentFrame() {
  const canvas = workbenchElements.canvas;
  const context = canvas?.getContext("2d");
  if (!canvas || !context || !workbench.ready) return;
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (workbench.videoReady && sourceVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    context.drawImage(sourceVideo, 0, 0, canvas.width, canvas.height);
  } else {
    context.fillStyle = "#02070d";
    context.fillRect(0, 0, canvas.width, canvas.height);
  }
  const frame = currentFrameIndex();
  if (workbench.adjustment.showSolved) drawProjectedLayer(context, "solved", frame);
  if (workbench.adjustment.showMocap) drawProjectedLayer(context, "mocap", frame);

  context.save();
  context.fillStyle = "rgba(2, 9, 16, 0.72)";
  context.fillRect(canvas.width - 204, 12, 192, 42);
  context.fillStyle = "#d5dfeb";
  context.font = "700 10px ui-monospace, monospace";
  context.textAlign = "left";
  const global = workbench.adjustment.global;
  context.fillText(`FRAME ${frame} · WORLD Δ`, canvas.width - 195, 28);
  context.fillStyle = "#42d9ee";
  context.fillText(`X ${global.x.toFixed(1)}  Y ${global.y.toFixed(1)}  Z ${global.z.toFixed(1)} mm`, canvas.width - 195, 44);
  context.restore();
  updateTransport(frame);
}

function scheduleAlignmentFrames() {
  if (sourceVideo.paused || workbench.frameCallbackPending) return;
  workbench.frameCallbackPending = true;
  const callback = () => {
    workbench.frameCallbackPending = false;
    drawAlignmentFrame();
    scheduleAlignmentFrames();
  };
  if (typeof sourceVideo.requestVideoFrameCallback === "function") sourceVideo.requestVideoFrameCallback(callback);
  else requestAnimationFrame(callback);
}

function seekAlignmentFrame(frame) {
  if (!workbench.metadata) return;
  const clamped = Math.max(0, Math.min(workbench.metadata.video.frame_count - 1, Math.round(frame)));
  sourceVideo.currentTime = clamped / workbench.metadata.video.fps;
  updateTransport(clamped);
  if (sourceVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) drawAlignmentFrame();
}

function formatOffset(values) {
  return `${values.map((value) => value.toFixed(1)).join(", ")} mm`;
}

function refreshAdjustmentUI() {
  for (const input of document.querySelectorAll("#manual-calibration [data-scope][data-axis]")) {
    const value = workbench.adjustment[input.dataset.scope]?.[input.dataset.axis];
    if (Number.isFinite(value) && document.activeElement !== input) input.value = String(value);
  }
  if (workbenchElements.linkHands) workbenchElements.linkHands.checked = workbench.adjustment.linkHands;
  if (workbenchElements.enableResidual) workbenchElements.enableResidual.checked = workbench.adjustment.residualEnabled;
  if (workbenchElements.showMocap) workbenchElements.showMocap.checked = workbench.adjustment.showMocap;
  if (workbenchElements.showSolved) workbenchElements.showSolved.checked = workbench.adjustment.showSolved;
  if (workbenchElements.residualControls) {
    workbenchElements.residualControls.classList.toggle("enabled", workbench.adjustment.residualEnabled);
    workbenchElements.residualControls.setAttribute("aria-disabled", String(!workbench.adjustment.residualEnabled));
    for (const fieldset of workbenchElements.residualControls.querySelectorAll("fieldset")) fieldset.disabled = !workbench.adjustment.residualEnabled;
  }
  if (workbenchElements.effectiveLeft) workbenchElements.effectiveLeft.textContent = formatOffset(effectiveOffset("left"));
  if (workbenchElements.effectiveRight) workbenchElements.effectiveRight.textContent = formatOffset(effectiveOffset("right"));
}

function applyAdjustmentChange() {
  refreshAdjustmentUI();
  saveAdjustment();
  drawAlignmentFrame();
}

function wireAdjustmentControls() {
  for (const input of document.querySelectorAll("#manual-calibration [data-scope][data-axis]")) {
    input.addEventListener("input", () => {
      const value = Number(input.value);
      if (!Number.isFinite(value)) return;
      const scope = input.dataset.scope;
      const axis = input.dataset.axis;
      workbench.adjustment[scope][axis] = value;
      if (scope !== "global" && workbench.adjustment.linkHands) {
        const otherScope = scope === "left" ? "right" : "left";
        workbench.adjustment[otherScope][axis] = value;
      }
      applyAdjustmentChange();
    });
    input.addEventListener("change", refreshAdjustmentUI);
  }

  workbenchElements.linkHands?.addEventListener("change", () => {
    workbench.adjustment.linkHands = workbenchElements.linkHands.checked;
    if (workbench.adjustment.linkHands) workbench.adjustment.right = { ...workbench.adjustment.left };
    applyAdjustmentChange();
  });
  workbenchElements.enableResidual?.addEventListener("change", () => {
    workbench.adjustment.residualEnabled = workbenchElements.enableResidual.checked;
    applyAdjustmentChange();
  });
  workbenchElements.showMocap?.addEventListener("change", () => {
    workbench.adjustment.showMocap = workbenchElements.showMocap.checked;
    applyAdjustmentChange();
  });
  workbenchElements.showSolved?.addEventListener("change", () => {
    workbench.adjustment.showSolved = workbenchElements.showSolved.checked;
    applyAdjustmentChange();
  });

  workbenchElements.reset?.addEventListener("click", () => {
    workbench.adjustment = defaultWorkbenchAdjustment();
    applyAdjustmentChange();
  });

  workbenchElements.export?.addEventListener("click", () => {
    const left = effectiveOffset("left");
    const right = effectiveOffset("right");
    const payload = {
      schema: ADJUSTMENT_SCHEMA,
      created_at: new Date().toISOString(),
      source_recording: workbench.metadata?.source_recording ?? "camera_glove_recording_20260831_161912",
      source_take: workbench.metadata?.source_take ?? "Take_007",
      coordinate_system: "mocap_world_mm",
      units: "mm",
      global_world_xyz_mm: AXES.map((axis) => workbench.adjustment.global[axis]),
      left_world_xyz_mm: AXES.map((axis) => workbench.adjustment.residualEnabled ? workbench.adjustment.left[axis] : 0),
      right_world_xyz_mm: AXES.map((axis) => workbench.adjustment.residualEnabled ? workbench.adjustment.right[axis] : 0),
      rear_offset_mm: workbench.metadata?.rear_offset_mm ?? 20,
      source: {
        alignment_index: WORKBENCH_INDEX,
        alignment_schema: workbench.metadata?.schema ?? WORKBENCH_SCHEMA,
        video: workbench.metadata?.video.path ?? "take007_clean_rgb.mp4",
        preview_frame_index: currentFrameIndex(),
      },
      coordinate_frame: "world",
      residual_enabled: workbench.adjustment.residualEnabled,
      linked_hands: workbench.adjustment.linkHands,
      side_residual_translation_mm: {
        left: AXES.map((axis) => workbench.adjustment.left[axis]),
        right: AXES.map((axis) => workbench.adjustment.right[axis]),
      },
      effective_translation_mm: { left, right },
      transform_order: "p_world + global_translation + side_residual -> world_to_color -> opencv_distortion -> rgb_pixel",
    };
    const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "take007_manual_xyz_calibration.json";
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 0);
    setSaveState("标定 JSON 已导出。", "saved");
  });
}

function wireAlignmentTransport() {
  workbenchElements.play?.addEventListener("click", async () => {
    if (!workbench.videoReady) return;
    if (sourceVideo.paused) {
      try {
        await sourceVideo.play();
      } catch (error) {
        setSaveState(`视频无法播放：${error.message}`, "error");
      }
    } else {
      sourceVideo.pause();
    }
    updateTransport(currentFrameIndex());
  });
  workbenchElements.previous?.addEventListener("click", () => {
    sourceVideo.pause();
    seekAlignmentFrame(currentFrameIndex() - 1);
  });
  workbenchElements.next?.addEventListener("click", () => {
    sourceVideo.pause();
    seekAlignmentFrame(currentFrameIndex() + 1);
  });
  workbenchElements.frame?.addEventListener("input", () => {
    sourceVideo.pause();
    seekAlignmentFrame(Number(workbenchElements.frame.value));
  });
  sourceVideo.addEventListener("play", () => {
    updateTransport(currentFrameIndex());
    scheduleAlignmentFrames();
  });
  sourceVideo.addEventListener("pause", () => {
    updateTransport(currentFrameIndex());
    drawAlignmentFrame();
  });
  sourceVideo.addEventListener("seeked", drawAlignmentFrame);
  sourceVideo.addEventListener("loadeddata", () => {
    workbench.videoReady = true;
    if (workbenchElements.play) workbenchElements.play.disabled = false;
    setWorkbenchLoading("ready", `已载入 ${workbench.metadata?.video.frame_count ?? 0} 帧`);
    drawAlignmentFrame();
  });
  sourceVideo.addEventListener("error", () => {
    const code = sourceVideo.error?.code ?? "unknown";
    setWorkbenchLoading("failed", `Take_007 clean RGB 无法读取（media error ${code}）`);
  });
}

async function initAlignmentWorkbench() {
  if (!workbenchElements.canvas) return;
  paintEmptyCanvas();
  workbench.adjustment = loadSavedAdjustment();
  refreshAdjustmentUI();
  wireAdjustmentControls();
  wireAlignmentTransport();

  try {
    const response = await fetch(WORKBENCH_INDEX, { cache: "no-store" });
    if (!response.ok) throw new Error(`take007_alignment.json HTTP ${response.status}`);
    const metadata = validateWorkbenchMetadata(await response.json());
    workbench.metadata = metadata;
    workbench.camera = metadata.normalizedCamera;
    if (workbenchElements.canvas) {
      workbenchElements.canvas.width = metadata.video.width;
      workbenchElements.canvas.height = metadata.video.height;
    }
    if (workbenchElements.rearOffset) workbenchElements.rearOffset.textContent = `${metadata.rear_offset_mm.toFixed(0)} mm`;
    if (workbenchElements.frame) {
      workbenchElements.frame.max = String(metadata.video.frame_count - 1);
      workbenchElements.frame.disabled = false;
    }
    if (workbenchElements.previous) workbenchElements.previous.disabled = false;
    if (workbenchElements.next) workbenchElements.next.disabled = false;
    setWorkbenchLoading("", "正在读取 MOCAP / solved Float32…");
    const [mocap, solved] = await Promise.all([
      loadBinaryLayer(metadata.layers.mocap, "take007_mocap.f32"),
      loadBinaryLayer(metadata.layers.solved, "take007_solved.f32"),
    ]);
    workbench.layers = { mocap, solved };
    workbench.ready = true;
    paintEmptyCanvas("视频缓冲中…");
    sourceVideo.src = workbenchAssetPath(metadata.video.path);
    sourceVideo.load();
    updateTransport(0);
  } catch (error) {
    setWorkbenchLoading("failed", `Take_007 标定台未就绪：${error.message}`);
    paintEmptyCanvas("Take_007 标定资源不可用");
  }
}

initAlignmentWorkbench();
