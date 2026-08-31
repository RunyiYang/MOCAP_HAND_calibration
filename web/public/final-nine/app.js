"use strict";

const DELIVERY_ROOT = "/downloads/final-nine";
const EXPECTED_SCHEMA = "gt_calib.final_nine_video_delivery.v1";
const cards = Array.from(document.querySelectorAll("[data-video-id]"));
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
let deliveryManifest = null;

function safeRelativePath(value) {
  if (typeof value !== "string" || !value || value.startsWith("/") || value.includes("..") || value.includes(":")) {
    throw new Error(`不安全的 manifest 路径: ${String(value)}`);
  }
  return `${DELIVERY_ROOT}/${value}`;
}

function cacheBustedPath(path, sha256) {
  if (typeof sha256 !== "string" || !SHA256_PATTERN.test(sha256)) {
    throw new Error(`资源缺少有效 SHA-256: ${String(sha256)}`);
  }
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}sha256=${sha256}`;
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
  const videoPath = cacheBustedPath(safeRelativePath(`videos/${entry.filename}`), entry.sha256);
  const posterPath = cacheBustedPath(safeRelativePath(entry.poster), entry.poster_sha256);

  if (video) video.poster = posterPath;
  if (source && source.getAttribute("src") !== videoPath) {
    source.setAttribute("src", videoPath);
    video?.load();
  }
  if (download) download.href = videoPath;
  if (metrics) metrics.href = cacheBustedPath(safeRelativePath(entry.metrics), entry.metrics_sha256);

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
  const ordered = manifest.videos.slice().sort((left, right) => Number(left.order) - Number(right.order));
  const orders = ordered.map((entry) => entry.order);
  if (orders.some((order, index) => !Number.isInteger(order) || order !== index + 1)) {
    throw new Error("manifest video order 必须严格为 1–9");
  }
  for (const entry of ordered) {
    if (typeof entry.id !== "string" || !entry.id || typeof entry.filename !== "string" || !entry.filename.endsWith(".mp4") || entry.filename.includes("/")) {
      throw new Error("manifest video id/filename 无效");
    }
    if (!SHA256_PATTERN.test(entry.sha256 ?? "")) throw new Error(`manifest video SHA 无效: ${entry.id}`);
    if (!Number.isInteger(entry.frame_count) || entry.frame_count <= 0 || !Number.isFinite(Number(entry.fps)) || Number(entry.fps) <= 0) {
      throw new Error(`manifest video timing 无效: ${entry.id}`);
    }
    if (!Number.isInteger(entry.width) || entry.width <= 0 || !Number.isInteger(entry.height) || entry.height <= 0) {
      throw new Error(`manifest video dimensions 无效: ${entry.id}`);
    }
    if (typeof entry.poster !== "string" || !SHA256_PATTERN.test(entry.poster_sha256 ?? "") || typeof entry.metrics !== "string" || !SHA256_PATTERN.test(entry.metrics_sha256 ?? "")) {
      throw new Error(`manifest poster/metrics provenance 无效: ${entry.id}`);
    }
  }
  const manifestIds = new Set(ordered.map((entry) => entry.id));
  const authoredIds = new Set(cards.map((card) => card.dataset.videoId));
  if (manifestIds.size !== 9 || authoredIds.size !== 9 || [...authoredIds].some((id) => !manifestIds.has(id))) {
    throw new Error("页面卡片与 manifest 视频 ID 不一致");
  }
  manifest.videos = ordered;
  return manifest;
}

async function loadManifest() {
  try {
    const response = await fetch(`${DELIVERY_ROOT}/manifest.json`, { cache: "no-store" });
    if (!response.ok) throw new Error(`manifest HTTP ${response.status}`);
    const manifest = validateManifest(await response.json());
    deliveryManifest = manifest;
    const byId = new Map(manifest.videos.map((entry) => [entry.id, entry]));
    for (const card of cards) hydrateCard(card, byId.get(card.dataset.videoId));

    const totalDuration = manifest.videos.reduce((sum, entry) => sum + Number(entry.duration_s), 0);
    const totalBytes = manifest.videos.reduce((sum, entry) => sum + Number(entry.bytes), 0);
    const videos = document.querySelector("#stat-videos");
    const duration = document.querySelector("#stat-duration");
    if (videos) videos.textContent = String(manifest.video_count);
    if (duration) duration.textContent = formatDuration(totalDuration);
    setManifestState("verified", `manifest 已验证 · 9/9 · ${formatBytes(totalBytes)}`);
    return manifest;
  } catch (error) {
    setManifestState("failed", `manifest 未加载 · ${error.message}`);
    return null;
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
const manifestPromise = loadManifest();

const WORKBENCH_ROOT = `${DELIVERY_ROOT}/calibration-workbench`;
const WORKBENCH_SCHEMA = "gt_calib.manual_xyz_workbench.v1";
const STATE_SCHEMA = "gt_calib.final_nine_manual_xyz_state.v3";
const EXPORT_SCHEMA = "gt_calib.final_nine_manual_xyz.v1";
const WORKBENCH_STORAGE_KEY = "gt_calib.final_nine.manual_xyz.v3";
const HAND_NAMES = ["left", "right"];
const AXES = ["x", "y", "z"];
const LIVE_REPROJECTION_VIDEO_IDS = new Set(["take007-mocap-markers", "take007-solved"]);
const EXCLUDED_VIDEO_IDS = new Set(["no-glove-calibration"]);
const LAYER_VIDEO_IDS = { mocap: "take007-mocap-markers", solved: "take007-solved" };

const workbenchElements = {
  canvas: document.querySelector("#alignment-canvas"),
  loading: document.querySelector("#alignment-loading"),
  play: document.querySelector("#alignment-play"),
  previous: document.querySelector("#alignment-prev"),
  next: document.querySelector("#alignment-next"),
  frame: document.querySelector("#alignment-frame"),
  time: document.querySelector("#alignment-time"),
  videoSelector: document.querySelector("#annotation-video"),
  annotationMode: document.querySelector("#annotation-mode"),
  videoXyzFieldset: document.querySelector("#video-xyz-fieldset"),
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

function defaultGlobalVector() {
  return { x: 0, y: -44, z: 0 };
}

function manifestVideoIds(manifest) {
  return Array.isArray(manifest?.videos) ? manifest.videos.map((entry) => entry.id) : [];
}

function manifestVideoFingerprints(manifest) {
  return Array.isArray(manifest?.videos)
    ? manifest.videos.map((entry) => `${entry.order}:${entry.id}:${entry.filename}:${entry.sha256}`)
    : [];
}

function supportsSideResidual(videoId) {
  return LIVE_REPROJECTION_VIDEO_IDS.has(videoId);
}

function translationApplies(videoId) {
  return !EXCLUDED_VIDEO_IDS.has(videoId);
}

function defaultVideoAdjustment(videoId) {
  return {
    xyz: zeroVector(),
    residualEnabled: false,
    linkHands: true,
    left: zeroVector(),
    right: zeroVector(),
    liveReprojectionAvailable: supportsSideResidual(videoId),
  };
}

function defaultWorkbenchAdjustment(manifest = null) {
  const videoIds = manifestVideoIds(manifest);
  const preferredVideoId = videoIds.includes("take007-mocap-markers") ? "take007-mocap-markers" : (videoIds[0] ?? null);
  return {
    global: defaultGlobalVector(),
    selectedVideoId: preferredVideoId,
    videos: Object.fromEntries(videoIds.map((videoId) => [videoId, defaultVideoAdjustment(videoId)])),
    showMocap: true,
    showSolved: false,
  };
}

const workbench = {
  metadata: null,
  camera: null,
  layers: {},
  adjustment: defaultWorkbenchAdjustment(null),
  baselineAdjustment: defaultWorkbenchAdjustment(null),
  draftBinding: null,
  appliedProfile: null,
  deliveryManifest: null,
  indexPath: null,
  ready: false,
  videoReady: false,
  frameCallbackPending: false,
};

function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function sanitizeVector(value, fallback = zeroVector()) {
  return Object.fromEntries(AXES.map((axis) => [axis, finiteNumber(value?.[axis], fallback[axis])]));
}

function sanitizeVideoAdjustment(value, videoId, defaults = defaultVideoAdjustment(videoId)) {
  if (!supportsSideResidual(videoId)) return defaults;
  return {
    xyz: sanitizeVector(value?.xyz, defaults.xyz),
    residualEnabled: typeof value?.residualEnabled === "boolean" ? value.residualEnabled : defaults.residualEnabled,
    linkHands: typeof value?.linkHands === "boolean" ? value.linkHands : defaults.linkHands,
    left: sanitizeVector(value?.left, defaults.left),
    right: sanitizeVector(value?.right, defaults.right),
    liveReprojectionAvailable: true,
  };
}

function sanitizeManifestVideoAdjustment(value, videoId, defaults = defaultVideoAdjustment(videoId)) {
  if (!translationApplies(videoId)) return defaultVideoAdjustment(videoId);
  if (supportsSideResidual(videoId)) {
    return {
      ...sanitizeVideoAdjustment(value, videoId, defaults),
      xyz: sanitizeVector(value?.xyz, defaults.xyz),
    };
  }
  return {
    ...defaultVideoAdjustment(videoId),
    xyz: sanitizeVector(value?.xyz, defaults.xyz),
  };
}

function sanitizeAdjustment(value, manifest, defaults = defaultWorkbenchAdjustment(manifest)) {
  const videoIds = manifestVideoIds(manifest);
  const selectedVideoId = videoIds.includes(value?.selectedVideoId) ? value.selectedVideoId : defaults.selectedVideoId;
  return {
    global: sanitizeVector(value?.global, defaults.global),
    selectedVideoId,
    videos: Object.fromEntries(videoIds.map((videoId) => [
      videoId,
      sanitizeManifestVideoAdjustment(value?.videos?.[videoId], videoId, defaults.videos?.[videoId] ?? defaultVideoAdjustment(videoId)),
    ])),
    showMocap: typeof value?.showMocap === "boolean" ? value.showMocap : defaults.showMocap,
    showSolved: typeof value?.showSolved === "boolean" ? value.showSolved : defaults.showSolved,
  };
}

function sameOrderedStrings(left, right) {
  return Array.isArray(left) && left.length === right.length && left.every((value, index) => value === right[index]);
}

function buildDraftBinding(manifest, descriptor) {
  const profileDescriptor = descriptor?.applied_manual_profile;
  if (!SHA256_PATTERN.test(descriptor?.sha256 ?? "") || !SHA256_PATTERN.test(profileDescriptor?.sha256 ?? "")) {
    throw new Error("manifest 标定资源缺少有效 SHA-256");
  }
  return {
    manifest_video_fingerprints: manifestVideoFingerprints(manifest),
    workbench_metadata_sha256: descriptor.sha256,
    applied_profile_sha256: profileDescriptor.sha256,
  };
}

function exactDraftBindingMatches(value, expected) {
  return sameOrderedStrings(value?.manifest_video_fingerprints, expected.manifest_video_fingerprints)
    && value?.workbench_metadata_sha256 === expected.workbench_metadata_sha256
    && value?.applied_profile_sha256 === expected.applied_profile_sha256;
}

function loadSavedAdjustment(manifest, baseline, binding) {
  try {
    const saved = localStorage.getItem(WORKBENCH_STORAGE_KEY);
    if (!saved) return sanitizeAdjustment(baseline, manifest, baseline);
    const parsed = JSON.parse(saved);
    if (parsed?.schema !== STATE_SCHEMA || !exactDraftBindingMatches(parsed?.binding, binding)) {
      return sanitizeAdjustment(baseline, manifest, baseline);
    }
    return sanitizeAdjustment(parsed.adjustment, manifest, baseline);
  } catch (_) {
    return sanitizeAdjustment(baseline, manifest, baseline);
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
  if (!workbench.deliveryManifest || !workbench.draftBinding) return;
  try {
    localStorage.setItem(WORKBENCH_STORAGE_KEY, JSON.stringify({
      schema: STATE_SCHEMA,
      binding: workbench.draftBinding,
      adjustment: workbench.adjustment,
      updated_at_utc: new Date().toISOString(),
    }));
    setSaveState("九段 XYZ 已自动保存到当前浏览器。", "saved");
  } catch (_) {
    setSaveState("浏览器禁止 localStorage；请使用“导出九视频 JSON”保存。", "error");
  }
}

function workbenchAssetPath(path, sha256) {
  if (typeof path !== "string" || !path || path.startsWith("/") || path.includes("..") || path.includes(":") || path.includes("\\")) {
    throw new Error(`不安全的标定资源路径: ${String(path)}`);
  }
  return cacheBustedPath(`${WORKBENCH_ROOT}/${path.replace(/^\.\//, "")}`, sha256);
}

function appliedProfileVector(value, label) {
  if (!Array.isArray(value) || value.length !== 3 || value.some((item) => typeof item !== "number" || !Number.isFinite(item))) {
    throw new Error(`${label} 必须是长度为 3 的有限数值数组`);
  }
  return Object.fromEntries(AXES.map((axis, index) => [axis, Object.is(value[index], -0) ? 0 : value[index]]));
}

function objectVectorIsZero(vector) {
  return AXES.every((axis) => vector[axis] === 0);
}

function objectVectorsMatch(left, right) {
  return AXES.every((axis) => left[axis] === right[axis]);
}

function validateAppliedManualProfile(profile, manifest) {
  if (profile?.schema !== EXPORT_SCHEMA) throw new Error(`applied profile schema 不匹配: ${profile?.schema ?? "missing"}`);
  if (!sameOrderedStrings(profile.axis_order, AXES) || profile.units !== "mm" || profile.coordinate_frame !== "per_video_mocap_world") {
    throw new Error("applied profile 坐标系、单位或轴序不匹配");
  }
  if (!Array.isArray(profile.video_annotations) || profile.video_annotations.length !== 9 || manifest.videos.length !== 9) {
    throw new Error("applied profile 必须恰好包含 9 条视频标注");
  }

  const adjustment = defaultWorkbenchAdjustment(manifest);
  adjustment.global = appliedProfileVector(profile.global_world_xyz_mm, "applied profile global_world_xyz_mm");
  for (let index = 0; index < manifest.videos.length; index += 1) {
    const source = manifest.videos[index];
    const row = profile.video_annotations[index];
    if (!row || row.order !== source.order || row.video_id !== source.id || row.filename !== source.filename) {
      throw new Error(`applied profile 第 ${index + 1} 条未绑定当前 manifest 的 order/id/filename`);
    }
    if (!SHA256_PATTERN.test(row.video_sha256 ?? "")) throw new Error(`applied profile video SHA 无效: ${source.id}`);
    const shouldApply = translationApplies(source.id);
    if (row.apply_translation !== shouldApply) throw new Error(`applied profile apply_translation 错误: ${source.id}`);

    const perVideo = appliedProfileVector(row.per_video_world_xyz_mm, `${source.id}.per_video_world_xyz_mm`);
    const left = appliedProfileVector(row.left_residual_world_xyz_mm, `${source.id}.left_residual_world_xyz_mm`);
    const right = appliedProfileVector(row.right_residual_world_xyz_mm, `${source.id}.right_residual_world_xyz_mm`);
    const sideAllowed = supportsSideResidual(source.id);
    if (!sideAllowed && (!objectVectorIsZero(left) || !objectVectorIsZero(right))) {
      throw new Error(`applied profile 仅允许 Take_007 使用左右 residual: ${source.id}`);
    }
    if (!shouldApply && !objectVectorIsZero(perVideo)) throw new Error("第 09 段无 MOCAP，applied profile 必须保持零 translation");

    adjustment.videos[source.id] = {
      xyz: shouldApply ? perVideo : zeroVector(),
      residualEnabled: sideAllowed && (!objectVectorIsZero(left) || !objectVectorIsZero(right)),
      linkHands: sideAllowed && objectVectorsMatch(left, right),
      left: sideAllowed ? left : zeroVector(),
      right: sideAllowed ? right : zeroVector(),
      liveReprojectionAvailable: sideAllowed,
    };
  }
  return { profile, adjustment };
}

async function loadAppliedManualProfile(manifest, descriptor) {
  const profileDescriptor = descriptor?.applied_manual_profile;
  if (profileDescriptor?.schema !== EXPORT_SCHEMA || typeof profileDescriptor.path !== "string" || !SHA256_PATTERN.test(profileDescriptor.sha256 ?? "")) {
    throw new Error("manifest 缺少有效 calibration_workbench.applied_manual_profile");
  }
  const profilePath = safeRelativePath(profileDescriptor.path);
  const response = await fetch(cacheBustedPath(profilePath, profileDescriptor.sha256), { cache: "no-store" });
  if (!response.ok) throw new Error(`applied_manual_profile.json HTTP ${response.status}`);
  return validateAppliedManualProfile(await response.json(), manifest);
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
  workbenchAssetPath(video.path, video.sha256);

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
  const response = await fetch(workbenchAssetPath(layer.path, layer.sha256), { cache: "no-store" });
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

function selectedVideoEntry() {
  return workbench.deliveryManifest?.videos.find((entry) => entry.id === workbench.adjustment.selectedVideoId) ?? null;
}

function selectedVideoAdjustment() {
  return workbench.adjustment.videos?.[workbench.adjustment.selectedVideoId] ?? null;
}

function activePreviewTiming() {
  const entry = selectedVideoEntry();
  if (!entry) return { frame_count: 1, fps: 30, width: 960, height: 540 };
  if (supportsSideResidual(entry.id) && workbench.metadata) return workbench.metadata.video;
  return {
    frame_count: positiveInteger(entry.frame_count, `${entry.id}.frame_count`),
    fps: finiteNumber(entry.fps, 30),
    width: positiveInteger(entry.width ?? 960, `${entry.id}.width`),
    height: positiveInteger(entry.height ?? 540, `${entry.id}.height`),
  };
}

function currentFrameIndex() {
  const timing = activePreviewTiming();
  const frameCount = timing.frame_count;
  const fps = timing.fps;
  return Math.max(0, Math.min(frameCount - 1, Math.round(finiteNumber(sourceVideo.currentTime) * fps)));
}

function effectiveOffset(videoId, handName) {
  if (!translationApplies(videoId)) return null;
  const global = workbench.adjustment.global;
  const video = workbench.adjustment.videos?.[videoId] ?? defaultVideoAdjustment(videoId);
  const sideResidual = supportsSideResidual(videoId) && video.residualEnabled ? video[handName] : zeroVector();
  return AXES.map((axis) => global[axis] + video.xyz[axis] + sideResidual[axis]);
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
  const videoId = LAYER_VIDEO_IDS[name];

  for (let hand = 0; hand < 2; hand += 1) {
    const color = LAYER_COLORS[name][hand];
    const offset = effectiveOffset(videoId, HAND_NAMES[hand]);
    if (!offset) continue;
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
  const timing = activePreviewTiming();
  if (workbenchElements.frame) workbenchElements.frame.value = String(frame);
  if (workbenchElements.time) {
    workbenchElements.time.textContent = `${frame} / ${timing.frame_count - 1} · ${(frame / timing.fps).toFixed(3)} s`;
  }
  if (workbenchElements.play) workbenchElements.play.textContent = sourceVideo.paused ? "播放" : "暂停";
}

function switchSelectedPreview() {
  const entry = selectedVideoEntry();
  if (!entry) return;
  if (supportsSideResidual(entry.id) && !workbench.metadata) {
    setWorkbenchLoading("", "正在读取 Take_007 实时重投影资源…");
    return;
  }

  sourceVideo.pause();
  workbench.videoReady = false;
  if (workbenchElements.play) workbenchElements.play.disabled = true;
  const timing = activePreviewTiming();
  if (workbenchElements.canvas) {
    workbenchElements.canvas.width = timing.width;
    workbenchElements.canvas.height = timing.height;
  }
  if (workbenchElements.frame) {
    workbenchElements.frame.max = String(timing.frame_count - 1);
    workbenchElements.frame.value = "0";
    workbenchElements.frame.disabled = false;
  }
  if (workbenchElements.previous) workbenchElements.previous.disabled = false;
  if (workbenchElements.next) workbenchElements.next.disabled = false;

  const source = supportsSideResidual(entry.id)
    ? workbenchAssetPath(workbench.metadata.video.path, workbench.metadata.video.sha256)
    : cacheBustedPath(safeRelativePath(`videos/${entry.filename}`), entry.sha256);
  const previewLabel = supportsSideResidual(entry.id) ? "clean RGB + 实时 3D 重投影" : "正式已烘焙 MP4";
  setWorkbenchLoading("", `正在载入 ${String(entry.order).padStart(2, "0")} · ${entry.id} · ${previewLabel}…`);
  sourceVideo.src = source;
  sourceVideo.load();
  updateTransport(0);
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
  const selectedId = workbench.adjustment.selectedVideoId;
  if (supportsSideResidual(selectedId)) {
    if (workbench.adjustment.showSolved) drawProjectedLayer(context, "solved", frame);
    if (workbench.adjustment.showMocap) drawProjectedLayer(context, "mocap", frame);
  }

  context.save();
  context.fillStyle = "rgba(2, 9, 16, 0.72)";
  context.fillRect(canvas.width - 304, 12, 292, 42);
  context.fillStyle = "#d5dfeb";
  context.font = "700 10px ui-monospace, monospace";
  context.textAlign = "left";
  const global = workbench.adjustment.global;
  const previewMode = supportsSideResidual(selectedId) ? "LIVE REPROJECT" : (translationApplies(selectedId) ? "BAKED · RECORD ONLY" : "NO MOCAP · EXCLUDED");
  context.fillText(`FRAME ${frame} · ${previewMode}`, canvas.width - 295, 28);
  context.fillStyle = translationApplies(selectedId) ? "#42d9ee" : "#ff8ea5";
  context.fillText(`GLOBAL X ${global.x.toFixed(1)}  Y ${global.y.toFixed(1)}  Z ${global.z.toFixed(1)} mm`, canvas.width - 295, 44);
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
  const timing = activePreviewTiming();
  const clamped = Math.max(0, Math.min(timing.frame_count - 1, Math.round(frame)));
  sourceVideo.currentTime = clamped / timing.fps;
  updateTransport(clamped);
  if (sourceVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) drawAlignmentFrame();
}

function formatOffset(values) {
  if (values === null) return "N/A · no MOCAP";
  return `${values.map((value) => value.toFixed(1)).join(", ")} mm`;
}

function vectorForScope(scope) {
  if (scope === "global") return workbench.adjustment.global;
  const selected = selectedVideoAdjustment();
  if (!selected) return null;
  if (scope === "video") return selected.xyz;
  if (scope === "left" || scope === "right") return selected[scope];
  return null;
}

function populateVideoSelector(manifest) {
  const selector = workbenchElements.videoSelector;
  if (!selector) return;
  selector.replaceChildren(...manifest.videos.map((entry) => {
    const option = document.createElement("option");
    option.value = entry.id;
    const suffix = EXCLUDED_VIDEO_IDS.has(entry.id) ? " · 无 MOCAP / excluded" : (supportsSideResidual(entry.id) ? " · live reprojection" : " · offline rerender");
    option.textContent = `${String(entry.order).padStart(2, "0")} · ${entry.id}${suffix}`;
    return option;
  }));
  selector.disabled = false;
}

function updateAnnotationMode(videoId) {
  const element = workbenchElements.annotationMode;
  if (!element) return;
  if (EXCLUDED_VIDEO_IDS.has(videoId)) {
    element.dataset.mode = "excluded";
    element.textContent = "第 09 段没有 MOCAP 手部数据：保留在九条导出中，但 apply_translation=false，effective=null。";
  } else if (LIVE_REPROJECTION_VIDEO_IDS.has(videoId)) {
    element.dataset.mode = "live";
    element.textContent = "第 07 / 08 段使用 clean RGB 与对应 3D layer 实时重投影；可启用左右手 residual。";
  } else {
    element.dataset.mode = "record";
    element.textContent = "参数记录模式：下方预览是正式已烘焙 MP4，XYZ 属于本视频自己的 mocap-world，需离线重渲染后才能看到变化。";
  }
}

function refreshAdjustmentUI() {
  const selectedId = workbench.adjustment.selectedVideoId;
  const selected = selectedVideoAdjustment();
  const applicable = translationApplies(selectedId);
  const sideAllowed = supportsSideResidual(selectedId);
  for (const input of document.querySelectorAll("#manual-calibration [data-scope][data-axis]")) {
    const value = vectorForScope(input.dataset.scope)?.[input.dataset.axis];
    if (Number.isFinite(value) && document.activeElement !== input) input.value = String(value);
  }
  if (workbenchElements.videoSelector && selectedId) workbenchElements.videoSelector.value = selectedId;
  if (workbenchElements.videoXyzFieldset) workbenchElements.videoXyzFieldset.disabled = !applicable;
  if (workbenchElements.linkHands) {
    workbenchElements.linkHands.checked = sideAllowed && Boolean(selected?.linkHands);
    workbenchElements.linkHands.disabled = !sideAllowed;
  }
  if (workbenchElements.enableResidual) {
    workbenchElements.enableResidual.checked = sideAllowed && Boolean(selected?.residualEnabled);
    workbenchElements.enableResidual.disabled = !sideAllowed;
  }
  if (workbenchElements.showMocap) workbenchElements.showMocap.checked = workbench.adjustment.showMocap;
  if (workbenchElements.showSolved) workbenchElements.showSolved.checked = workbench.adjustment.showSolved;
  if (workbenchElements.showMocap) workbenchElements.showMocap.disabled = !sideAllowed;
  if (workbenchElements.showSolved) workbenchElements.showSolved.disabled = !sideAllowed;
  if (workbenchElements.residualControls) {
    const residualEnabled = sideAllowed && Boolean(selected?.residualEnabled);
    workbenchElements.residualControls.classList.toggle("enabled", residualEnabled);
    workbenchElements.residualControls.setAttribute("aria-disabled", String(!residualEnabled));
    for (const fieldset of workbenchElements.residualControls.querySelectorAll("fieldset")) fieldset.disabled = !residualEnabled;
  }
  if (workbenchElements.effectiveLeft) workbenchElements.effectiveLeft.textContent = formatOffset(effectiveOffset(selectedId, "left"));
  if (workbenchElements.effectiveRight) workbenchElements.effectiveRight.textContent = formatOffset(effectiveOffset(selectedId, "right"));
  if (workbenchElements.export) workbenchElements.export.disabled = !workbench.deliveryManifest;
  updateAnnotationMode(selectedId);
  for (const card of cards) {
    const selectedCard = card.dataset.videoId === selectedId;
    card.classList.toggle("annotation-selected", selectedCard);
    if (selectedCard) card.setAttribute("aria-current", "true");
    else card.removeAttribute("aria-current");
  }
}

function applyAdjustmentChange() {
  refreshAdjustmentUI();
  saveAdjustment();
  drawAlignmentFrame();
}

function strictVectorArray(vector, label) {
  const values = AXES.map((axis) => Number(vector?.[axis]));
  if (values.length !== 3 || values.some((value) => !Number.isFinite(value))) throw new Error(`${label} 必须是有限 XYZ`);
  return values.map((value) => Object.is(value, -0) ? 0 : value);
}

function assertArrayVector(value, label) {
  if (!Array.isArray(value) || value.length !== 3 || value.some((item) => typeof item !== "number" || !Number.isFinite(item))) {
    throw new Error(`${label} 必须是长度为 3 的有限数值数组`);
  }
}

function addVectors(...vectors) {
  return AXES.map((_, index) => vectors.reduce((sum, vector) => sum + vector[index], 0));
}

function zeroArray() {
  return [0, 0, 0];
}

function sameVector(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === 3 && right.length === 3 && left.every((value, index) => value === right[index]);
}

function buildExportPayload() {
  const manifest = workbench.deliveryManifest;
  if (!manifest) throw new Error("manifest 尚未验证");
  const global = strictVectorArray(workbench.adjustment.global, "global_world_xyz_mm");
  const videoAnnotations = manifest.videos.map((entry) => {
    const adjustment = workbench.adjustment.videos?.[entry.id];
    if (!adjustment) throw new Error(`缺少视频标注状态: ${entry.id}`);
    const applyTranslation = translationApplies(entry.id);
    const liveReprojection = supportsSideResidual(entry.id);
    const perVideo = applyTranslation ? strictVectorArray(adjustment.xyz, `${entry.id}.per_video_world_xyz_mm`) : zeroArray();
    const residualEnabled = liveReprojection && Boolean(adjustment.residualEnabled);
    const leftResidual = residualEnabled ? strictVectorArray(adjustment.left, `${entry.id}.left_residual_world_xyz_mm`) : zeroArray();
    const rightResidual = residualEnabled ? strictVectorArray(adjustment.right, `${entry.id}.right_residual_world_xyz_mm`) : zeroArray();
    const base = applyTranslation ? addVectors(global, perVideo) : null;
    return {
      order: entry.order,
      video_id: entry.id,
      filename: entry.filename,
      video_sha256: entry.sha256,
      apply_translation: applyTranslation,
      exclusion_reason: applyTranslation ? null : "no_mocap_hand_pose",
      live_reprojection_available: liveReprojection,
      per_video_world_xyz_mm: perVideo,
      side_residual_enabled: residualEnabled,
      linked_hands: liveReprojection ? Boolean(adjustment.linkHands) : false,
      left_residual_world_xyz_mm: leftResidual,
      right_residual_world_xyz_mm: rightResidual,
      effective_left_world_xyz_mm: applyTranslation ? addVectors(base, residualEnabled ? leftResidual : zeroArray()) : null,
      effective_right_world_xyz_mm: applyTranslation ? addVectors(base, residualEnabled ? rightResidual : zeroArray()) : null,
    };
  });
  return {
    schema: EXPORT_SCHEMA,
    created_at_utc: new Date().toISOString(),
    coordinate_frame: "per_video_mocap_world",
    coordinate_frame_semantics: "The same global operator override is copied independently into each applicable video's own mocap-world; it is not a shared physical extrinsic.",
    axis_order: AXES.slice(),
    units: "mm",
    global_world_xyz_mm: global,
    default_global_world_xyz_mm: [0, -44, 0],
    source_manifest: {
      path: `${DELIVERY_ROOT}/manifest.json`,
      schema: manifest.schema,
      generated_at_utc: manifest.generated_at_utc ?? null,
    },
    video_annotations: videoAnnotations,
    transform_order: "p_video_mocap_world + global_translation + per_video_translation + optional_side_residual",
  };
}

function validateExportPayload(payload, manifest) {
  if (payload?.schema !== EXPORT_SCHEMA) throw new Error("导出 schema 不匹配");
  if (payload.coordinate_frame !== "per_video_mocap_world" || payload.units !== "mm" || !sameOrderedStrings(payload.axis_order, AXES)) {
    throw new Error("导出坐标系/单位/轴序不匹配");
  }
  if (payload.source_manifest?.schema !== manifest?.schema || payload.source_manifest?.generated_at_utc !== (manifest?.generated_at_utc ?? null)) {
    throw new Error("导出未绑定当前 manifest");
  }
  assertArrayVector(payload.global_world_xyz_mm, "global_world_xyz_mm");
  if (!Array.isArray(payload.video_annotations) || payload.video_annotations.length !== 9 || manifest?.videos?.length !== 9) {
    throw new Error("导出必须恰好包含 9 条视频标注");
  }
  const exportedIds = new Set();
  for (let index = 0; index < manifest.videos.length; index += 1) {
    const source = manifest.videos[index];
    const item = payload.video_annotations[index];
    if (!item || item.order !== index + 1 || item.order !== source.order || item.video_id !== source.id || item.filename !== source.filename || item.video_sha256 !== source.sha256) {
      throw new Error(`第 ${index + 1} 条标注未严格绑定 manifest`);
    }
    if (exportedIds.has(item.video_id)) throw new Error(`导出含重复视频 ID: ${item.video_id}`);
    exportedIds.add(item.video_id);
    assertArrayVector(item.per_video_world_xyz_mm, `${item.video_id}.per_video_world_xyz_mm`);
    assertArrayVector(item.left_residual_world_xyz_mm, `${item.video_id}.left_residual_world_xyz_mm`);
    assertArrayVector(item.right_residual_world_xyz_mm, `${item.video_id}.right_residual_world_xyz_mm`);
    const shouldApply = !EXCLUDED_VIDEO_IDS.has(item.video_id);
    const shouldReproject = LIVE_REPROJECTION_VIDEO_IDS.has(item.video_id);
    if (item.apply_translation !== shouldApply || item.live_reprojection_available !== shouldReproject) throw new Error(`${item.video_id} apply/live 标志错误`);
    if (!shouldReproject && (item.side_residual_enabled || item.linked_hands || !sameVector(item.left_residual_world_xyz_mm, zeroArray()) || !sameVector(item.right_residual_world_xyz_mm, zeroArray()))) {
      throw new Error(`${item.video_id} 不允许左右手 residual`);
    }
    if (!shouldApply) {
      if (item.exclusion_reason !== "no_mocap_hand_pose" || item.effective_left_world_xyz_mm !== null || item.effective_right_world_xyz_mm !== null || !sameVector(item.per_video_world_xyz_mm, zeroArray())) {
        throw new Error(`${item.video_id} 必须 excluded 且 effective=null`);
      }
      continue;
    }
    assertArrayVector(item.effective_left_world_xyz_mm, `${item.video_id}.effective_left_world_xyz_mm`);
    assertArrayVector(item.effective_right_world_xyz_mm, `${item.video_id}.effective_right_world_xyz_mm`);
    const base = addVectors(payload.global_world_xyz_mm, item.per_video_world_xyz_mm);
    const expectedLeft = addVectors(base, item.side_residual_enabled ? item.left_residual_world_xyz_mm : zeroArray());
    const expectedRight = addVectors(base, item.side_residual_enabled ? item.right_residual_world_xyz_mm : zeroArray());
    if (!sameVector(item.effective_left_world_xyz_mm, expectedLeft) || !sameVector(item.effective_right_world_xyz_mm, expectedRight)) {
      throw new Error(`${item.video_id} effective XYZ 与变换顺序不一致`);
    }
  }
  if (exportedIds.size !== 9) throw new Error("导出视频 ID 必须唯一且恰好为 9 个");
}

function wireAdjustmentControls() {
  for (const input of document.querySelectorAll("#manual-calibration [data-scope][data-axis]")) {
    input.addEventListener("input", () => {
      const value = Number(input.value);
      if (!Number.isFinite(value)) return;
      const scope = input.dataset.scope;
      const axis = input.dataset.axis;
      const target = vectorForScope(scope);
      const selected = selectedVideoAdjustment();
      if (!target || (scope === "video" && !translationApplies(workbench.adjustment.selectedVideoId)) || ((scope === "left" || scope === "right") && !supportsSideResidual(workbench.adjustment.selectedVideoId))) return;
      target[axis] = value;
      if ((scope === "left" || scope === "right") && selected?.linkHands) {
        const otherScope = scope === "left" ? "right" : "left";
        selected[otherScope][axis] = value;
      }
      applyAdjustmentChange();
    });
    input.addEventListener("change", refreshAdjustmentUI);
  }

  workbenchElements.linkHands?.addEventListener("change", () => {
    const selected = selectedVideoAdjustment();
    if (!selected || !supportsSideResidual(workbench.adjustment.selectedVideoId)) return;
    selected.linkHands = workbenchElements.linkHands.checked;
    if (selected.linkHands) selected.right = { ...selected.left };
    applyAdjustmentChange();
  });
  workbenchElements.enableResidual?.addEventListener("change", () => {
    const selected = selectedVideoAdjustment();
    if (!selected || !supportsSideResidual(workbench.adjustment.selectedVideoId)) return;
    selected.residualEnabled = workbenchElements.enableResidual.checked;
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

  workbenchElements.videoSelector?.addEventListener("change", () => {
    const videoId = workbenchElements.videoSelector.value;
    if (!manifestVideoIds(workbench.deliveryManifest).includes(videoId)) return;
    workbench.adjustment.selectedVideoId = videoId;
    if (videoId === "take007-mocap-markers") {
      workbench.adjustment.showMocap = true;
      workbench.adjustment.showSolved = false;
    } else if (videoId === "take007-solved") {
      workbench.adjustment.showMocap = false;
      workbench.adjustment.showSolved = true;
    }
    applyAdjustmentChange();
    switchSelectedPreview();
  });

  workbenchElements.reset?.addEventListener("click", () => {
    workbench.adjustment = sanitizeAdjustment(
      workbench.baselineAdjustment,
      workbench.deliveryManifest,
      workbench.baselineAdjustment,
    );
    applyAdjustmentChange();
    setSaveState("已恢复 manifest 指向的 applied profile 基线。", "saved");
    switchSelectedPreview();
  });

  workbenchElements.export?.addEventListener("click", () => {
    try {
      const payload = buildExportPayload();
      validateExportPayload(payload, workbench.deliveryManifest);
      const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "final_nine_manual_xyz_annotations.json";
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 0);
      setSaveState("严格九视频 XYZ JSON 已导出。", "saved");
    } catch (error) {
      setSaveState(`导出失败：${error.message}`, "error");
    }
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
    const entry = selectedVideoEntry();
    const timing = activePreviewTiming();
    const mode = supportsSideResidual(entry?.id) ? "clean RGB + live layer" : "正式已烘焙 MP4";
    setWorkbenchLoading("ready", `已载入 ${String(entry?.order ?? "?").padStart(2, "0")} · ${timing.frame_count} 帧 · ${mode}`);
    drawAlignmentFrame();
  });
  sourceVideo.addEventListener("error", () => {
    const code = sourceVideo.error?.code ?? "unknown";
    setWorkbenchLoading("failed", `${selectedVideoEntry()?.id ?? "当前视频"} 无法读取（media error ${code}）`);
  });
}

async function initAlignmentWorkbench() {
  if (!workbenchElements.canvas) return;
  paintEmptyCanvas("等待九视频 manifest");
  wireAdjustmentControls();
  wireAlignmentTransport();

  try {
    const manifest = await manifestPromise;
    if (!manifest) throw new Error("九视频 manifest 未通过验证");
    workbench.deliveryManifest = manifest;

    const descriptor = manifest.calibration_workbench;
    if (!descriptor || typeof descriptor.path !== "string" || !SHA256_PATTERN.test(descriptor.sha256 ?? "")) throw new Error("manifest 缺少有效 calibration_workbench");
    workbench.draftBinding = buildDraftBinding(manifest, descriptor);
    workbench.indexPath = safeRelativePath(descriptor.path);
    const [response, appliedProfile] = await Promise.all([
      fetch(cacheBustedPath(workbench.indexPath, descriptor.sha256), { cache: "no-store" }),
      loadAppliedManualProfile(manifest, descriptor),
    ]);
    if (!response.ok) throw new Error(`take007_alignment.json HTTP ${response.status}`);
    workbench.appliedProfile = appliedProfile.profile;
    workbench.baselineAdjustment = appliedProfile.adjustment;
    workbench.adjustment = loadSavedAdjustment(manifest, workbench.baselineAdjustment, workbench.draftBinding);
    populateVideoSelector(manifest);
    refreshAdjustmentUI();

    const metadata = validateWorkbenchMetadata(await response.json());
    workbench.metadata = metadata;
    workbench.camera = metadata.normalizedCamera;
    if (workbenchElements.rearOffset) workbenchElements.rearOffset.textContent = `${metadata.rear_offset_mm.toFixed(0)} mm`;
    setWorkbenchLoading("", "正在读取 MOCAP / solved Float32…");
    const [mocap, solved] = await Promise.all([
      loadBinaryLayer(metadata.layers.mocap, "take007_mocap.f32"),
      loadBinaryLayer(metadata.layers.solved, "take007_solved.f32"),
    ]);
    workbench.layers = { mocap, solved };
    workbench.ready = true;
    paintEmptyCanvas("视频缓冲中…");
    switchSelectedPreview();
  } catch (error) {
    workbench.ready = Boolean(workbench.deliveryManifest);
    setWorkbenchLoading("failed", `标定台未完整就绪：${error.message}`);
    paintEmptyCanvas("标定资源不可用；可切换到已烘焙视频检查");
  }
}

initAlignmentWorkbench();
