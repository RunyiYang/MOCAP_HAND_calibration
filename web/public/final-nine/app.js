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
