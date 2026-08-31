"use strict";

const state = {
  data: null,
  selectedTake: "take-001",
  viewMode: "mocap",
};

const takeButtons = Array.from(document.querySelectorAll('[role="tab"][data-take]'));
const takePanels = Array.from(document.querySelectorAll('[role="tabpanel"][data-take]'));
const viewModeButtons = Array.from(document.querySelectorAll("[data-view-mode]"));

function currentTakes() {
  if (!state.data) return [];
  if (state.viewMode === "mocap") return state.data.mocapTakes || [];
  if (state.viewMode === "depth") return state.data.depthTakes || [];
  if (state.viewMode === "diagnostic") return state.data.diagnosticTakes || [];
  return state.data.takes || [];
}

function selectedMediaSource(source) {
  if (state.viewMode === "mocap") return source?.dataset.mocapSrc;
  if (state.viewMode === "depth") return source?.dataset.depthSrc;
  if (state.viewMode === "diagnostic") return source?.dataset.diagnosticSrc;
  return source?.dataset.fusionSrc;
}

function selectedPoster(video) {
  if (state.viewMode === "mocap") return video?.dataset.mocapPoster;
  if (state.viewMode === "depth") return video?.dataset.depthPoster;
  if (state.viewMode === "diagnostic") return video?.dataset.diagnosticPoster;
  return video?.dataset.fusionPoster;
}

function nestedValue(object, path) {
  return path.split(".").reduce((value, key) => value?.[key], object);
}

function formatNumber(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function showToast(message) {
  const toast = document.querySelector("#toast");
  if (!toast) return;
  toast.textContent = message;
  toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    toast.hidden = true;
  }, 3200);
}

function activateTake(takeId, focusButton = false) {
  const valid = takeButtons.some((button) => button.dataset.take === takeId);
  if (!valid) return;

  state.selectedTake = takeId;
  for (const button of takeButtons) {
    const selected = button.dataset.take === takeId;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    if (selected && focusButton) button.focus();
  }

  for (const panel of takePanels) {
    const selected = panel.dataset.take === takeId;
    panel.hidden = !selected;
    const video = panel.querySelector("video");
    const source = video?.querySelector("source[data-fusion-src]");
    const desiredSource = selectedMediaSource(source);
    const desiredPoster = selectedPoster(video);
    const download = video?.querySelector("[data-video-download]");
    if (selected && download && desiredSource) download.href = desiredSource;
    if (selected && video && desiredPoster && video.getAttribute("poster") !== desiredPoster) {
      video.poster = desiredPoster;
    } else if (!selected && video?.hasAttribute("poster")) {
      video.removeAttribute("poster");
    }
    if (selected && video && source && desiredSource && source.getAttribute("src") !== desiredSource) {
      source.setAttribute("src", desiredSource);
      video.load();
    } else if (!selected && video && source && source.hasAttribute("src")) {
      video.pause();
      source.removeAttribute("src");
      video.load();
    }
  }

  const url = new URL(window.location.href);
  url.searchParams.set("take", String(Number(takeId.slice(-3)) + 1).padStart(2, "0"));
  url.searchParams.set("mode", state.viewMode);
  window.history.replaceState(null, "", url);

  const selectedButton = takeButtons.find((button) => button.dataset.take === takeId);
  const tabList = selectedButton?.parentElement;
  if (selectedButton && tabList) {
    const left = selectedButton.offsetLeft - (tabList.clientWidth - selectedButton.clientWidth) / 2;
    tabList.scrollTo({ left: Math.max(0, left), behavior: focusButton ? "smooth" : "auto" });
  }
}

function activateViewMode(mode, updateUrl = true) {
  if (!viewModeButtons.some((button) => button.dataset.viewMode === mode)) return;
  state.viewMode = mode;
  document.documentElement.dataset.viewMode = mode;

  for (const button of viewModeButtons) {
    button.setAttribute("aria-pressed", String(button.dataset.viewMode === mode));
  }

  const explanation = document.querySelector("#mode-explanation");
  if (explanation) {
    explanation.textContent = mode === "mocap"
      ? "动作审阅：原始 Skeleton_0/1 每手 21 joints 保持不变；正式视频里带白色中心的同侧彩色圆点仅是朝后腕黑色模块外推的 VIZ-only proxy，不是动捕 GT。"
      : mode === "depth"
        ? "深度审阅：直接解码 raw mono16 毫米值，使用每个 depth frame 自身时间戳插值 MOCAP，并冻结 delivered world→depth SE(3)。"
      : mode === "fusion"
        ? "融合视图：MOCAP 提供 wrist position + orientation，glove 提供 root-local 手指姿态。"
        : "诊断视图：只共享 MOCAP wrist translation，保留 glove wrist orientation，用来暴露跨 session 漂移。";
  }

  const microcopy = document.querySelector("#mode-microcopy");
  if (microcopy) {
    microcopy.textContent = mode === "mocap"
      ? "洋红/青色分别是原始左右手 21 joints；带白色中心的同侧彩色圆点是 rear wrist module proxy = wrist − 0.8 × (middle MCP − wrist)。它不改任何原 joint，也不是新增 GT joint；可下载 raw 21-joint 视频逐帧对照。"
      : mode === "depth"
        ? "固定色标显示 350–1800 mm raw depth，黑色为无效深度；骨架是冻结外参投影。joint pixel 的 nonzero depth coverage 只表示传感器有返回值，不是 pose accuracy，也不测 skeleton-to-surface error。"
      : mode === "fusion"
        ? "绿色融合骨架的 wrist SE(3) 来自曝光时刻 MOCAP；视频可验证 composition 与手指形态，但可视贴合不等于独立逐关节像素 GT。拇指会绘制，但不进入当前指标。"
        : "旧诊断骨架只共享曝光时刻 MOCAP wrist translation，orientation 保留 glove solver 输出；它用于暴露跨 session 漂移，不是推荐融合结果。拇指会绘制，但不进入当前指标。";
  }

  const metricsLabel = document.querySelector("#metrics-mode-label");
  if (metricsLabel) {
    metricsLabel.textContent = mode === "mocap"
      ? "EXPOSURE-TIME MOCAP · REAR PROXY VIZ-ONLY"
      : mode === "depth"
        ? "RAW METRIC DEPTH × MOCAP PROJECTION"
      : mode === "fusion"
        ? "MOCAP-ROOT CONDITIONED METRICS"
        : "WRIST-PRESERVED DIAGNOSTIC METRICS";
  }

  configureMetricsSection(mode);

  const takes = currentTakes();
  if (takes.length) {
    populateTakePanels(takes);
    renderMetricsTable(takes);
  }

  for (const panel of takePanels) {
    const video = panel.querySelector("video");
    const source = video?.querySelector("source[data-fusion-src]");
    if (video) {
      video.pause();
      video.removeAttribute("poster");
    }
    if (source?.hasAttribute("src")) source.removeAttribute("src");
    if (video && source) video.load();
  }

  if (updateUrl) {
    const url = new URL(window.location.href);
    url.searchParams.set("mode", state.viewMode);
    window.history.replaceState(null, "", url);
  }
  activateTake(state.selectedTake);
}

function wireViewModes() {
  for (const button of viewModeButtons) {
    button.addEventListener("click", () => activateViewMode(button.dataset.viewMode));
  }
}

function wireTabs() {
  for (const [index, button] of takeButtons.entries()) {
    button.addEventListener("click", () => activateTake(button.dataset.take));
    button.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      let nextIndex = index;
      if (event.key === "ArrowLeft") nextIndex = (index - 1 + takeButtons.length) % takeButtons.length;
      if (event.key === "ArrowRight") nextIndex = (index + 1) % takeButtons.length;
      if (event.key === "Home") nextIndex = 0;
      if (event.key === "End") nextIndex = takeButtons.length - 1;
      activateTake(takeButtons[nextIndex].dataset.take, true);
    });
  }

}

function readInitialState() {
  const params = new URL(window.location.href).searchParams;
  const requestedMode = params.get("mode");
  state.viewMode = ["mocap", "depth", "fusion", "diagnostic"].includes(requestedMode)
    ? requestedMode
    : "mocap";
  const requestedTake = params.get("take");
  if (requestedTake && /^0?([123])$/.test(requestedTake)) {
    state.selectedTake = `take-00${Number(requestedTake) - 1}`;
  }
}

function populateTakePanels(takes) {
  for (const take of takes) {
    const panel = takePanels.find((candidate) => candidate.dataset.take === take.id);
    if (!panel) continue;

    const summary = panel.querySelector("[data-take-summary]");
    if (summary) summary.textContent = take.summary;
    const title = panel.querySelector("[data-take-title]");
    if (title) title.textContent = take.title;
    const kicker = panel.querySelector("[data-take-kicker]");
    if (kicker) kicker.textContent = `${take.label.toUpperCase()} · ${take.roleZh}`;
    const coverageLabel = panel.querySelector("[data-coverage-label]");
    if (coverageLabel) coverageLabel.textContent = take.coverageLabel || "有效覆盖";
    const badge = panel.querySelector("[data-role-badge]");
    if (badge) {
      badge.textContent = take.badgeLabel || badge.dataset.defaultLabel;
      badge.classList.toggle("depth", state.viewMode === "depth");
    }
    const metricsLink = panel.querySelector("[data-metrics-link]");
    if (metricsLink) metricsLink.href = take.metricsDownload;
    const alignmentLink = panel.querySelector("[data-alignment-link]");
    if (alignmentLink) {
      alignmentLink.hidden = !take.alignmentDownload;
      if (take.alignmentDownload) alignmentLink.href = take.alignmentDownload;
    }
    const rawVideoLink = panel.querySelector("[data-raw-mocap-video-link]");
    if (rawVideoLink) {
      rawVideoLink.hidden = !take.rawVideoDownload;
      if (take.rawVideoDownload) rawVideoLink.href = take.rawVideoDownload;
    }
    const rawPosterLink = panel.querySelector("[data-raw-mocap-poster-link]");
    if (rawPosterLink) {
      rawPosterLink.hidden = !take.rawPosterDownload;
      if (take.rawPosterDownload) rawPosterLink.href = take.rawPosterDownload;
    }

    const cards = panel.querySelector(".hand-metrics");
    if (cards) {
      cards.replaceChildren();
      for (const [index, cardData] of (take.cards || []).entries()) {
        const card = document.createElement("div");
        card.className = "hand-card";
        if (state.viewMode === "diagnostic" && take.id === "take-002" && index === 1) {
          card.classList.add("critical");
        }
        const label = document.createElement("span");
        label.textContent = cardData.label;
        const primary = document.createElement("strong");
        primary.textContent = typeof cardData.primary === "number"
          ? formatNumber(cardData.primary, Math.abs(cardData.primary) < 2 ? 3 : 2)
          : cardData.primary;
        const primaryLabel = document.createElement("small");
        primaryLabel.textContent = cardData.primaryLabel;
        const secondary = document.createElement("b");
        secondary.textContent = typeof cardData.secondary === "number"
          ? formatNumber(cardData.secondary, Math.abs(cardData.secondary) < 2 ? 3 : 2)
          : cardData.secondary;
        const secondaryLabel = document.createElement("small");
        secondaryLabel.textContent = cardData.secondaryLabel;
        card.append(label, primary, primaryLabel, secondary, secondaryLabel);
        cards.append(card);
      }
    }

    for (const node of panel.querySelectorAll("[data-field]")) {
      const path = node.dataset.field;
      const value = nestedValue(take, path);
      if (path === "coveragePercent") {
        node.textContent = `${formatNumber(value, 1)}% · ${take.validFrames}/${take.synchronizedFrames} frames`;
      } else if (typeof value === "number") {
        node.textContent = formatNumber(value);
      }
    }

    const progress = panel.querySelector("[data-progress]");
    if (progress) {
      progress.value = take.coveragePercent;
      progress.textContent = `${take.coveragePercent}%`;
      progress.setAttribute("aria-label", `${take.label} 有效覆盖 ${take.coveragePercent}%`);
    }
  }
}

function cell(text, className) {
  const td = document.createElement("td");
  td.textContent = text;
  if (className) td.className = className;
  return td;
}

function renderMetricsTable(takes) {
  const body = document.querySelector("#metrics-body");
  if (!body) return;
  body.replaceChildren();

  for (const take of takes) {
    const row = document.createElement("tr");
    const takeCell = document.createElement("td");
    const wrapper = document.createElement("div");
    wrapper.className = "take-cell";
    const label = document.createElement("strong");
    label.textContent = take.label;
    const role = document.createElement("span");
    role.textContent = take.roleZh;
    wrapper.append(label, role);
    takeCell.append(wrapper);

    if (state.viewMode === "mocap") {
      row.append(
        takeCell,
        cell(`${take.validFrames} (${formatNumber(take.coveragePercent, 1)}%)`),
        cell(`${take.contentEvidenceLabel} · 可辨 ${take.contentDistinguishableSameIndexCount}/${take.contentDistinguishableCount} 为 offset 0`),
        cell(`${formatNumber(take.captureLatencyMedianMs, 3)} / ${formatNumber(take.captureLatencyP95Ms, 3)} ms`),
        cell(`${formatNumber(take.visualLegacyCorr, 3)} → ${formatNumber(take.visualCorrectedCorr, 3)}`),
        cell(`${take.residualLagMs >= 0 ? "+" : ""}${take.residualLagMs} ms（未回写） · nearest P95 ${formatNumber(take.nearestP95Ms, 3)} ms`)
      );
    } else if (state.viewMode === "depth") {
      row.append(
        takeCell,
        cell(`${take.validFrames} (${formatNumber(take.coveragePercent, 1)}%)`),
        cell(take.timestampGatesPassed ? `PASS · depth ${take.depthMessageCount} / RGB ${take.rgbMessageCount}${take.depthExtraTailFrames ? ` · +${take.depthExtraTailFrames} tail 未强配` : ""}` : "FAIL"),
        cell(`${formatNumber(take.depthRgbMedianMs, 3)} / ${formatNumber(take.depthRgbP95Ms, 3)} ms`),
        cell(`${formatNumber(take.nearestMedianMs, 3)} ms / ${formatNumber(take.bracketMaxMs, 3)} ms`),
        cell(`${formatNumber(take.insideDepthFramePercent, 2)}% / ${formatNumber(take.nonzeroDepthAtJointPercent, 2)}%（非精度）`)
      );
    } else {
      row.append(
        takeCell,
        cell(`${take.validFrames}/${take.synchronizedFrames} (${formatNumber(take.coveragePercent, 1)}%)`),
        cell(`${formatNumber(take.nearestMocapMedianMs, 3)} / ${formatNumber(take.nearestMocapP95Ms, 3)} ms`),
        cell(`${formatNumber(take.left.jointMedianMm)} / ${formatNumber(take.left.tipMedianMm)} mm`),
        cell(
          `${formatNumber(take.right.jointMedianMm)} / ${formatNumber(take.right.tipMedianMm)} mm`,
          state.viewMode === "diagnostic" && take.id === "take-002" ? "value-critical" : ""
        ),
        cell(`${formatNumber(take.left.frameMpjpeMeanMm)} / ${formatNumber(take.right.frameMpjpeMeanMm)} mm`)
      );
    }
    body.append(row);
  }
}

function configureMetricsSection(mode) {
  const title = document.querySelector("#metrics-title");
  const intro = document.querySelector("#metrics-intro");
  const definition = document.querySelector("#metric-definition");
  const overview = document.querySelector("#fusion-overview");
  const sensitivity = document.querySelector("#fusion-sensitivity");
  const caption = document.querySelector("#metrics-caption");
  const headers = Array.from(document.querySelectorAll("[data-table-heading]"));
  if (mode === "mocap") {
    if (title) title.textContent = "原始 21 joints + 后腕模块显示 proxy";
    if (intro) intro.textContent = "时序、CMAvatar bracket 与原始 42 joint 坐标保持不变；仅额外画出 rear wrist module proxy。该点用于解释可见安装位置，不进入指标，也不是 GT。";
    if (definition) definition.innerHTML = "<strong>joints</strong> = 原始每手 21 点，不修改<br><strong>proxy</strong> = wrist 后向外推 0.8×middle-MCP 向量，仅 VIZ";
    if (caption) caption.textContent = "曝光时刻 MOCAP 逐帧匹配结果";
    const values = ["Take / 角色", "匹配帧", "BAG ↔ MP4 内容同帧", "capture→poll median / P95", "视觉 corr 旧 → 新", "残余峰值 / nearest MOCAP P95"];
    headers.forEach((node, index) => { node.textContent = values[index]; });
    if (overview) overview.hidden = true;
    if (sensitivity) sensitivity.hidden = true;
  } else if (mode === "depth") {
    if (title) title.textContent = "Raw metric depth 上的 MOCAP 时序与投影覆盖";
    if (intro) intro.textContent = "每帧使用 depth 自身设备时间戳和冻结的刚性 world→depth 外参。表格中的 nonzero depth 只表示该投影像素有传感器返回值，不是位姿精度。";
    if (definition) definition.innerHTML = "<strong>timestamp</strong> = depth Image.timestamp_usec<br><strong>coverage</strong> = 投影在画内 / 对应 raw depth 非零（非 accuracy）";
    if (caption) caption.textContent = "完整共同区间 raw depth–MOCAP 诊断";
    const values = ["Take / 角色", "共同区间 depth 帧", "Depth timestamp gate", "depth−RGB median / P95", "nearest MOCAP median / bracket max", "画内投影 / nonzero depth（非精度）"];
    headers.forEach((node, index) => { node.textContent = values[index]; });
    if (overview) overview.hidden = true;
    if (sensitivity) sensitivity.hidden = true;
  } else {
    if (title) title.textContent = "跨 Take 手指 articulation 一致性";
    if (intro) intro.textContent = "表格显示当前所选的 calibrated fusion 或 wrist diagnostic。数值越低越好，当前没有人为设置“通过阈值”。";
    if (definition) definition.innerHTML = "<strong>joint</strong> = 16 个非拇指关节 pooled EPE median<br><strong>tip</strong> = 4 个非拇指指尖 pooled EPE median";
    if (caption) caption.textContent = "严格 25 ms 当前所选模式结果";
    const values = ["Take / 角色", "保留帧", "nearest MOCAP median / P95", "左 joint / tip", "右 joint / tip", "左 / 右 frame MPJPE mean"];
    headers.forEach((node, index) => { node.textContent = values[index]; });
    if (overview) overview.hidden = false;
    if (sensitivity) sensitivity.hidden = false;
  }
}

function renderRequests(requests) {
  const list = document.querySelector("#request-list");
  if (!list) return;
  list.replaceChildren();

  for (const request of requests.filter((item) => item.priority === "P0")) {
    const item = document.createElement("li");
    const index = document.createElement("span");
    index.className = "request-index";
    index.textContent = request.id.replace("REQ-", "");

    const content = document.createElement("div");
    content.className = "request-content";
    const title = document.createElement("h3");
    title.textContent = request.item;
    const files = document.createElement("p");
    files.className = "request-files";
    files.textContent = request.files.replaceAll(";", " · ");
    const acceptance = document.createElement("p");
    acceptance.className = "request-acceptance";
    acceptance.textContent = `验收：${request.acceptance.replaceAll(";", "；")}`;
    content.append(title, files, acceptance);
    item.append(index, content);
    list.append(item);
  }
}

function renderLimitations(limitations) {
  const list = document.querySelector("#limitations-list");
  if (!list) return;
  list.replaceChildren();
  for (const limitation of limitations) {
    const item = document.createElement("li");
    item.textContent = limitation;
    list.append(item);
  }
}

async function copyRequestMessage() {
  if (!state.data?.copyMessage) {
    showToast("数据清单尚未载入，请下载 Markdown。");
    return;
  }

  try {
    await navigator.clipboard.writeText(state.data.copyMessage);
    showToast("已复制给数据方的短消息。");
  } catch {
    const textarea = document.createElement("textarea");
    textarea.value = state.data.copyMessage;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.append(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    textarea.remove();
    showToast(copied ? "已复制给数据方的短消息。" : "复制失败，请下载 Markdown。");
  }
}

function wireAnimations() {
  const items = document.querySelectorAll("[data-animate]");
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches || !("IntersectionObserver" in window)) {
    for (const item of items) item.classList.add("is-visible");
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    },
    { threshold: 0.08, rootMargin: "0px 0px -30px" }
  );
  for (const item of items) observer.observe(item);
}

async function loadSiteData() {
  const response = await fetch("/data/site-data.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`site-data returned HTTP ${response.status}`);
  const data = await response.json();
  if (
    data.schema !== "gt-calib.review-site.v2" ||
    !Array.isArray(data.mocapTakes) ||
    !Array.isArray(data.depthTakes) ||
    !Array.isArray(data.takes) ||
    !Array.isArray(data.diagnosticTakes)
  ) {
    throw new Error("site-data schema mismatch");
  }
  state.data = data;
  populateTakePanels(currentTakes());
  renderMetricsTable(currentTakes());
  activateTake(state.selectedTake);
  renderRequests(data.requests || []);
  renderLimitations(data.limitations || []);
}

async function main() {
  readInitialState();
  wireViewModes();
  wireTabs();
  activateViewMode(state.viewMode, false);
  wireAnimations();
  document.querySelector("#copy-request")?.addEventListener("click", copyRequestMessage);

  try {
    await loadSiteData();
  } catch (error) {
    console.error(error);
    const body = document.querySelector("#metrics-body");
    if (body) {
      body.replaceChildren();
      const row = document.createElement("tr");
      row.append(cell("指标数据载入失败；请直接下载 metrics JSON。"));
      row.firstChild.colSpan = 6;
      body.append(row);
    }
    const list = document.querySelector("#request-list");
    if (list) {
      list.replaceChildren();
      const item = document.createElement("li");
      item.className = "request-loading";
      item.textContent = "数据清单载入失败；请下载 CSV 或 Markdown。";
      list.append(item);
    }
    showToast("页面数据未完全载入，请使用下载区的原始文件。");
  }
}

main();
