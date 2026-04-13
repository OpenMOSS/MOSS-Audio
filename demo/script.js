const TASK_LABELS = {
  ASR: "Speech Recognition",
  "General Caption": "General Caption",
  "Music Caption": "Music Caption",
  "Speech Caption": "Speech Caption",
  "Time-Aware QA": "Time-Aware QA",
  "Emotion & Speaker": "Emotion & Speaker",
  "Speech Captioning": "Speech Captioning",
  "Audio Description": "Audio Description",
  "Audio QA": "Audio QA",
  "Timestamp ASR": "Timestamp ASR",
  "Complex Reasoning": "Complex Reasoning",
  "Music Understanding": "Music Understanding",
  "Sound Event Timeline": "Sound Event Timeline",
};

const TASK_ORDER = [
  "General Caption",
  "Time-Aware QA",
  "Complex Reasoning",
  "Music Caption",
  "Speech Caption",
];

const TASK_ICONS = {
  "General Caption": "📝",
  "Time-Aware QA": "⏱",
  "Complex Reasoning": "🧠",
  "Music Caption": "🎵",
  "Speech Caption": "🗣",
};

const DEMO_TABS_ROOT = document.querySelector("#demoTabs");
const DEMO_PANELS_ROOT = document.querySelector("#demoPanels");
const SAMPLE_CONTAINERS = new Map();

const DEMO_STATS = document.querySelector("#demoStats");
const DEMO_LEGEND = document.querySelector("#demoLegend");
let DEMO_TABS = [];

// ─── Reveal observer ──────────────────────────────────────
const revealObserver =
  typeof IntersectionObserver === "function"
    ? new IntersectionObserver(
        (entries, obs) => {
          entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            entry.target.classList.add("is-visible");
            obs.unobserve(entry.target);
          });
        },
        { threshold: 0, rootMargin: "0px 0px -6% 0px" },
      )
    : null;

function observeReveal(el) {
  if (el.dataset.revealBound) return;
  el.dataset.revealBound = "true";
  if (!revealObserver) {
    el.classList.add("is-visible");
    return;
  }
  revealObserver.observe(el);
}

function setupRevealAnimations() {
  document.querySelectorAll(".reveal").forEach(observeReveal);
}

// ─── Sidebar nav active state ─────────────────────────────
function setupSidebarActive() {
  const links = document.querySelectorAll(".sidebar-link");
  if (!links.length || typeof IntersectionObserver !== "function") return;

  const sectionIds = [
    "overview",
    "introduction",
    "features",
    "architecture",
    "evaluation",
    "demo",
    "quickstart",
  ];

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const id = entry.target.id;
        links.forEach((link) => {
          const href = link.getAttribute("href");
          const match =
            href === `#${id}` ||
            (href === "#overview" && id === "introduction");
          link.classList.toggle("active", match);
        });
      });
    },
    { threshold: 0.25, rootMargin: "-80px 0px -40% 0px" },
  );

  sectionIds.forEach((id) => {
    const el = document.getElementById(id);
    if (el) observer.observe(el);
  });
}

// ─── Tab switching ────────────────────────────────────────
function setupTabs() {
  function showTab(targetKey) {
    DEMO_TABS.forEach((tab) => {
      const isActive = tab.dataset.tab === targetKey;
      tab.classList.toggle("active", isActive);
      tab.setAttribute("aria-selected", isActive ? "true" : "false");
    });

    SAMPLE_CONTAINERS.forEach((panel, key) => {
      panel.style.display = key === targetKey ? "grid" : "none";
    });

    setupRevealAnimations();
  }

  const firstEnabledTab = DEMO_TABS.find((tab) => !tab.disabled);
  if (firstEnabledTab) showTab(firstEnabledTab.dataset.tab);

  DEMO_TABS.forEach((tab) => {
    tab.addEventListener("click", () => showTab(tab.dataset.tab));
  });
}

// ─── Asset URL ────────────────────────────────────────────
function buildAssetUrl(assetPath) {
  if (!assetPath) return "";
  if (/^https?:\/\//.test(assetPath)) return assetPath;

  const normalized = assetPath
    .replace(/^(?:\.\/)?(?:demo\/)?assets\//, "")
    .replace(/^\/+/, "");

  return encodeURI(`./assets/${normalized}`);
}

// ─── Text helpers ─────────────────────────────────────────
function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function normalizeString(value) {
  return typeof value === "string" ? value.trim() : "";
}

function normalizeWhitespace(value) {
  return value.replace(/\s+/g, " ").trim();
}

function truncateText(value, maxLength = 180) {
  if (value.length <= maxLength) return value;
  return `${value.slice(0, maxLength).trimEnd()}...`;
}

function shouldCollapseText(value) {
  return value.length > 220 || value.split("\n").length > 3;
}

function formatInlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function renderRichText(container, value) {
  const paragraphs = value.trim().split(/\n{2,}/);
  paragraphs.forEach((paragraph) => {
    const node = document.createElement("p");
    node.className = "rich-paragraph";
    node.innerHTML = formatInlineMarkdown(paragraph).replace(/\n/g, "<br>");
    container.append(node);
  });
}

// ─── DOM helpers ──────────────────────────────────────────
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (typeof text === "string") node.textContent = text;
  return node;
}

function createPill(text, variant = "") {
  const pill = el("span", `sample-pill${variant ? ` ${variant}` : ""}`, text);
  return pill;
}

function createAudioPanel(src) {
  const wrap = el("div", "audio-panel");
  const lbl = el("p", "panel-label", "Audio Input");
  const audio = document.createElement("audio");
  audio.controls = true;
  audio.preload = "none";
  audio.src = src;
  wrap.append(lbl, audio);
  return wrap;
}

function createImagePanel(src, alt) {
  if (!src) return null;

  const wrap = el("div", "image-panel");
  const lbl = el("p", "panel-label", "Paired Visual");
  const image = document.createElement("img");
  image.className = "sample-image";
  image.src = src;
  image.alt = alt || "Demo illustration";
  image.loading = "lazy";
  wrap.append(lbl, image);
  return wrap;
}

function createExpandableBlock(summaryText, value) {
  const details = document.createElement("details");
  details.className = "expandable-block";

  const summary = el("summary", "expandable-summary", summaryText);
  const body = el("pre", "expandable-body", value);

  details.append(summary, body);
  return details;
}

function createPromptPanel(prompt) {
  const wrap = el("div", "prompt-panel");
  const lbl = el("p", "panel-label", "Prompt");
  wrap.append(lbl);

  if (!prompt) {
    wrap.append(el("div", "panel-body prompt-body", "No prompt metadata."));
    return wrap;
  }

  const preview = normalizeWhitespace(prompt);
  wrap.append(
    el(
      "div",
      "panel-body prompt-body",
      shouldCollapseText(prompt) ? truncateText(preview, 220) : prompt,
    ),
  );

  if (shouldCollapseText(prompt)) {
    wrap.append(createExpandableBlock("Show full prompt", prompt));
  }

  return wrap;
}

function createAnalysisPanel(analysis) {
  if (!analysis) return null;
  const wrap = el("div", "analysis-panel");
  const lbl = el("p", "panel-label", "Analysis");
  wrap.append(lbl);

  const preview = normalizeWhitespace(analysis);
  wrap.append(
    el(
      "div",
      "panel-body analysis-body",
      shouldCollapseText(analysis) ? truncateText(preview, 260) : analysis,
    ),
  );

  if (shouldCollapseText(analysis)) {
    wrap.append(createExpandableBlock("Show full analysis", analysis));
  }

  return wrap;
}

function createResponsePanel(sample) {
  const wrap = el("div", "response-panel");
  const lbl = el("p", "panel-label", "Response");
  wrap.append(lbl);

  if (sample.responseFormat === "json" || sample.responseFormat === "timeline") {
    const body = el(
      "pre",
      `panel-body response-body response-pre response-${sample.responseFormat}`,
      sample.response,
    );
    wrap.append(body);
    return wrap;
  }

  const body = document.createElement("div");
  body.className = "panel-body response-body response-rich";
  renderRichText(body, sample.response);
  wrap.append(body);
  return wrap;
}

function createSampleMeta(sample) {
  const row = el("div", "sample-meta");

  if (sample.language) {
    row.append(createPill(sample.language, "sample-pill-muted"));
  }

  sample.tags.forEach((tag) => {
    row.append(createPill(tag));
  });

  return row.childNodes.length ? row : null;
}

function createSampleCard(sample, index) {
  const card = el("article", "sample-card reveal");
  card.style.setProperty("--delay", `${Math.min(index * 60, 300)}ms`);

  const header = el("header", "sample-header");
  const copy = document.createElement("div");
  const taskEl = el("p", "sample-task", sample.task);
  const title = el("h4", "sample-title", sample.name);
  const chip = el(
    "span",
    "status-chip",
    TASK_LABELS[sample.task] ?? sample.task,
  );

  copy.append(taskEl, title);
  header.append(copy, chip);
  card.append(header);

  const meta = createSampleMeta(sample);
  if (meta) card.append(meta);

  const imagePanel = createImagePanel(sample.imageUrl, `${sample.name} illustration`);
  if (imagePanel) card.append(imagePanel);

  card.append(createAudioPanel(sample.audioUrl), createPromptPanel(sample.prompt));

  const analysisPanel = createAnalysisPanel(sample.analysis);
  if (analysisPanel) card.append(analysisPanel);

  card.append(createResponsePanel(sample));

  return card;
}

function createStatCard(label, value, note) {
  const card = el("article", "demo-stat-card");
  const labelEl = el("p", "demo-stat-label", label);
  const valueEl = el("p", "demo-stat-value", value);
  const noteEl = el("p", "demo-stat-note", note);
  card.append(labelEl, valueEl, noteEl);
  return card;
}

// ─── Data parsing ─────────────────────────────────────────
function parseSample(line, index) {
  const raw = JSON.parse(line);

  return {
    id: normalizeString(raw.id) || `sample-${index + 1}`,
    category: normalizeString(raw.category) || "understanding",
    task: normalizeString(raw.task) || "Audio Description",
    name: normalizeString(raw.name) || `Sample ${index + 1}`,
    audioUrl: buildAssetUrl(normalizeString(raw.audio)),
    imageUrl: buildAssetUrl(normalizeString(raw.image)),
    prompt: normalizeString(raw.prompt),
    analysis: normalizeString(raw.analysis),
    response: normalizeString(raw.response),
    responseFormat: normalizeString(raw.response_format) || "richtext",
    language: normalizeString(raw.language),
    tags: Array.isArray(raw.tags)
      ? raw.tags.map((tag) => normalizeString(tag)).filter(Boolean)
      : [],
  };
}

// ─── Demo overview ────────────────────────────────────────
function getTaskKeys(samples) {
  const sampleTasks = Array.from(new Set(samples.map((sample) => sample.task)));
  const ordered = TASK_ORDER.filter((task) => sampleTasks.includes(task));
  const remaining = sampleTasks
    .filter((task) => !TASK_ORDER.includes(task))
    .sort((a, b) => a.localeCompare(b));
  return [...ordered, ...remaining];
}

function ensureTaskPanels(taskKeys) {
  if (!DEMO_TABS_ROOT || !DEMO_PANELS_ROOT) return;

  DEMO_TABS_ROOT.textContent = "";
  DEMO_PANELS_ROOT.textContent = "";
  SAMPLE_CONTAINERS.clear();

  taskKeys.forEach((task, index) => {
    const key = task;
    const panelId = `panel-${task.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;

    const tab = document.createElement("button");
    tab.type = "button";
    tab.className = `demo-tab${index === 0 ? " active" : ""}`;
    tab.dataset.tab = key;
    tab.dataset.label = TASK_LABELS[task] ?? task;
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-selected", index === 0 ? "true" : "false");
    tab.setAttribute("aria-controls", panelId);
    tab.innerHTML = `<span class="demo-tab-icon" aria-hidden="true">${TASK_ICONS[task] ?? "•"}</span><span>${TASK_LABELS[task] ?? task}</span><span class="demo-tab-count">0</span>`;
    DEMO_TABS_ROOT.append(tab);

    const panel = document.createElement("div");
    panel.id = panelId;
    panel.className = "sample-grid";
    panel.setAttribute("role", "tabpanel");
    panel.style.display = index === 0 ? "grid" : "none";
    DEMO_PANELS_ROOT.append(panel);

    SAMPLE_CONTAINERS.set(key, panel);
  });

  DEMO_TABS = Array.from(DEMO_TABS_ROOT.querySelectorAll(".demo-tab"));
}

function updateTabCounts(samples) {
  const counts = {};
  samples.forEach((sample) => {
    counts[sample.task] = (counts[sample.task] ?? 0) + 1;
  });

  DEMO_TABS.forEach((tab) => {
    const count = counts[tab.dataset.tab] ?? 0;
    const countEl = tab.querySelector(".demo-tab-count");
    if (countEl) countEl.textContent = String(count);
    tab.disabled = count === 0;
  });
}

function renderDemoOverview(samples) {
  if (DEMO_STATS) {
    DEMO_STATS.textContent = "";

    const uniqueTasks = new Set(samples.map((sample) => sample.task)).size;
    const uniqueLanguages = new Set(
      samples.map((sample) => sample.language).filter(Boolean),
    ).size;

    DEMO_STATS.append(
      createStatCard("Samples", String(samples.length), "Curated audio demos"),
      createStatCard("Task Families", String(uniqueTasks), "Across demo cards"),
      createStatCard("Visible Tabs", String(uniqueTasks), "Grouped by demo task"),
      createStatCard("Languages", String(uniqueLanguages), "Prompt or audio language"),
    );
  }

  if (DEMO_LEGEND) {
    DEMO_LEGEND.textContent = "";
    const counts = new Map();
    samples.forEach((sample) => {
      counts.set(sample.task, (counts.get(sample.task) ?? 0) + 1);
    });

    Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .forEach(([task, count]) => {
        DEMO_LEGEND.append(
          createPill(`${TASK_LABELS[task] ?? task} · ${count}`, "sample-pill-outline"),
        );
      });
  }

  ensureTaskPanels(getTaskKeys(samples));
  updateTabCounts(samples);
  setupTabs();
}

// ─── Sample rendering ─────────────────────────────────────
function renderSamples(samples) {
  SAMPLE_CONTAINERS.forEach((container) => {
    container.textContent = "";
  });

  const indexes = new Map();

  samples.forEach((sample) => {
    const target = SAMPLE_CONTAINERS.get(sample.task);
    if (!target) return;
    const index = indexes.get(sample.task) ?? 0;
    const card = createSampleCard(sample, index);
    indexes.set(sample.task, index + 1);
    target.append(card);
  });

  SAMPLE_CONTAINERS.forEach((target, task) => {
    if (!target || target.childNodes.length) return;
    target.append(
      el(
        "article",
        "notice-card",
        `No ${(TASK_LABELS[task] ?? task).toLowerCase()} demos available yet.`,
      ),
    );
  });
}

function renderError(message) {
  SAMPLE_CONTAINERS.forEach((container) => {
    container.textContent = "";
    container.append(el("article", "notice-card", message));
  });
}

// ─── Data loading ─────────────────────────────────────────
async function loadSamples() {
  const response = await fetch("./assets/metadata.jsonl");
  if (!response.ok) throw new Error(`HTTP ${response.status}`);

  const text = await response.text();
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map(parseSample);
}

// ─── General Audio Chart ──────────────────────────────────
const CHART_COLORS = {
  moss: "#ffab91",
  osSmall: "#4a90d9",
  osLarge: "#e7d58d",
  closed: "#b0bec5",
};

const CHART_DATA = [
  { name: "MiniCPM-o-4.5 (9B)", value: 56.83, group: "osSmall" },
  { name: "Audio Flamingo 3 (7B)", value: 57.73, group: "osSmall" },
  { name: "Qwen2.5-Omni (7B)", value: 58.96, group: "osSmall" },
  { name: "GPT4o-Audio", value: 59.13, group: "closed" },
  { name: "Kimi-Audio (7B)", value: 61.14, group: "osSmall" },
  { name: "MiMo-Audio-7B (8B)", value: 62.97, group: "osSmall" },
  { name: "MOSS-Audio-4B-Instruct", value: 64.04, group: "moss" },
  { name: "MOSS-Audio-8B-Instruct", value: 66.32, group: "moss" },
  { name: "Step-Audio-R1.1 (33B)", value: 66.48, group: "osLarge" },
  { name: "Qwen3-Omni-30B-A3B-Instruct", value: 67.91, group: "osLarge" },
  { name: "MOSS-Audio-4B-Thinking", value: 68.37, group: "moss" },
  { name: "Step-Audio-R1 (33B)", value: 70.67, group: "osLarge" },
  { name: "MOSS-Audio-8B-Thinking", value: 70.8, group: "moss" },
];

const SPEECH_CAPTION_DIMENSIONS = [
  "Gender",
  "Age",
  "Accent",
  "Pitch",
  "Volume",
  "Speed",
  "Texture",
  "Clarity",
  "Fluency",
  "Emotion",
  "Tone",
  "Personality",
  "Summary",
];

const SPEECH_CAPTION_RADAR_DATA = [
  {
    name: "Qwen3-Omni-30B-A3B-Instruct",
    color: "#4a90d9",
    values: [4.436, 3.936, 4.356, 3.59, 3.682, 3.614, 3.093, 3.521, 3.531, 3.328, 3.224, 3.292, 3.179],
  },
  {
    name: "Qwen3-Omni-30B-A3B-Thinking",
    color: "#7aa4d8",
    values: [4.419, 4.026, 4.327, 3.61, 3.577, 3.61, 3.179, 3.403, 3.526, 3.232, 3.154, 3.197, 3.107],
  },
  {
    name: "Gemini-3-Pro",
    color: "#b0bec5",
    values: [4.191, 3.835, 4.181, 3.392, 3.254, 3.32, 2.998, 3.347, 3.524, 3.055, 2.997, 3.023, 2.775],
  },
  {
    name: "Gemini-3.1-Pro",
    color: "#d1b86b",
    values: [4.436, 3.936, 4.356, 3.59, 3.682, 3.614, 3.093, 3.521, 3.531, 3.328, 3.224, 3.292, 3.179],
  },
  {
    name: "MOSS-Audio-4B-Instruct",
    color: "#ffb08a",
    values: [4.697, 3.98, 4.497, 3.628, 3.722, 3.564, 3.407, 3.841, 3.744, 3.311, 3.282, 3.305, 3.259],
  },
  {
    name: "MOSS-Audio-8B-Instruct",
    color: "#ff8f6b",
    values: [4.683, 3.979, 4.572, 3.682, 3.709, 3.638, 3.403, 3.869, 3.747, 3.314, 3.253, 3.272, 3.307],
  },
];

function hexToRgb(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return { r, g, b };
}

function lightenColor(hex, ratio) {
  const { r, g, b } = hexToRgb(hex);
  return `rgb(${Math.round(r + (255 - r) * ratio)},${Math.round(
    g + (255 - g) * ratio,
  )},${Math.round(b + (255 - b) * ratio)})`;
}

function createBarGradient(baseColor) {
  return {
    type: "linear",
    x: 0,
    y: 0,
    x2: 1,
    y2: 0,
    colorStops: [
      { offset: 0, color: baseColor },
      { offset: 0.6, color: lightenColor(baseColor, 0.35) },
      { offset: 1, color: lightenColor(baseColor, 0.58) },
    ],
  };
}

function initGeneralAudioChart() {
  const chartRoot = document.getElementById("generalAudioChart");
  if (!chartRoot || typeof echarts === "undefined") return;

  const chart = echarts.init(chartRoot, null, { renderer: "canvas" });
  const names = CHART_DATA.map((entry) => entry.name);

  const option = {
    backgroundColor: "#f9f7f3",
    legend: {
      top: 14,
      right: 16,
      itemWidth: 10,
      itemHeight: 10,
      itemGap: 14,
      borderRadius: 3,
      textStyle: {
        fontFamily: "Sora, sans-serif",
        fontSize: 11,
        color: "#2c251e",
      },
      data: [
        {
          name: "MOSS-Audio (Ours)",
          icon: "roundRect",
          itemStyle: { color: CHART_COLORS.moss },
        },
        {
          name: "Open Source (≤10B)",
          icon: "roundRect",
          itemStyle: { color: CHART_COLORS.osSmall },
        },
        {
          name: "Open Source (>10B)",
          icon: "roundRect",
          itemStyle: { color: CHART_COLORS.osLarge },
        },
        {
          name: "Closed Source",
          icon: "roundRect",
          itemStyle: { color: CHART_COLORS.closed },
        },
      ],
    },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      backgroundColor: "#1a1a1a",
      borderColor: "#1a1a1a",
      padding: [10, 14],
      textStyle: {
        color: "#fff",
        fontFamily: "Sora, sans-serif",
        fontSize: 12,
      },
      formatter(params) {
        const point = params[0];
        const datum = CHART_DATA[point.dataIndex];
        const groupLabels = {
          moss: "MOSS-Audio (Ours)",
          osSmall: "Open Source (≤10B)",
          osLarge: "Open Source (>10B)",
          closed: "Closed Source",
        };
        return (
          `<div style="font-weight:500;margin-bottom:5px">${datum.name}</div>` +
          `<div>Avg Accuracy: <strong>${datum.value.toFixed(2)}</strong></div>` +
          `<div style="color:#999;font-size:11px;margin-top:3px">${groupLabels[datum.group]}</div>`
        );
      },
    },
    // containLabel keeps long y-axis labels inside the canvas
    grid: { left: "2%", right: 58, top: 50, bottom: 40, containLabel: true },
    xAxis: {
      type: "value",
      min: 50,
      max: 72,
      interval: 5,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: {
        color: "#8b8680",
        fontFamily: "Sora, sans-serif",
        fontSize: 11,
      },
      splitLine: { lineStyle: { color: "#e0ddd6", width: 0.5 } },
      name: "Average Accuracy",
      nameLocation: "middle",
      nameGap: 30,
      nameTextStyle: {
        fontFamily: "Sora, sans-serif",
        fontSize: 12,
        fontWeight: 500,
        color: "#2c251e",
      },
    },
    yAxis: {
      type: "category",
      data: names,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: {
        color: "#2c251e",
        fontFamily: "Sora, sans-serif",
        fontSize: 11,
        // Align labels to right, fixed width prevents layout shift
        width: 220,
        overflow: "truncate",
        formatter(name) {
          return name.startsWith("MOSS") ? `{moss|${name}}` : `{normal|${name}}`;
        },
        rich: {
          normal: {
            fontWeight: 500,
            color: "#2c251e",
            fontFamily: "Sora, sans-serif",
            fontSize: 11,
            align: "right",
          },
          moss: {
            fontWeight: 700,
            color: "#b84e18",
            fontFamily: "Sora, sans-serif",
            fontSize: 11,
            align: "right",
          },
        },
      },
    },
    series: [
      {
        type: "bar",
        barWidth: 15,
        barCategoryGap: "35%",
        data: CHART_DATA.map((entry) => ({
          value: entry.value,
          name: entry.name,
          itemStyle: {
            color: createBarGradient(CHART_COLORS[entry.group]),
            borderRadius: [0, 3, 3, 0],
          },
          label: {
            show: true,
            position: "right",
            distance: 6,
            formatter: entry.value.toFixed(2),
            fontSize: 11,
            fontFamily: "Sora, sans-serif",
            fontWeight: entry.group === "moss" ? 700 : 500,
            color: entry.group === "moss" ? "#b84e18" : "#8b8680",
          },
        })),
        z: 2,
      },
    ],
  };

  chart.setOption(option);
  window.addEventListener("resize", () => chart.resize());
}

function initSpeechCaptionRadarChart() {
  const chartRoot = document.getElementById("speechCaptionRadarChart");
  if (!chartRoot || typeof echarts === "undefined") return;

  const chart = echarts.init(chartRoot, null, { renderer: "canvas" });

  // Per-dimension max = min(dimension_max_across_all_models + 0.2, 5)
  // This makes the outer ring tight to the actual data while never exceeding 5.
  const indicators = SPEECH_CAPTION_DIMENSIONS.map((name, i) => {
    const dimMax = Math.max(...SPEECH_CAPTION_RADAR_DATA.map((m) => m.values[i]));
    return {
      name,
      min: 1.5,
      max: Math.min(dimMax + 0.2, 5),
    };
  });

  // MOSS is the last entry; render it on top with a thicker, more vibrant line
  const isMoss = (name) => name.startsWith("MOSS");

  const option = {
    backgroundColor: "#f9f7f3",
    color: SPEECH_CAPTION_RADAR_DATA.map((item) => item.color),
    legend: {
      // Move to bottom so the legend does not overlap the radar polygon
      bottom: 10,
      left: "center",
      orient: "horizontal",
      itemWidth: 10,
      itemHeight: 10,
      itemGap: 16,
      icon: "circle",
      textStyle: {
        fontFamily: "Sora, sans-serif",
        fontSize: 11,
        color: "#2c251e",
      },
    },
    tooltip: {
      trigger: "item",
      backgroundColor: "#1a1a1a",
      borderColor: "#1a1a1a",
      padding: [10, 14],
      extraCssText: "max-width:240px;white-space:normal",
      textStyle: {
        color: "#fff",
        fontFamily: "Sora, sans-serif",
        fontSize: 11,
      },
      formatter(params) {
        const rows = SPEECH_CAPTION_DIMENSIONS.map((dim, i) => {
          const v = Number(params.value[i]).toFixed(3);
          const isBest = SPEECH_CAPTION_RADAR_DATA.every(
            (other) => other.name === params.seriesName || other.values[i] <= params.value[i],
          );
          return `<div style="display:flex;justify-content:space-between;gap:16px">
            <span style="color:#aaa">${dim}</span>
            <span style="font-weight:${isBest ? 700 : 400};color:${isBest ? "#ffab91" : "#fff"}">${v}</span>
          </div>`;
        });
        return `<div style="font-weight:600;margin-bottom:8px;font-size:12px">${params.seriesName}</div>${rows.join("")}`;
      },
    },
    radar: {
      // With legend at bottom (~40px) and top padding (~12px), center at 48%
      // gives the polygon room on all sides
      center: ["50%", "48%"],
      radius: "68%",
      startAngle: 90,
      splitNumber: 5,
      shape: "polygon",
      axisName: {
        color: "#2c251e",
        fontFamily: "Sora, sans-serif",
        fontSize: 11,
        fontWeight: 500,
        // Give axis labels breathing room from the polygon edge
        padding: [0, 4],
      },
      splitLine: {
        lineStyle: { color: "rgba(44, 37, 30, 0.10)", width: 0.8 },
      },
      splitArea: {
        areaStyle: {
          color: [
            "rgba(255,255,255,0.68)",
            "rgba(44,37,30,0.018)",
            "rgba(255,255,255,0.68)",
            "rgba(44,37,30,0.018)",
            "rgba(255,255,255,0.68)",
          ],
        },
      },
      axisLine: {
        lineStyle: { color: "rgba(44, 37, 30, 0.13)", width: 0.8 },
      },
      indicator: indicators,
      nameGap: 6,
    },
    series: [
      {
        type: "radar",
        symbol: "circle",
        symbolSize: 4,
        emphasis: {
          lineStyle: { width: 3 },
          itemStyle: { borderWidth: 2 },
        },
        data: SPEECH_CAPTION_RADAR_DATA.map((item) => ({
          value: item.values,
          name: item.name,
          lineStyle: {
            color: item.color,
            // MOSS line is noticeably thicker and fully opaque; others are subtler
            width: isMoss(item.name) ? 2.8 : 1.4,
            opacity: isMoss(item.name) ? 1 : 0.65,
          },
          itemStyle: {
            color: item.color,
            opacity: isMoss(item.name) ? 1 : 0.6,
          },
          areaStyle: {
            color: item.color,
            opacity: isMoss(item.name) ? 0.18 : 0.04,
          },
        })),
      },
    ],
  };

  chart.setOption(option);
  window.addEventListener("resize", () => chart.resize());
}

// ─── Init ─────────────────────────────────────────────────
async function init() {
  setupRevealAnimations();
  setupSidebarActive();
  setupTabs();
  initGeneralAudioChart();
  initSpeechCaptionRadarChart();

  try {
    const samples = await loadSamples();
    renderDemoOverview(samples);
    renderSamples(samples);
    setupRevealAnimations();
  } catch (error) {
    console.error(error);
    renderError("Unable to load the demo samples right now.");
  }
}

init();
