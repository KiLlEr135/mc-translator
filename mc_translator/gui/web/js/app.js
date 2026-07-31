/* MC Translator — frontend logic. Talks to the Python backend exclusively
   through pywebview.api.* (see mc_translator/gui/bridge.py for the exact
   contract) and window.onBridgeBatch (Python -> JS push events). */

const MAX_LOG_LINES = 2000;

window.onerror = function (message, source, lineno, colno, error) {
  window.__lastError = `${message} @ ${source}:${lineno}:${colno}`;
};
window.addEventListener("unhandledrejection", (event) => {
  window.__lastError = `unhandledrejection: ${event.reason}`;
});

const el = (id) => document.getElementById(id);
const qs = (sel, root = document) => root.querySelector(sel);
const qsa = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
  running: false,
};

// ---------------------------------------------------------------------
// Initial load
// ---------------------------------------------------------------------

let _initialized = false;

async function init() {
  // The pywebviewready listener and the "already loaded" fallback below can
  // both fire (if window.pywebview exists by the time this script runs but
  // the pywebviewready event hasn't dispatched yet) -- without this guard,
  // wireEvents() ran twice and every button ended up with two listeners
  // (e.g. one folder-select click opened the native dialog twice).
  if (_initialized) return;
  _initialized = true;

  const initial = await window.pywebview.api.get_initial_state();

  populateSelect(el("select-lang"), initial.languages, initial.session.language);
  populateSelect(el("select-mcver"), initial.mcVersions, initial.session.mcVersion);
  el("folder-label").textContent = shortenPath(initial.mcDir);

  setRadioValue("output-mode", initial.session.outputMode);
  el("input-pack-name").value = initial.session.packName || "MCTranslator_Pack";
  el("chk-mods").checked = !!initial.session.translateMods;
  el("chk-books").checked = !!initial.session.translateBooks;
  el("chk-quests").checked = !!initial.session.translateQuests;
  setRadioValue("engine", initial.session.engine);
  setRadioValue("google-mode", initial.session.googleMode);
  setRadioValue("ai-provider", initial.aiProvider);
  setRadioValue("ai-mode", initial.session.aiMode);
  setRadioValue("process-mode", initial.session.processMode);
  setAutoCycleButton(!!initial.openrouterAutoCycle);
  populateCustomPresetSelect();

  updateOutputUi();
  updateEngineUi();
  document.title = `MC Translator v${initial.version}`;

  wireEvents();
  await window.pywebview.api.notify_ready();
}

function populateSelect(selectEl, values, selected) {
  selectEl.innerHTML = "";
  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    if (v === selected) opt.selected = true;
    selectEl.appendChild(opt);
  }
}

function setRadioValue(name, value) {
  const input = qs(`input[name="${name}"][value="${value}"]`);
  if (input) input.checked = true;
}

function getRadioValue(name) {
  const input = qs(`input[name="${name}"]:checked`);
  return input ? input.value : null;
}

function shortenPath(path) {
  if (!path) return "Не выбрана";
  return path.length > 40 ? `...${path.slice(-37)}` : path;
}

// ---------------------------------------------------------------------
// Sub-panel show/hide (animation #3)
// ---------------------------------------------------------------------

function updateOutputUi() {
  const mode = getRadioValue("output-mode");
  el("input-pack-name").disabled = mode !== "resourcepack";
}

function updateEngineUi() {
  const engine = getRadioValue("engine");
  el("panel-google").classList.toggle("open", engine === "google");
  el("panel-ai").classList.toggle("open", engine === "ai");
  updateAiProviderUi();
}

// The auto-cycle button only makes sense for engine=ai + provider=openrouter
// -- hidden otherwise (local KoboldCPP has no concept of a model list, and
// the button lives inside the shared #panel-ai sub-panel next to the
// provider radios rather than a provider-specific sub-panel of its own).
function updateAiProviderUi() {
  const engine = getRadioValue("engine");
  const provider = getRadioValue("ai-provider");
  el("btn-or-autocycle").style.display = engine === "ai" && provider === "openrouter" ? "" : "none";
}

function setAutoCycleButton(on) {
  el("btn-or-autocycle").textContent = on
    ? "🔁 Авто-перебор бесплатных моделей: ВКЛ"
    : "🔁 Авто-перебор бесплатных моделей: ВЫКЛ";
}

// ---------------------------------------------------------------------
// Log / status / row rendering
// ---------------------------------------------------------------------

function _appendToLog(element) {
  const panel = el("log-panel");
  const atBottom = panel.scrollHeight - panel.scrollTop - panel.clientHeight < 24;
  panel.appendChild(element);
  while (panel.children.length > MAX_LOG_LINES) {
    panel.removeChild(panel.firstChild);
  }
  if (atBottom) panel.scrollTop = panel.scrollHeight;
}

function appendLogLine(message, tag) {
  const p = document.createElement("p");
  p.className = `log-line tag-${tag || "white"}`;
  p.textContent = message;
  _appendToLog(p);

  if (tag === "green" && message.startsWith("✅")) {
    celebrate(message);
  }
}

function appendLogRow(row) {
  const p = document.createElement("p");
  p.className = "log-line";
  const pctClass = row.pct >= 90 ? "log-row-pct-green" : row.pct >= 50 ? "log-row-pct-yellow" : "log-row-pct-red";
  p.innerHTML =
    `<span class="log-row-icon">${row.icon} ${escapeHtml(row.name)}</span> ` +
    `<span class="log-row-kind">[${escapeHtml(row.kind)}]</span> ` +
    `${row.translated}/${row.total} ` +
    `<span class="${pctClass}">${row.pct}%</span>`;
  _appendToLog(p);
}

function appendTranslationLine(payload) {
  const state_ = { source: payload.source, translated: payload.translated, engine: payload.engine, language: payload.language };

  const p = document.createElement("p");
  p.className = "log-line log-translation";

  const sourceSpan = document.createElement("span");
  sourceSpan.className = "log-translation-source";
  sourceSpan.textContent = state_.source;

  const arrow = document.createElement("span");
  arrow.className = "log-translation-arrow";
  arrow.textContent = " → ";

  const valueSpan = document.createElement("span");
  valueSpan.className = "log-translation-value";
  valueSpan.textContent = state_.translated;
  valueSpan.title = "Нажмите, чтобы изменить перевод";
  valueSpan.addEventListener("click", () =>
    startEditingTranslatable(p, valueSpan, state_, (result) => {
      editsSinceLastRun = result.editsSinceLastRun;
      updateRebuildButton();
    })
  );

  const searchLink = document.createElement("span");
  searchLink.className = "log-translation-search";
  searchLink.textContent = "🔍";
  searchLink.title = "Показать все вхождения в «Проверить переводы»";
  searchLink.addEventListener("click", (e) => {
    e.stopPropagation();
    openReviewScreenWithSearch(state_.source, state_.language);
  });

  p.append(sourceSpan, arrow, valueSpan, searchLink);
  _appendToLog(p);
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function clearLog() {
  el("log-panel").innerHTML = "";
}

function setStatus(text, progress) {
  if (progress !== null && progress !== undefined) {
    el("progress-fill").style.width = `${Math.max(0, Math.min(1, progress)) * 100}%`;
  }
  el("status-label").textContent = text;
}

function setJobState(running, kind) {
  state.running = running;
  el("btn-analyze").disabled = running;
  el("btn-start").disabled = running;
  el("btn-stop").disabled = !running;
  el("btn-pause").disabled = !running;
  // Clearing the cache mid-run would race the job's own in-memory usage_log
  // snapshot (only saved once, at the very end) and could wipe entries the
  // job already produced this run -- see bridge.py's clear_cache guard.
  el("btn-clear-cache").disabled = running;
  if (!running) {
    el("btn-pause").textContent = "⏸ ПАУЗА";
    el("btn-pause").classList.remove("btn-pause-active");
  }
  el("busy-sprite").classList.toggle("active", running);
  el("log-panel").classList.toggle("job-running", running);
  // Starting a *translation* run is exactly what "Пересобрать пак" would
  // itself trigger — the server resets its own edit counter the same moment
  // (bridge.py's start_translation), so mirror that here. start_analysis
  // never touches the counter server-side, so don't reset for kind
  // "analysis" either, or the two would drift out of sync.
  if (running && kind === "translation") editsSinceLastRun = 0;
  updateRebuildButton();
}

// A download (potentially multi-gigabyte) uses the same status bar/progress
// fill as a translation job (see ai_setup_api.py's on_status calls) -- block
// the ordinary job buttons while it's active, mirroring setJobState, and
// re-enable them once it finishes (mirrors bridge.py's symmetric guard on
// start_translation/start_analysis while a download is active).
function setAiDownloadState(active) {
  el("btn-analyze").disabled = active;
  el("btn-start").disabled = active;
  el("btn-auto-setup-ai").disabled = active;
  el("busy-sprite").classList.toggle("active", active);
  if (!active) setStatus("Ожидание...", 0);
}

function celebrate(message) {
  const overlay = el("celebrate-overlay");
  const banner = el("celebrate-banner");
  banner.textContent = message;
  overlay.classList.remove("show");
  // Force reflow so re-triggering the animation works on consecutive runs.
  void overlay.offsetWidth;
  overlay.classList.add("show");
}

// ---------------------------------------------------------------------
// Bridge push events (batched — see bridge.py's _EventBatcher)
// ---------------------------------------------------------------------

window.onBridgeBatch = function onBridgeBatch(events) {
  for (const evt of events) {
    switch (evt.type) {
      case "log":
        appendLogLine(evt.payload.message, evt.payload.tag);
        break;
      case "row":
        appendLogRow(evt.payload);
        break;
      case "translation":
        appendTranslationLine(evt.payload);
        break;
      case "status":
        setStatus(evt.payload.text, evt.payload.progress);
        break;
      case "clearLog":
        clearLog();
        break;
      case "jobState":
        setJobState(evt.payload.running, evt.payload.kind);
        break;
      case "downloadState":
        setAiDownloadState(evt.payload.active);
        break;
      default:
        break;
    }
  }
};

// ---------------------------------------------------------------------
// Confirm modal (themed replacement for messagebox.askyesno)
// ---------------------------------------------------------------------

function confirmDialog(title, message) {
  return new Promise((resolve) => {
    el("confirm-title").textContent = title;
    el("confirm-message").textContent = message;
    const backdrop = el("confirm-backdrop");
    backdrop.classList.add("visible");

    function cleanup(result) {
      backdrop.classList.remove("visible");
      el("confirm-yes").removeEventListener("click", onYes);
      el("confirm-no").removeEventListener("click", onNo);
      resolve(result);
    }
    function onYes() { cleanup(true); }
    function onNo() { cleanup(false); }
    el("confirm-yes").addEventListener("click", onYes);
    el("confirm-no").addEventListener("click", onNo);
  });
}

// ---------------------------------------------------------------------
// Selective cache-clear modal -- "Весь кэш" keeps the original one-click
// wipe-everything behavior; "По модам"/"По типам" build a scope object from
// bridge.py's get_cache_scopes() (derived from the usage log) and pass it to
// clear_cache(scope), which only touches matching entries.
// ---------------------------------------------------------------------

async function openCacheClearModal() {
  el("cache-clear-lang-name").textContent = el("select-lang").value;
  setRadioValue("cache-clear-lang", "current");
  setRadioValue("cache-clear-mode", "all");
  updateCacheClearUi();
  await refreshCacheClearChecklist();
  el("cache-clear-backdrop").classList.add("visible");
}

function closeCacheClearModal() {
  el("cache-clear-backdrop").classList.remove("visible");
}

function updateCacheClearUi() {
  const mode = getRadioValue("cache-clear-mode");
  const showChecklist = mode === "mods" || mode === "kinds";
  el("cache-clear-checklist").style.display = showChecklist ? "flex" : "none";
  el("cache-clear-empty").style.display = "none";
  el("cache-clear-hint").textContent = mode === "all"
    ? "Весь накопленный кэш переводов будет очищен из памяти, а файлы кэша переименованы в резервные копии (*.bak-дата) рядом с ними."
    : "Будут удалены только отмеченные записи — остальной кэш останется цел. Затронутые файлы кэша будут скопированы в резервные копии (*.bak-дата) перед изменением.";
}

let _cacheClearRequestId = 0;

async function refreshCacheClearChecklist() {
  const mode = getRadioValue("cache-clear-mode");
  const container = el("cache-clear-checklist");
  container.innerHTML = "";
  if (mode !== "mods" && mode !== "kinds") return;
  const allLangs = getRadioValue("cache-clear-lang") === "all";
  // Pass the dropdown's current language explicitly rather than letting
  // bridge.py read SESSION -- SESSION only updates when a run actually
  // starts, so switching languages here without starting a run used to
  // build the checklist from the wrong (stale/default) language's usage log.
  // Rapidly toggling mode/language fires overlapping calls here with no
  // guaranteed response order (pywebview runs each js_api call on its own
  // thread) -- a generation token discards a response that's no longer for
  // the CURRENT selection, instead of letting it render on top of (or
  // instead of) the latest one.
  const requestId = ++_cacheClearRequestId;
  const scopes = await window.pywebview.api.get_cache_scopes(allLangs, el("select-lang").value);
  if (requestId !== _cacheClearRequestId) return;
  container.innerHTML = "";
  const values = mode === "mods" ? scopes.mods : scopes.kinds;
  el("cache-clear-empty").style.display = values.length === 0 ? "block" : "none";
  for (const v of values) {
    const label = document.createElement("label");
    label.className = "mc-checkbox";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = v;
    const span = document.createElement("span");
    span.textContent = v;
    label.appendChild(input);
    label.appendChild(span);
    container.appendChild(label);
  }
}

function wireCacheClearModal() {
  qsa('input[name="cache-clear-mode"]').forEach((r) => r.addEventListener("change", async () => {
    updateCacheClearUi();
    await refreshCacheClearChecklist();
  }));
  qsa('input[name="cache-clear-lang"]').forEach((r) => r.addEventListener("change", refreshCacheClearChecklist));
  el("cache-clear-cancel").addEventListener("click", closeCacheClearModal);
  el("cache-clear-confirm").addEventListener("click", async () => {
    const mode = getRadioValue("cache-clear-mode");
    const allLangs = getRadioValue("cache-clear-lang") === "all";
    let scope;
    if (mode === "all") {
      scope = { mode: "all" };
    } else {
      const values = qsa("#cache-clear-checklist input:checked").map((c) => c.value);
      if (values.length === 0) {
        appendLogLine("⚠️ Выберите хотя бы один пункт для очистки.", "yellow");
        return;
      }
      scope = { mode, values, languages: allLangs ? "all" : [el("select-lang").value] };
    }
    closeCacheClearModal();
    const result = await window.pywebview.api.clear_cache(scope);
    if (!result.ok) {
      for (const err of result.errors) appendLogLine(`⚠️ ${err}`, "yellow");
      return;
    }
    if (mode === "all") {
      appendLogLine("🧹 Кэш успешно очищен. Старые файлы сохранены как резервные копии (*.bak-дата).", "yellow");
    } else {
      appendLogLine(`🧹 Удалено записей из кэша: ${result.removed}. Резервные копии затронутых файлов сохранены (*.bak-дата).`, "yellow");
    }
  });
}

// ---------------------------------------------------------------------
// Settings drawer
// ---------------------------------------------------------------------

// Quick-fill presets for the "Другой ИИ" tab -- these are all OpenAI-
// compatible chat-completions APIs (same request/response shape
// CustomApiEngine already speaks server-side), so picking one here only
// fills in the URL/model placeholder client-side; nothing provider-specific
// is persisted or sent to Python. Claude (Anthropic) is NOT in this list --
// it's a different protocol entirely and has its own dedicated tab/engine.
const CUSTOM_AI_PRESETS = [
  { value: "", label: "— свой (введите URL вручную) —", url: "", modelHint: "" },
  { value: "nvidia", label: "NVIDIA NIM", url: "https://integrate.api.nvidia.com/v1/chat/completions", modelHint: "напр. meta/llama-3.1-70b-instruct" },
  { value: "openai", label: "OpenAI", url: "https://api.openai.com/v1/chat/completions", modelHint: "напр. gpt-4o-mini" },
  { value: "gemini", label: "Google Gemini", url: "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", modelHint: "напр. gemini-2.0-flash" },
  { value: "grok", label: "xAI Grok", url: "https://api.x.ai/v1/chat/completions", modelHint: "напр. grok-2-latest" },
];

function populateCustomPresetSelect() {
  const select = el("set-custom-preset");
  select.innerHTML = "";
  for (const preset of CUSTOM_AI_PRESETS) {
    const opt = document.createElement("option");
    opt.value = preset.value;
    opt.textContent = preset.label;
    select.appendChild(opt);
  }
}

function applyCustomPreset() {
  const preset = CUSTOM_AI_PRESETS.find((p) => p.value === el("set-custom-preset").value);
  if (!preset || !preset.value) return;
  el("set-custom-url").value = preset.url;
  el("set-custom-model").placeholder = preset.modelHint;
}

async function openSettingsDrawer() {
  const s = await window.pywebview.api.get_settings();
  el("set-ai-exe").value = s.aiExePath || "";
  el("set-ai-model").value = s.aiModelPath || "";
  el("set-ai-context").value = s.aiGlobalContext || "";
  el("set-gpu-layers").value = s.aiGpuLayers;
  el("lbl-gpu-val").textContent = s.aiGpuLayers;
  el("set-or-key").value = s.openrouterApiKey || "";
  el("set-or-model").value = s.openrouterModel || "";
  el("set-or-site").value = s.openrouterSiteUrl || "";
  el("set-or-app").value = s.openrouterAppName || "";
  el("set-or-free-models").value = s.openrouterFreeModels || "";
  el("set-custom-preset").value = "";
  el("set-custom-url").value = s.customBaseUrl || "";
  el("set-custom-key").value = s.customApiKey || "";
  el("set-custom-model").value = s.customModel || "";
  el("set-anthropic-key").value = s.anthropicApiKey || "";
  el("set-anthropic-model").value = s.anthropicModel || "";
  el("set-smart-glue").checked = !!s.smartGlue;
  el("set-google-workers").value = s.googleWorkers;
  el("lbl-workers-val").textContent = s.googleWorkers;
  el("set-deepl-key").value = s.deeplKey || "";
  renderFileTypesList(s.fileTypes || []);

  el("drawer-backdrop").classList.add("visible");
  el("settings-drawer").classList.add("open");
}

// Renders the "Типы файлов" tab's checklist from bridge.py's get_settings()
// fileTypes array ({key, label, enabled}) -- an additional AND-filter over
// the sidebar's 3 coarse translate_mods/books/quests toggles, not a
// replacement for them.
function renderFileTypesList(fileTypes) {
  const container = el("filetypes-list");
  container.innerHTML = "";
  for (const ft of fileTypes) {
    const label = document.createElement("label");
    label.className = "mc-checkbox";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.dataset.filetypeKey = ft.key;
    input.checked = !!ft.enabled;
    const span = document.createElement("span");
    span.textContent = ft.label;
    label.appendChild(input);
    label.appendChild(span);
    container.appendChild(label);
  }
}

function collectFileTypes() {
  return qsa("#filetypes-list input[type=checkbox]").map((input) => ({
    key: input.dataset.filetypeKey,
    enabled: input.checked,
  }));
}

function closeSettingsDrawer() {
  el("drawer-backdrop").classList.remove("visible");
  el("settings-drawer").classList.remove("open");
}

async function saveSettingsDrawer() {
  await window.pywebview.api.save_settings({
    aiExePath: el("set-ai-exe").value,
    aiModelPath: el("set-ai-model").value,
    aiGlobalContext: el("set-ai-context").value,
    aiGpuLayers: parseInt(el("set-gpu-layers").value, 10),
    openrouterApiKey: el("set-or-key").value,
    openrouterModel: el("set-or-model").value,
    openrouterSiteUrl: el("set-or-site").value,
    openrouterAppName: el("set-or-app").value,
    openrouterFreeModels: el("set-or-free-models").value,
    customBaseUrl: el("set-custom-url").value,
    customApiKey: el("set-custom-key").value,
    customModel: el("set-custom-model").value,
    anthropicApiKey: el("set-anthropic-key").value,
    anthropicModel: el("set-anthropic-model").value,
    smartGlue: el("set-smart-glue").checked,
    googleWorkers: parseInt(el("set-google-workers").value, 10),
    deeplKey: el("set-deepl-key").value,
    fileTypes: collectFileTypes(),
  });
  closeSettingsDrawer();
}

// ---------------------------------------------------------------------
// Auto-setup local AI -- detect_ai_hardware() (bridge.py, via
// gui/ai_setup_api.py) probes the GPU and resolves a model+engine plan
// without downloading anything; this builds the confirmation dialog from
// that plan, and download_ai_bundle() executes it in the background on
// confirm (progress reuses the main status bar via the usual on_status
// push -- see setAiDownloadState above).
// ---------------------------------------------------------------------

async function autoSetupAi() {
  const btn = el("btn-auto-setup-ai");
  btn.disabled = true;
  let info;
  try {
    info = await window.pywebview.api.detect_ai_hardware();
  } finally {
    btn.disabled = false;
  }
  if (!info || !info.ok) {
    appendLogLine(`❌ ${(info && info.error) || "Не удалось определить оборудование."}`, "red");
    return;
  }

  const gpuLine = info.gpu.vendor === "none"
    ? "Видеокарта не обнаружена — будет использован режим CPU (медленно)."
    : `Видеокарта: ${info.gpu.name} (${info.gpu.vramGb} ГБ видеопамяти)`;

  let message =
    `Обнаружено оборудование:\n• ${gpuLine}\n\n` +
    `Будет загружено:\n` +
    `• Модель: ${info.model.label} — ${info.model.sizeGb} ГБ\n` +
    `• Движок: ${info.engine.name} — ${info.engine.sizeGb} ГБ\n` +
    `• Итого: ${info.totalSizeGb} ГБ\n\n` +
    `Файлы сохранятся в папку «AI» рядом с программой, настройки обновятся автоматически.`;
  if (info.overwriteFiles && info.overwriteFiles.length) {
    message += `\n⚠️ Существующие файлы будут перезаписаны: ${info.overwriteFiles.join(", ")}.`;
  }
  if (!info.enoughDisk) {
    message += `\n⚠️ Похоже, на диске недостаточно свободного места.`;
  }
  message += `\n\nПродолжить?`;

  const proceed = await confirmDialog("Автонастройка локального ИИ", message);
  if (!proceed) return;

  closeSettingsDrawer();
  const result = await window.pywebview.api.download_ai_bundle();
  if (result && result.error) {
    appendLogLine(`❌ ${result.error}`, "red");
  }
}

// ---------------------------------------------------------------------
// Translation review screen
// ---------------------------------------------------------------------

const REVIEW_PAGE_SIZE = 300;

// How many edits the backend has recorded since the last translation run
// started — mirrors bridge.py's Api._edits_since_last_run. Updated by every
// successful edit (whether made in the Review screen's table or live in the
// log) and reset to 0 the moment a new run starts (setJobState).
let editsSinceLastRun = 0;

const review = {
  rows: [],        // everything the backend returned for the current language
  filtered: [],     // rows matching the current search term
  rendered: 0,      // how many of `filtered` are currently in the DOM
  language: null,   // language the loaded rows belong to (for update_translation)
};

async function loadReviewData(language) {
  review.language = language;
  const data = await window.pywebview.api.get_review_data(language);
  review.rows = data.rows || [];
  review.filtered = review.rows;
  editsSinceLastRun = data.editsSinceLastRun || 0;
  updateRebuildButton();
  renderReviewRows(true);
}

async function openReviewScreen() {
  el("review-search").value = "";
  await loadReviewData(el("select-lang").value);
  el("review-overlay").classList.add("open");
}

async function refreshReviewScreen() {
  // loadReviewData never touches the search box itself, so just re-filter
  // against whatever's *currently* typed there once fresh data lands —
  // capturing/restoring the term around the await would clobber anything
  // the user typed while the (potentially slow, on a large modpack) fetch
  // was in flight.
  await loadReviewData(review.language || el("select-lang").value);
  filterReviewRows();
}

// Jumps into the Review screen already filtered to `term`, for the "show all
// occurrences" link on a live-edited log line. Uses the exact language the
// translation event carried, not whatever the sidebar dropdown currently
// shows -- those can legitimately differ if it's been changed mid-run.
async function openReviewScreenWithSearch(term, languageLabel) {
  await loadReviewData(languageLabel || el("select-lang").value);
  el("review-search").value = term;
  filterReviewRows();
  el("review-overlay").classList.add("open");
}

function closeReviewScreen() {
  el("review-overlay").classList.remove("open");
}

function updateRebuildButton() {
  const btn = el("btn-rebuild-pack");
  btn.disabled = editsSinceLastRun <= 0 || state.running;
  btn.textContent = editsSinceLastRun > 0
    ? `🔨 Пересобрать пак (${editsSinceLastRun} испр.)`
    : "🔨 Пересобрать пак";
}

function filterReviewRows() {
  const term = el("review-search").value.trim().toLowerCase();
  review.filtered = !term
    ? review.rows
    : review.rows.filter(
        (r) => r.source.toLowerCase().includes(term) || r.translated.toLowerCase().includes(term)
      );
  review.rendered = 0;
  renderReviewRows(true);
}

function renderReviewRows(reset) {
  const body = el("review-table-body");
  if (reset) {
    body.innerHTML = "";
    review.rendered = 0;
  }

  const nextBatch = review.filtered.slice(review.rendered, review.rendered + REVIEW_PAGE_SIZE);
  const frag = document.createDocumentFragment();
  for (const row of nextBatch) {
    frag.appendChild(createReviewRow(row));
  }
  body.appendChild(frag);
  review.rendered += nextBatch.length;

  el("review-empty").style.display = review.filtered.length === 0 ? "block" : "none";
  el("btn-review-load-more").style.display =
    review.rendered < review.filtered.length ? "block" : "none";
  el("review-count").textContent = `${review.filtered.length} строк`;
}

function createReviewRow(row) {
  const div = document.createElement("div");
  div.className = "review-row";

  const source = document.createElement("div");
  source.className = "review-col-source";
  source.textContent = row.source;

  const translated = document.createElement("div");
  translated.className = "review-col-translated";
  translated.textContent = row.translated;
  translated.title = "Нажмите, чтобы изменить перевод";
  row.language = review.language;
  translated.addEventListener("click", () =>
    startEditingTranslatable(div, translated, row, (result) => {
      editsSinceLastRun = result.editsSinceLastRun;
      updateRebuildButton();
    })
  );

  const mods = document.createElement("div");
  mods.className = "review-col-mods";
  const modList = row.mods || [];
  mods.textContent = modList.length
    ? `${modList.length} ${modsWord(modList.length)}: ${modList.slice(0, 3).join(", ")}${modList.length > 3 ? "…" : ""}`
    : "—";

  const kind = document.createElement("div");
  kind.className = "review-col-kind";
  kind.textContent = row.kind || "";

  div.append(source, translated, mods, kind);
  return div;
}

function modsWord(n) {
  const mod100 = n % 100;
  const mod10 = n % 10;
  if (mod100 >= 11 && mod100 <= 14) return "модов";
  if (mod10 === 1) return "мод";
  if (mod10 >= 2 && mod10 <= 4) return "мода";
  return "модов";
}

// Shared click-to-edit behavior for any "translatable" element -- a Review
// screen row or a live log-translation line. `containerEl` gets the
// edited/saved-pulse classes, `valueEl` is swapped for a textarea, `state_`
// just needs {source, translated, engine, language}; `onSaved(result)` is
// called with the bridge's response once the edit actually commits.
function startEditingTranslatable(containerEl, valueEl, state_, onSaved) {
  if (containerEl.querySelector("textarea")) return; // already editing
  const textarea = document.createElement("textarea");
  textarea.value = state_.translated;
  valueEl.textContent = "";
  valueEl.appendChild(textarea);
  textarea.focus();
  textarea.setSelectionRange(textarea.value.length, textarea.value.length);

  async function commit() {
    const newValue = textarea.value;
    const previousValue = state_.translated;
    valueEl.removeChild(textarea);
    valueEl.textContent = newValue;
    if (newValue === previousValue) return; // no actual change
    state_.translated = newValue;
    let result;
    try {
      result = await window.pywebview.api.update_translation(
        state_.source, newValue, state_.engine, state_.language
      );
    } catch (err) {
      result = { ok: false, error: String(err) };
    }
    if (result.ok) {
      containerEl.classList.add("edited");
      containerEl.classList.remove("saved-pulse");
      void containerEl.offsetWidth;
      containerEl.classList.add("saved-pulse");
      if (onSaved) onSaved(result);
    } else {
      // Save failed (or the bridge call itself rejected) -- roll back the
      // optimistic DOM/state update above so the displayed value matches
      // what's actually persisted, and surface the failure instead of
      // leaving it silently unreported (previously the DOM/state_ were
      // already mutated unconditionally, before this await even resolved,
      // with no error path at all -- a failed save looked identical to a
      // successful one).
      state_.translated = previousValue;
      valueEl.textContent = previousValue;
      appendLogLine(`❌ Не удалось сохранить перевод: ${result.error || "неизвестная ошибка"}`, "red");
    }
  }

  textarea.addEventListener("blur", commit);
  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      textarea.blur();
    } else if (e.key === "Escape") {
      textarea.value = state_.translated;
      textarea.blur();
    }
  });
}

// ---------------------------------------------------------------------
// Options collection for start_analysis / start_translation
// ---------------------------------------------------------------------

function collectOptions() {
  return {
    language: el("select-lang").value,
    mcVersion: el("select-mcver").value,
    outputMode: getRadioValue("output-mode"),
    packName: el("input-pack-name").value,
    translateMods: el("chk-mods").checked,
    translateBooks: el("chk-books").checked,
    translateQuests: el("chk-quests").checked,
    engine: getRadioValue("engine"),
    googleMode: getRadioValue("google-mode"),
    aiProvider: getRadioValue("ai-provider"),
    aiMode: getRadioValue("ai-mode"),
    processMode: getRadioValue("process-mode"),
  };
}

// ---------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------

function wireEvents() {
  qsa('input[name="output-mode"]').forEach((r) => r.addEventListener("change", updateOutputUi));
  qsa('input[name="engine"]').forEach((r) => r.addEventListener("change", updateEngineUi));
  qsa('input[name="ai-provider"]').forEach((r) => r.addEventListener("change", updateAiProviderUi));
  el("set-custom-preset").addEventListener("change", applyCustomPreset);

  el("btn-or-autocycle").addEventListener("click", async () => {
    const on = await window.pywebview.api.toggle_auto_cycle();
    setAutoCycleButton(on);
  });

  el("btn-select-folder").addEventListener("click", async () => {
    const result = await window.pywebview.api.select_mc_folder();
    if (!result) return;
    el("folder-label").textContent = shortenPath(result.path);
    if (result.detectedVersion) {
      el("select-mcver").value = result.detectedVersion;
      appendLogLine(`🔍 Автоматически определена версия игры: ${result.detectedVersion}`, "green");
      const verEl = el("select-mcver");
      verEl.classList.remove("pulse-success");
      void verEl.offsetWidth;
      verEl.classList.add("pulse-success");
    }
  });

  el("btn-analyze").addEventListener("click", async () => {
    await window.pywebview.api.start_analysis(collectOptions());
  });

  el("btn-start").addEventListener("click", async () => {
    const options = collectOptions();
    // "inplace" overwrites the original mod .jar files directly (breaking
    // their signatures, per the option's own label) with no separate
    // resourcepack/datapack output -- irreversible unless the user has their
    // own backup. Nothing else in the app gated this before start_translation
    // was called.
    if (options.outputMode === "inplace") {
      const proceed = await confirmDialog(
        "Перезаписать .jar файлы?",
        "Оригинальные .jar файлы модов будут изменены на месте, их цифровые подписи будут сломаны. " +
          "Это необратимо, если у вас нет резервной копии. Продолжить?"
      );
      if (!proceed) return;
    }
    const result = await window.pywebview.api.start_translation(options);
    if (result && result.error) {
      appendLogLine(`❌ ${result.error}`, "red");
    }
  });

  el("btn-pause").addEventListener("click", async () => {
    const paused = await window.pywebview.api.toggle_pause();
    el("btn-pause").textContent = paused ? "▶ ПРОДОЛЖИТЬ" : "⏸ ПАУЗА";
    el("btn-pause").classList.toggle("btn-pause-active", paused);
  });

  el("btn-stop").addEventListener("click", async () => {
    await window.pywebview.api.stop();
    el("btn-stop").disabled = true;
    el("btn-pause").disabled = true;
    setStatus("🛑 Остановка...", 1.0);
  });

  el("btn-dict").addEventListener("click", () => window.pywebview.api.open_dict_file());

  el("btn-clear-cache").addEventListener("click", openCacheClearModal);
  wireCacheClearModal();

  el("btn-open-output").addEventListener("click", () => {
    window.pywebview.api.open_output_folder(getRadioValue("output-mode"));
  });

  // Settings drawer
  el("btn-settings").addEventListener("click", openSettingsDrawer);
  el("drawer-backdrop").addEventListener("click", closeSettingsDrawer);
  el("btn-save-settings").addEventListener("click", saveSettingsDrawer);
  qsa(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      qsa(".tab-btn").forEach((b) => b.classList.remove("active"));
      qsa(".tab-content").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      el(btn.dataset.tab).classList.add("active");
    });
  });
  el("set-gpu-layers").addEventListener("input", (e) => {
    el("lbl-gpu-val").textContent = e.target.value;
  });
  el("set-google-workers").addEventListener("input", (e) => {
    el("lbl-workers-val").textContent = e.target.value;
  });
  el("btn-auto-setup-ai").addEventListener("click", autoSetupAi);
  el("btn-browse-exe").addEventListener("click", async () => {
    const path = await window.pywebview.api.select_ai_exe();
    if (path) el("set-ai-exe").value = path;
  });
  el("btn-browse-model").addEventListener("click", async () => {
    const path = await window.pywebview.api.select_ai_model();
    if (path) el("set-ai-model").value = path;
  });
  el("btn-filetypes-all").addEventListener("click", () => {
    qsa("#filetypes-list input[type=checkbox]").forEach((c) => (c.checked = true));
  });
  el("btn-filetypes-none").addEventListener("click", () => {
    qsa("#filetypes-list input[type=checkbox]").forEach((c) => (c.checked = false));
  });

  // Translation review screen (usable during an active run too — only
  // rebuilding, which starts a second run, is blocked while one is active)
  el("btn-review").addEventListener("click", openReviewScreen);
  el("btn-review-close").addEventListener("click", closeReviewScreen);
  el("btn-review-refresh").addEventListener("click", refreshReviewScreen);
  el("review-search").addEventListener("input", filterReviewRows);
  el("btn-review-load-more").addEventListener("click", () => renderReviewRows(false));
  el("btn-rebuild-pack").addEventListener("click", async () => {
    const opts = collectOptions();
    // Stash the user's actual saved process-mode choice separately, then
    // force THIS one-shot run to "force" (rebuild everything from the
    // now-corrected cache). Without persistProcessMode, bridge.py's
    // _save_session_settings would persist "force" as the user's default,
    // so the next ordinary "Начать перевод" click would silently re-run in
    // force mode too -- re-translating the whole modpack from scratch.
    opts.persistProcessMode = opts.processMode;
    opts.processMode = "force";
    closeReviewScreen();
    appendLogLine("🔨 Пересборка пака с учётом исправлений из проверки переводов...", "magenta");
    const result = await window.pywebview.api.start_translation(opts);
    if (result && result.error) {
      appendLogLine(`❌ ${result.error}`, "red");
    }
  });
}

window.addEventListener("pywebviewready", init);
// Fallback in case pywebviewready already fired before this script attached.
if (window.pywebview) init();
