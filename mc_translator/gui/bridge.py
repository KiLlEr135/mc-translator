"""
Python <-> JS bridge for the pywebview-based GUI.

Every JS-callable method here is a 1:1 port of what the old Tkinter
TranslatorApp's button commands did; on_log/on_status/on_row push events to
the page through a batched evaluate_js queue instead of touching widgets
directly, since a real translation run can log hundreds of lines a second
and one evaluate_js call per line would visibly lag the page.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import threading
from logging.handlers import RotatingFileHandler

import requests
import webview

from mc_translator import __version__, text_processing
from mc_translator.cache import load_both_caches
from mc_translator.config import settings
from mc_translator.constants import DICT_FILE, LANGUAGES, MC_VERSIONS, MOD_CONTENT_FILE_TYPES
from mc_translator.gui.ai_setup_api import AiSetupMixin
from mc_translator.gui.cache_ops import CacheOpsMixin
from mc_translator.gui.settings_api import SettingsMixin
from mc_translator.runtime.job import TranslationJob, TranslationOptions
from mc_translator.runtime.state import JobState
from mc_translator.text_processing import is_displayable_source
from mc_translator.usage_log import load_usage_log
from mc_translator.utils.app_paths import app_path
from mc_translator.utils.version_detector import detect_mc_version

# Public repo this build reports updates against -- see README's release
# link. NOT the old upstream project; this is the user's own published fork.
UPDATE_REPO = "KiLlEr135/mc-translator"


def _is_newer_version(current: str, latest: str) -> bool:
    """Numeric (not lexicographic) comparison, so e.g. '9.9' isn't
    incorrectly "newer" than '9.10'."""
    def _parts(v: str) -> tuple[int, ...]:
        return tuple(int(p) for p in re.findall(r"\d+", v))
    return _parts(latest) > _parts(current)


def _check_latest_release(repo: str, current_version: str, timeout: float = 5) -> str | None:
    """Returns the latest GitHub release's version string if it's newer than
    current_version, else None. Never raises -- an unreachable GitHub or a
    malformed response is treated the same as "no update available" so a
    flaky connection can't affect startup."""
    try:
        resp = requests.get(f"https://api.github.com/repos/{repo}/releases/latest", timeout=timeout)
        resp.raise_for_status()
        latest = resp.json().get("tag_name", "").lstrip("vV")
    except Exception:
        return None
    return latest if latest and _is_newer_version(current_version, latest) else None


def _setup_file_logger() -> logging.Logger:
    """Persistent rotating log file, so history survives a crash or window close."""
    logger = logging.getLogger("mc_translator")
    if not logger.handlers:
        # app_path (not a bare "logs" dir) so a frozen .exe launched with a
        # working directory other than its own folder still writes logs next
        # to itself, matching settings.ini/cache.json (see constants.py).
        logs_dir = app_path("logs")
        os.makedirs(logs_dir, exist_ok=True)
        handler = RotatingFileHandler(
            os.path.join(logs_dir, "mc_translator.log"),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class _EventBatcher:
    """Coalesces frequent on_log/on_status/on_row pushes from the background
    translation thread into periodic batched evaluate_js calls instead of one
    call per event."""

    def __init__(self, get_window, interval: float = 0.08) -> None:
        self._get_window = get_window
        self._interval = interval
        self._queue: list[dict] = []
        self._lock = threading.Lock()
        self._schedule()

    def push(self, event_type: str, payload: dict) -> None:
        with self._lock:
            self._queue.append({"type": event_type, "payload": payload})

    def _schedule(self) -> None:
        timer = threading.Timer(self._interval, self._flush)
        timer.daemon = True
        timer.start()

    def _flush(self) -> None:
        with self._lock:
            batch, self._queue = self._queue, []
        window = self._get_window()
        if batch and window:
            try:
                window.evaluate_js(f"window.onBridgeBatch({json.dumps(batch)})")
            except Exception:
                pass
        self._schedule()


class Api(SettingsMixin, CacheOpsMixin, AiSetupMixin):
    def __init__(self) -> None:
        self._file_logger = _setup_file_logger()
        self.job_state = JobState()
        self.cache_std, self.cache_ai, self._polish_total = load_both_caches()
        self._job: TranslationJob | None = None
        self._window = None
        self._batcher = _EventBatcher(lambda: self._window)
        self._edits_since_last_run = 0
        # "Auto-setup local AI" state (see gui/ai_setup_api.py) -- guarded
        # the same way a translation/analysis job is (job_state.is_running),
        # and start_translation/start_analysis below are symmetrically
        # guarded against a download being active, since both mutate the
        # same AI settings a job would read.
        self._ai_download_active = False
        self._ai_download_cancelled = False
        self._pending_ai_plan = None

    def set_window(self, window) -> None:
        self._window = window

    # ------------------------------------------------------------------
    # Push helpers — these are exactly what used to be Tkinter callbacks
    # (self.log / self.set_status / self.log_row); TranslationJob calls them
    # completely unchanged.
    # ------------------------------------------------------------------

    def on_log(self, message: str, tag: str = "white") -> None:
        self._file_logger.info(message)
        self._batcher.push("log", {"message": message, "tag": tag})

    def on_status(self, text: str, progress: float | None) -> None:
        self._batcher.push("status", {"text": text, "progress": progress})

    def on_row(self, icon: str, name: str, kind: str, trans_c: int, en_c: int, pct: int) -> None:
        self._batcher.push(
            "row",
            {"icon": icon, "name": name, "kind": kind, "translated": trans_c, "total": en_c, "pct": pct},
        )

    def on_translation(self, source: str, translated: str, engine: str, language_label: str) -> None:
        """Fired once per freshly-translated string (not cache hits) so the
        log can render a live, editable line for it -- see translate_dict in
        engines/service.py. Replaces the old plain-text ' > src -> dst' line
        1:1, so this doesn't add event volume, just structure."""
        if not is_displayable_source(source):
            # Translation keys / technical IDs / bare numbers still get
            # translated and cached -- they're just not worth a GUI line.
            return
        self._batcher.push(
            "translation",
            {"source": source, "translated": translated, "engine": engine, "language": language_label},
        )

    # ------------------------------------------------------------------
    # JS -> Python: initial state / settings
    # ------------------------------------------------------------------

    def get_initial_state(self) -> dict:
        return {
            "version": __version__,
            "languages": list(LANGUAGES.keys()),
            "mcVersions": MC_VERSIONS,
            "mcDir": settings.get("GENERAL", "mc_dir"),
            "session": {
                "language": settings.get("SESSION", "language"),
                "mcVersion": settings.get("SESSION", "mc_version"),
                "outputMode": settings.get("SESSION", "output_mode"),
                "packName": settings.get("SESSION", "pack_name"),
                "translateMods": settings.getboolean("SESSION", "translate_mods"),
                "translateBooks": settings.getboolean("SESSION", "translate_books"),
                "translateQuests": settings.getboolean("SESSION", "translate_quests"),
                "engine": settings.get("SESSION", "engine"),
                "googleMode": settings.get("SESSION", "google_mode"),
                "aiMode": settings.get("SESSION", "ai_mode"),
                "processMode": settings.get("SESSION", "process_mode"),
            },
            "aiProvider": settings.get("AI", "ai_provider") or "local",
            "polishTotal": self._polish_total,
            "openrouterAutoCycle": settings.getboolean("OPENROUTER", "auto_cycle_free"),
        }

    # ------------------------------------------------------------------
    # JS -> Python: translation review screen
    # ------------------------------------------------------------------
    #
    # The review table is built from two files that already exist for a
    # translated modpack: the usage log (mc_translator/usage_log.py --
    # per-mc_dir, per-language, "this source string appears in these mods")
    # and the translation cache itself (mc_translator/cache.py -- the actual
    # current translated value, shared/deduped across every mod). Editing a
    # row writes straight into the cache; because the cache is keyed by
    # source text alone, the fix automatically applies to every mod that
    # happens to share that exact string. Applying edits to the already-built
    # resourcepack/datapack/jar is handled by clicking "Rebuild pack" in the
    # UI, which just calls start_translation again with processMode forced
    # to "force" -- every string is re-resolved from the (now-corrected)
    # cache at effectively zero cost for anything already cached.

    def get_review_data(self, language_label: str) -> dict:
        mc_dir = settings.get("GENERAL", "mc_dir")
        lang = LANGUAGES.get(language_label)
        if not mc_dir or not lang:
            return {"rows": []}
        usage_log = load_usage_log(mc_dir, lang["file"])
        rows = []
        for source, entry in usage_log.items():
            if not is_displayable_source(source):
                continue  # keys/technical IDs/numbers -- still cached, just not shown
            engine = entry.get("engine", "google")
            cache = self.cache_ai if engine == "ai" else self.cache_std
            translated = cache.get(lang["api"], source)
            if translated is None:
                continue  # cache was cleared/edited out from under the usage log
            rows.append(
                {
                    "source": source,
                    "translated": translated,
                    "mods": entry.get("mods", []),
                    "kind": entry.get("kind", ""),
                    "engine": engine,
                }
            )
        return {"rows": rows, "editsSinceLastRun": self._edits_since_last_run}

    def update_translation(self, source: str, new_value: str, engine: str, language_label: str) -> dict:
        lang = LANGUAGES.get(language_label)
        if not lang:
            return {"ok": False, "error": "no language selected"}
        cache = self.cache_ai if engine == "ai" else self.cache_std
        # Deliberately skip polish_translation here -- a manual correction is
        # the user's final word, not something the dictionary auto-fixer
        # should mutate further.
        cache.set(lang["api"], source, new_value)
        cache.save()
        self._edits_since_last_run += 1
        return {"ok": True, "editsSinceLastRun": self._edits_since_last_run}

    # ------------------------------------------------------------------
    # JS -> Python: file/folder dialogs (native OS dialogs via pywebview)
    # ------------------------------------------------------------------

    def select_mc_folder(self) -> dict | None:
        result = self._window.create_file_dialog(webview.FOLDER_DIALOG)
        if not result:
            return None
        path = result[0]
        settings.set("GENERAL", "mc_dir", path)
        detected = detect_mc_version(path)
        return {"path": path, "detectedVersion": detected if detected in MC_VERSIONS else None}

    def select_ai_exe(self) -> str | None:
        result = self._window.create_file_dialog(
            webview.OPEN_DIALOG, file_types=("Executables (*.exe)", "All files (*.*)")
        )
        return result[0] if result else None

    def select_ai_model(self) -> str | None:
        result = self._window.create_file_dialog(
            webview.OPEN_DIALOG, file_types=("GGUF Models (*.gguf)", "All files (*.*)")
        )
        return result[0] if result else None

    # ------------------------------------------------------------------
    # JS -> Python: simple actions
    # ------------------------------------------------------------------

    def open_dict_file(self) -> None:
        if not os.path.exists(DICT_FILE):
            text_processing.load_dictionary()
        if platform.system() == "Windows":
            os.startfile(DICT_FILE)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", DICT_FILE])
        else:
            subprocess.Popen(["xdg-open", DICT_FILE])

    def open_output_folder(self, output_mode: str) -> None:
        mc_dir = settings.get("GENERAL", "mc_dir")
        if not mc_dir:
            return
        path = mc_dir
        if output_mode == "resourcepack":
            rp_dir = os.path.join(mc_dir, "resourcepacks")
            if os.path.isdir(rp_dir):
                path = rp_dir
        try:
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            self.on_log(f"❌ Не удалось открыть папку: {exc}", "red")

    def toggle_pause(self) -> bool:
        self.job_state.is_paused = not self.job_state.is_paused
        if self.job_state.is_paused:
            self.on_log("⏸ Пауза", "yellow")
        else:
            self.on_log("▶ Продолжение", "green")
        return self.job_state.is_paused

    def toggle_auto_cycle(self) -> bool:
        """Flips OPENROUTER/auto_cycle_free -- picked up live by
        TranslationService the next time it builds an OpenRouter engine
        (roughly once per file), same as every other OPENROUTER setting."""
        new_value = not settings.getboolean("OPENROUTER", "auto_cycle_free")
        settings.set("OPENROUTER", "auto_cycle_free", new_value)
        self.on_log("🔁 Авто-перебор бесплатных моделей: " + ("ВКЛ" if new_value else "ВЫКЛ"), "cyan")
        return new_value

    def stop(self) -> None:
        # Snapshot once: self._job is set to None by the worker thread's
        # run() finally-block when a job finishes, so re-reading self._job
        # between the None-check and the .stop() call could see it flip to
        # None right as a run completes, raising AttributeError across the
        # JS boundary (and skipping the UI's post-stop cleanup, since the
        # awaited pywebview call would reject).
        job = self._job
        if job is not None:
            job.stop()
        else:
            self.job_state.stop()

    # ------------------------------------------------------------------
    # JS -> Python: analyze / translate
    # ------------------------------------------------------------------

    def _job_instance(self) -> TranslationJob:
        return TranslationJob(
            settings,
            self.cache_std,
            self.cache_ai,
            self.job_state,
            on_log=self.on_log,
            on_status=self.on_status,
            on_row=self.on_row,
            on_translation=self.on_translation,
        )

    def _options_from_dict(self, opts: dict) -> TranslationOptions:
        return TranslationOptions(
            mc_dir=settings.get("GENERAL", "mc_dir"),
            language_label=opts["language"],
            mc_version=opts["mcVersion"],
            output_mode=opts["outputMode"],
            pack_name=(opts.get("packName") or "").strip(),
            engine=opts["engine"],
            google_mode=opts.get("googleMode", "single"),
            ai_mode=opts.get("aiMode", "safe"),
            ai_provider=opts.get("aiProvider", "local"),
            global_context=settings.get("AI", "global_context"),
            process_mode=opts["processMode"],
            translate_mods=bool(opts.get("translateMods")),
            translate_books=bool(opts.get("translateBooks")),
            translate_quests=bool(opts.get("translateQuests")),
            file_types={key: settings.getboolean("FILETYPES", key) for key, _label in MOD_CONTENT_FILE_TYPES},
        )

    def _save_session_settings(self, opts: dict) -> None:
        settings.set("SESSION", "language", opts["language"])
        settings.set("SESSION", "mc_version", opts["mcVersion"])
        settings.set("SESSION", "output_mode", opts["outputMode"])
        settings.set("SESSION", "pack_name", (opts.get("packName") or "").strip() or "MCTranslator_Pack")
        settings.set("SESSION", "translate_mods", bool(opts.get("translateMods")))
        settings.set("SESSION", "translate_books", bool(opts.get("translateBooks")))
        settings.set("SESSION", "translate_quests", bool(opts.get("translateQuests")))
        settings.set("SESSION", "engine", opts["engine"])
        settings.set("SESSION", "google_mode", opts.get("googleMode", "single"))
        settings.set("SESSION", "ai_mode", opts.get("aiMode", "safe"))
        # persistProcessMode (set by app.js's btn-rebuild-pack handler) is
        # the user's actual saved choice -- opts.processMode itself may be a
        # one-shot "force" override for a Rebuild-pack run, which must not
        # become the new persisted default (that would silently switch a
        # later ordinary "Начать перевод" click into a full from-scratch
        # re-translation too).
        settings.set("SESSION", "process_mode", opts.get("persistProcessMode", opts["processMode"]))
        if opts["engine"] == "ai":
            settings.set("AI", "ai_provider", opts.get("aiProvider", "local"))

    def start_analysis(self, opts: dict) -> dict:
        if self._ai_download_active:
            return {"ok": False, "error": "Дождитесь завершения автонастройки локального ИИ."}
        if not self.job_state.try_start():
            return {"ok": False, "error": "already running"}
        self._save_session_settings(opts)
        self._batcher.push("clearLog", {})
        self._batcher.push("jobState", {"running": True, "kind": "analysis"})

        def run() -> None:
            self._job = self._job_instance()
            try:
                self._job.run_analysis(self._options_from_dict(opts))
            finally:
                self.job_state.is_running = False
                self._job = None
                self._batcher.push("jobState", {"running": False, "kind": "analysis"})

        threading.Thread(target=run, daemon=True).start()
        return {"ok": True}

    def start_translation(self, opts: dict) -> dict:
        if self._ai_download_active:
            return {"ok": False, "error": "Дождитесь завершения автонастройки локального ИИ."}
        if not self.job_state.try_start():
            return {"ok": False, "error": "already running"}
        if not settings.get("GENERAL", "mc_dir"):
            self.job_state.is_running = False
            return {"ok": False, "error": "Выберите папку Minecraft!"}

        text_processing.TERMINOLOGY_FIXES = text_processing.load_dictionary()

        self._save_session_settings(opts)
        self._edits_since_last_run = 0
        # clearLog must be queued BEFORE the "dictionary reloaded" log line:
        # onBridgeBatch replays queued events in order, so logging first and
        # clearing second (the old order) wiped this message out of the
        # panel in the very same batch -- it only ever survived in the
        # rotating file log.
        self._batcher.push("clearLog", {})
        self.on_log("📖 Словарь перезагружен.", "dim")
        self._batcher.push("jobState", {"running": True, "kind": "translation"})

        def run() -> None:
            self._job = self._job_instance()
            try:
                self._job.run_translation(self._options_from_dict(opts))
            finally:
                self.job_state.is_running = False
                self._job = None
                self._batcher.push("jobState", {"running": False, "kind": "translation"})

        threading.Thread(target=run, daemon=True).start()
        return {"ok": True}

    def check_updates(self) -> None:
        """Checks GitHub's latest release in a background thread (daemon, so
        it never blocks/delays shutdown) and logs a notice if a newer build
        exists. Silent no-op on any failure -- see _check_latest_release."""
        def run() -> None:
            latest = _check_latest_release(UPDATE_REPO, __version__)
            if latest:
                self.on_log(f"🔔 Доступна новая версия: v{latest}!", "cyan")
                self.on_log(f"   Скачать: https://github.com/{UPDATE_REPO}/releases/latest", "dim")

        threading.Thread(target=run, daemon=True).start()

    def notify_ready(self) -> None:
        """Called once by JS after the page has finished its initial render,
        so the polish log line appears after the log panel exists."""
        if self._polish_total:
            self.on_log(f"✨ Кэш отполирован: исправлено {self._polish_total} строк.", "magenta")
        self.check_updates()
