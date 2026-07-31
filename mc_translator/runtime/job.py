import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from mc_translator.cache import TranslationCache
from mc_translator.config import ConfigManager
from mc_translator.constants import LANGUAGES
from mc_translator.engines.base import EngineCallbacks
from mc_translator.engines.service import TranslationService
from mc_translator.output.pack_writer import PackWriter
from mc_translator.processors.analyzer import ModpackAnalyzer
from mc_translator.processors.archive import ArchiveProcessor
from mc_translator.processors.bat import BatProcessor
from mc_translator.processors.cfg import CfgProcessor
from mc_translator.processors.estimator import StringEstimator
from mc_translator.processors.generic_json import GenericJsonProcessor
from mc_translator.processors.jar import JarProcessor
from mc_translator.processors.lang import LangProcessor
from mc_translator.processors.loose_json import LooseJsonProcessor
from mc_translator.processors.mcfunction import McfunctionProcessor
from mc_translator.processors.nbt import NbtProcessor
from mc_translator.processors.quest_lang import QuestLangProcessor
from mc_translator.processors.snbt import SnbtProcessor
from mc_translator.processors.text import TextProcessor
from mc_translator.processors.xml import XmlProcessor
from mc_translator.processors.yaml_toml import YamlTomlProcessor
from mc_translator.runtime.ai_launcher import AiLauncher
from mc_translator.runtime.cost_estimate import estimate_openrouter_cost, estimate_translation_chars, format_count
from mc_translator.runtime.manifest import jar_fingerprint, save_estimate_cache, save_jar_manifest
from mc_translator.runtime.state import JobState
from mc_translator.runtime.translation_pipeline import build_buckets, discover_translation_files
from mc_translator.usage_log import load_usage_log, save_usage_log


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)} сек"
    if seconds < 3600:
        return f"{int(seconds // 60)} мин {int(seconds % 60)} сек"
    return f"{int(seconds // 3600)} ч {int((seconds % 3600) // 60)} мин"


@dataclass
class TranslationOptions:
    mc_dir: str
    language_label: str
    mc_version: str
    output_mode: str  # resourcepack | inplace
    pack_name: str
    engine: str  # google | deepl | ai
    google_mode: str
    ai_mode: str
    ai_provider: str  # local | openrouter | custom | anthropic
    global_context: str
    process_mode: str  # append | skip | force
    translate_mods: bool
    translate_books: bool
    translate_quests: bool
    file_types: dict | None = None  # key -> bool, from constants.MOD_CONTENT_FILE_TYPES; None/missing key = enabled


class TranslationJob:
    def __init__(
        self,
        config: ConfigManager,
        cache_std: TranslationCache,
        cache_ai: TranslationCache,
        state: JobState,
        *,
        on_log,
        on_status,
        on_row,
        on_translation=None,
    ) -> None:
        self.config = config
        self.cache_std = cache_std
        self.cache_ai = cache_ai
        self.state = state
        self.on_log = on_log
        self.on_status = on_status
        self.on_row = on_row
        self.on_translation = on_translation
        self.ai_launcher = AiLauncher(config)

    def _callbacks(self) -> EngineCallbacks:
        return EngineCallbacks(
            should_run=self.state.should_run,
            wait_if_paused=self.state.wait_if_paused,
            on_log=self.on_log,
            on_status=lambda msg: self.on_status(msg, None),
            on_translation=self.on_translation,
        )

    # Small, conservative cap on concurrent files: each file's own
    # translate_dict() call may ALSO spin up its own internal concurrency
    # (OpenRouter's BatchLlmEngine uses max_concurrent=2-4). 3 files x 2-4
    # internal workers = 6-12 threads, all funneling through OpenRouterEngine's
    # single shared RateLimiter (engines/service.py), which paces the actual
    # outgoing request rate regardless of how many threads are queued behind
    # it -- more threads here just means more waiting, not more real
    # concurrent requests. This constant is intentionally NOT used for
    # Google/DeepL (see run_translation's use_parallel gate).
    _FILE_WORKER_LIMIT = 3

    def _process_bucket(
        self,
        items: list,
        process_one,
        *,
        label: str,
        done_start: int,
        total_items: int,
        parallel: bool,
        on_item_done=None,
    ) -> tuple[int, int]:
        """Run process_one(path) for every item in `items`, either
        sequentially (today's behavior) or -- when `parallel` is True and
        there's more than one item -- across a small thread pool. Returns
        (new_done_count, failed_count).

        process_one(path) always runs OUTSIDE any lock here; only the
        done-counter/on_status update and `on_item_done` (e.g. the jar
        loop's post-success manifest bookkeeping) run under this bucket's
        own lock. Never acquire another lock while holding it -- PackWriter
        and TranslationService already have their own internal locks for
        the actual concurrent work, and nesting would risk deadlock.

        A per-item exception is caught here and logged rather than
        propagated, so one bad file no longer aborts every remaining bucket
        in the run (previously: any uncaught exception jumped straight to
        run_translation's outer except-Exception, skipping everything after
        it -- a single corrupt jar out of 300 killed the whole translation).
        """
        if not items:
            return done_start, 0

        done = done_start
        failed = 0
        lock = threading.Lock()

        def run_one(path) -> None:
            nonlocal done, failed
            if not self.state.should_run():
                return
            self.state.wait_if_paused()
            if not self.state.should_run():
                return
            ok = True
            try:
                process_one(path)
            except Exception:
                ok = False
                self.on_log(
                    f"❌ Ошибка обработки {os.path.basename(path)}:\n{traceback.format_exc()}",
                    "red",
                )
            with lock:
                # Only run on_item_done if processing succeeded AND we
                # weren't interrupted mid-item (matches the original jar
                # loop's "only mark as fully done if we weren't interrupted"
                # check -- a processor can partially process an item and
                # return normally if Stop was clicked mid-way, since
                # processors have their own internal should_run() checks).
                if ok and self.state.should_run() and on_item_done:
                    on_item_done(path)
                if not ok:
                    failed += 1
                done += 1
                self.on_status(
                    f"{label}: {done}/{total_items} | ETA: {self.state.eta_text()}",
                    done / max(total_items, 1),
                )

        if not parallel or len(items) <= 1:
            for path in items:
                if not self.state.should_run():
                    break
                run_one(path)
            return done, failed

        max_workers = min(self._FILE_WORKER_LIMIT, len(items))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(run_one, path) for path in items if self.state.should_run()]
            for fut in as_completed(futures):
                fut.result()  # run_one already catches its own errors; this just surfaces any pool-internal bug instead of swallowing it silently

        return done, failed

    def run_analysis(self, options: TranslationOptions) -> None:
        lang = LANGUAGES[options.language_label]
        analyzer = ModpackAnalyzer(self.state)
        self.on_log(f"🚀 Сканирование сборки ({lang['name']})...\n", "yellow")
        self.on_log(f"{'ФАЙЛ / МОД':<37}{'ТИП':<15}{'СТРОКИ':<12}ПРОГРЕСС", "white")
        self.on_log("-" * 75, "dim")

        total_en, total_tr = analyzer.analyze(
            options.mc_dir,
            target_lang=lang,
            translate_mods=options.translate_mods,
            translate_books=options.translate_books,
            translate_quests=options.translate_quests,
            file_types=options.file_types,
            pack_name=options.pack_name,
            on_row=self.on_row,
            on_log=self.on_log,
            on_status=lambda text, val: self.on_status(text, val),
        )

        self.on_log("-" * 75, "dim")
        if not self.state.should_run():
            self.on_log("🛑 АНАЛИЗ ПРЕРВАН", "red")
        elif total_en > 0:
            pct = int(total_tr / total_en * 100)
            color = "green" if pct >= 90 else ("yellow" if pct >= 50 else "red")
            self.on_log(f"✅ АНАЛИЗ ЗАВЕРШЕН! Готовность: {pct}% | Строк: {total_en}", color)
        else:
            self.on_log("❌ Нечего переводить!", "red")
        self.on_status("Готово", 1.0)

    def _build_processors(self, service: TranslationService, callbacks: EngineCallbacks) -> dict:
        """One TranslationJob-owned instance per file-type processor, keyed
        by the short name build_buckets (runtime/translation_pipeline.py)
        expects."""
        return {
            "jar": JarProcessor(service, self.state, callbacks),
            "loose": LooseJsonProcessor(service, self.state, callbacks),
            "snbt": SnbtProcessor(service, self.state, callbacks),
            "quest_lang": QuestLangProcessor(service, self.state, callbacks),
            "generic_json": GenericJsonProcessor(service, self.state, callbacks),
            "lang": LangProcessor(service, self.state, callbacks),
            "nbt": NbtProcessor(service, self.state, callbacks),
            "yaml_toml": YamlTomlProcessor(service, self.state, callbacks),
            "text": TextProcessor(service, self.state, callbacks),
            "archive": ArchiveProcessor(service, self.state, callbacks),
            "cfg": CfgProcessor(service, self.state, callbacks),
            "mcfunction": McfunctionProcessor(service, self.state, callbacks),
            "xml": XmlProcessor(service, self.state, callbacks),
            "bat": BatProcessor(service, self.state, callbacks),
        }

    def run_translation(self, options: TranslationOptions) -> None:
        lang = LANGUAGES[options.language_label]
        cache = self.cache_ai if options.engine == "ai" else self.cache_std

        if options.engine == "deepl" and not self.config.get("API", "deepl_key").strip():
            self.on_log("❌ Введите ключ DeepL в настройках!", "red")
            return
        if options.engine == "ai":
            if options.ai_provider == "openrouter":
                if not self.config.get("OPENROUTER", "api_key").strip():
                    self.on_log("❌ Укажите API-ключ OpenRouter в настройках!", "red")
                    return
                # Auto-cycle mode resolves its own model list (user-supplied
                # or auto-fetched free models) at run start, so a blank
                # OPENROUTER/model is only an error when it's off -- see
                # engines/service.py's _get_or_build_rotator.
                auto_cycle = self.config.getboolean("OPENROUTER", "auto_cycle_free")
                if not auto_cycle and not self.config.get("OPENROUTER", "model").strip():
                    self.on_log("❌ Укажите ID модели OpenRouter в настройках!", "red")
                    return
            elif options.ai_provider == "custom":
                if not self.config.get("CUSTOM_AI", "base_url").strip():
                    self.on_log("❌ Укажите URL эндпоинта в настройках «Другой ИИ»!", "red")
                    return
                if not self.config.get("CUSTOM_AI", "model").strip():
                    self.on_log("❌ Укажите ID модели в настройках «Другой ИИ»!", "red")
                    return
            elif options.ai_provider == "anthropic":
                if not self.config.get("ANTHROPIC", "api_key").strip():
                    self.on_log("❌ Укажите API-ключ Claude (Anthropic) в настройках!", "red")
                    return
                if not self.config.get("ANTHROPIC", "model").strip():
                    self.on_log("❌ Укажите ID модели Claude в настройках!", "red")
                    return
            elif not self.config.get("AI", "model_path").strip():
                self.on_log("❌ Выберите модель .gguf в настройках!", "red")
                return

        self.on_log("🔍 Сканирование файлов сборки...", "dim")
        found = discover_translation_files(options, lang, self.on_log)
        # Per-mc_dir, per-language record of which mods use which translated
        # string -- feeds the GUI's translation review screen. Loaded (not
        # cleared) regardless of process_mode, so an incremental append/skip
        # run keeps entries for files it doesn't touch this time, while a
        # force run naturally refreshes everything it reprocesses.
        usage_log = load_usage_log(options.mc_dir, lang["file"])

        if found.is_empty():
            if found.skipped_unchanged:
                self.on_log("✅ Все моды уже переведены (без изменений с прошлого раза).", "green")
            else:
                self.on_log("❌ Нечего переводить!", "red")
            return

        self.on_log("📊 Подсчёт строк...", "yellow")
        estimator = StringEstimator(self.state)
        self.state.total_strings = estimator.estimate(
            found.jars,
            found.loose,
            found.snbt,
            target_lang=lang,
            mode=options.process_mode,
            translate_mods=options.translate_mods,
            translate_books=options.translate_books,
            translate_quests=options.translate_quests,
            smart_glue=self.config.getboolean("GENERAL", "smart_glue"),
            generic_json_files=found.generic_json,
            lang_files=found.lang_files,
            nbt_files=found.nbt_files,
            yaml_toml_files=found.yaml_toml_files,
            text_files=found.text_files,
            archives=found.archives,
            cfg_files=found.cfg_files,
            mcfunction_files=found.mcfunction_files,
            xml_files=found.xml_files,
            bat_files=found.bat_files,
            estimate_cache=found.estimate_cache,
        )
        self.on_log(f"   Найдено: {self.state.total_strings}", "cyan")

        # Pre-flight volume/cost estimate for engines that can incur real
        # cost, shown before any API calls happen so Stop is still cheap.
        total_chars = estimate_translation_chars(self.state.total_strings)
        if options.engine == "deepl":
            self.on_log(
                f"   💰 Примерный объём: ~{format_count(total_chars)} символов "
                f"(DeepL тарифицирует по символам — сверьте с лимитом вашего плана; "
                f"грубая оценка по среднему числу символов на строку).",
                "yellow",
            )
        elif options.engine == "ai" and options.ai_provider == "openrouter":
            # Auto-cycle mode only ever rotates through free (":free") models
            # -- no paid model can enter its list (see resolve_free_models) --
            # so a cost estimate would always be $0 and isn't worth showing.
            if not self.config.getboolean("OPENROUTER", "auto_cycle_free"):
                model = self.config.get("OPENROUTER", "model")
                if ":free" not in model.lower():
                    cost = estimate_openrouter_cost(model, total_chars)
                    if cost:
                        self.on_log(
                            f"   💰 Примерная стоимость ({model}): ≈{cost} (грубая оценка по среднему числу символов на строку)",
                            "yellow",
                        )
                    else:
                        self.on_log(
                            f"   💰 Примерный объём: ~{format_count(total_chars)} символов "
                            f"(не удалось получить актуальные цены модели {model})",
                            "yellow",
                        )

        if options.engine == "ai" and options.ai_provider == "local":
            if not self.state.should_run():
                return
            if not self.ai_launcher.ensure_running(
                self.state.should_run,
                lambda msg: self.on_status(msg, None),
                self.on_log,
            ):
                return
        elif options.engine == "ai" and options.ai_provider == "openrouter":
            if self.config.getboolean("OPENROUTER", "auto_cycle_free"):
                self.on_log("🌐 OpenRouter: 🔁 авто-перебор бесплатных моделей", "cyan")
            else:
                model = self.config.get("OPENROUTER", "model")
                self.on_log(f"🌐 OpenRouter: {model}", "cyan")
        elif options.engine == "ai" and options.ai_provider == "custom":
            model = self.config.get("CUSTOM_AI", "model")
            base_url = self.config.get("CUSTOM_AI", "base_url")
            self.on_log(f"🌐 Другой ИИ: {model} ({base_url})", "cyan")
        elif options.engine == "ai" and options.ai_provider == "anthropic":
            model = self.config.get("ANTHROPIC", "model")
            self.on_log(f"🌐 Claude (Anthropic): {model}", "cyan")

        pack_writer: PackWriter | None = None
        if options.output_mode == "resourcepack":
            pack_writer = PackWriter(
                options.mc_dir,
                options.pack_name,
                options.mc_version,
                lang["name"],
            )
            self.on_log(f"📦 Ресурспак: {pack_writer.rp_zip_path}", "cyan")
            self.on_log(f"📂 Датапак: {pack_writer.dp_zip_path}", "magenta")

        service = TranslationService(
            options.engine,
            cache,
            self.config,
            google_mode=options.google_mode,
            ai_mode=options.ai_mode,
            ai_provider=options.ai_provider,
            global_context=options.global_context,
            state=self.state,
            usage_log=usage_log,
            language_label=options.language_label,
        )
        callbacks = self._callbacks()
        processors = self._build_processors(service, callbacks)

        self.state.start_time = time.time()
        self.state.translated_strings = 0
        self.state.cached_strings = 0
        self.state.untranslated_strings = 0
        self.on_log(f"🚀 ЗАПУСК ПЕРЕВОДА ({lang['name']})...\n", "yellow")

        total_items = found.total_items()
        done = 0
        total_failed = 0

        # File-level concurrency: only OpenRouter has a shared, safe-by-design
        # rate limiter (engines/service.py's _openrouter_rate_limiter) that
        # keeps the true outgoing request rate constant no matter how many
        # files are in flight. Google has no cross-file rate limit -- stacking
        # file-level concurrency on top of its own per-file worker pool risks
        # 429 storms / soft IP blocks. DeepL doesn't retry 429s at all, so a
        # concurrency-induced rate-limit hit would silently drop those strings
        # back to English for this run, on a paid-by-volume API. Local Kobold
        # is sequential by design (one GPU -- concurrent requests just queue
        # against it with no throughput gain).
        use_parallel = options.engine == "ai" and options.ai_provider == "openrouter"

        new_jar_manifest = dict(found.jar_manifest)
        try:
            def _on_jar_done(path: str) -> None:
                fp = jar_fingerprint(
                    path,
                    translate_mods=options.translate_mods,
                    translate_books=options.translate_books,
                    output_mode=options.output_mode,
                )
                if fp is not None:
                    new_jar_manifest[path] = fp

            buckets = build_buckets(found, lang, options, pack_writer, processors, _on_jar_done)
            for items, process_one, label, on_item_done in buckets:
                done, failed = self._process_bucket(
                    items, process_one, label=label, done_start=done, total_items=total_items,
                    parallel=use_parallel, on_item_done=on_item_done,
                )
                total_failed += failed

        except Exception:
            self.on_log(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА:\n{traceback.format_exc()}", "red")
        finally:
            # Always persist whatever was translated so far -- even on an
            # unexpected exception from a processor/engine -- so a crash
            # mid-run never silently discards already-paid-for translations
            # (cache.save_if_threshold() only flushes every 500 entries
            # during the run itself). Individually guarded against OSError
            # (disk full, AV/OneDrive lock, permission denied) so a failure
            # here doesn't itself skip the rest of cleanup below (jar/estimate
            # manifest persistence, pack_writer.close(), and the post-finally
            # status/summary logging) -- the sibling save_usage_log/
            # save_jar_manifest/save_estimate_cache calls already protect
            # themselves the same way internally.
            try:
                cache.save()
            except OSError as exc:
                self.on_log(f"❌ Не удалось сохранить кэш переводов: {exc}", "red")
            save_usage_log(options.mc_dir, lang["file"], usage_log)
            if found.all_jars:
                # Prune entries for jars no longer present in the modpack.
                # Saved regardless of mode: a "force" run also leaves behind
                # fresh fingerprints so a later append/skip run benefits right away.
                current_paths = set(found.all_jars)
                new_jar_manifest = {p: fp for p, fp in new_jar_manifest.items() if p in current_paths}
                save_jar_manifest(options.mc_dir, lang["file"], new_jar_manifest)
                # estimate_cache was mutated in place by estimator.estimate()
                # above (fresh/updated entries for every jar it had to
                # actually re-parse) -- prune stale entries the same way and
                # persist so the NEXT run's "Подсчёт строк" pass can skip
                # re-parsing every unchanged jar.
                pruned_estimate_cache = {p: fp for p, fp in found.estimate_cache.items() if p in current_paths}
                save_estimate_cache(options.mc_dir, lang["file"], pruned_estimate_cache)
            if pack_writer:
                try:
                    pack_writer.close()
                except OSError as exc:
                    self.on_log(f"❌ Ошибка при закрытии архива пака: {exc}", "red")
            if options.engine == "ai":
                pass  # keep AI running for user

        if not self.state.should_run():
            self.on_log("\n🛑 ОСТАНОВЛЕНО.", "red")
            self.on_status("Остановлено", 1.0)
        else:
            if total_failed > 0:
                self.on_log(
                    f"\n⚠ ПЕРЕВОД ЗАВЕРШЁН С ОШИБКАМИ: {total_failed} файл(ов) пропущено "
                    f"(см. сообщения ❌ выше).",
                    "yellow",
                )
            elif self.state.untranslated_strings > 0:
                # translate_dict() (engines/service.py) falls back to the
                # original English text -- without raising -- whenever the
                # engine couldn't translate a string (e.g. the AI backend
                # gave up after repeated failures: llm_common.py's
                # CONSECUTIVE_FAILURE_LIMIT circuit breaker). That path never
                # touches total_failed above, so a run that silently left
                # large portions of the pack in English used to still print
                # "УСПЕШНО ЗАВЕРШЕН!" here -- this branch makes that honest.
                self.on_log(
                    f"\n⚠ ПЕРЕВОД ЗАВЕРШЁН ЧАСТИЧНО: {self.state.untranslated_strings} строк "
                    f"остались непереведёнными (движок перевода не смог их обработать — не "
                    f"отвечал или возвращал повреждённые ответы). Запустите перевод ещё раз "
                    f"(«Доперевод») — уже переведённое повторно переводиться не будет.",
                    "yellow",
                )
            else:
                self.on_log("\n✅ ПЕРЕВОД УСПЕШНО ЗАВЕРШЕН!", "green")
            if options.output_mode == "resourcepack":
                self.on_log("💡 Включите ресурспак и датапак в игре.", "yellow")
            self.on_status("Все задачи выполнены!", 1.0)

        elapsed = time.time() - self.state.start_time if self.state.start_time else 0.0
        self.on_log(
            f"📋 Итог: переведено {self.state.translated_strings} строк, "
            f"из кэша — {self.state.cached_strings}, "
            f"не переведено — {self.state.untranslated_strings}, "
            f"время — {_format_duration(elapsed)}.",
            "cyan",
        )

    def stop(self) -> None:
        self.state.stop()
        self.ai_launcher.terminate(on_log=self.on_log)
