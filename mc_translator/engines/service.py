import threading

from mc_translator.cache import TranslationCache
from mc_translator.config import ConfigManager
from mc_translator.constants import DEFAULT_OPENROUTER_MODEL
from mc_translator.engines.anthropic import AnthropicEngine
from mc_translator.engines.base import EngineCallbacks, EngineItem, TranslationEngine
from mc_translator.engines.custom_api import CustomApiEngine
from mc_translator.engines.deepl import DeepLEngine
from mc_translator.engines.google import GoogleEngine
from mc_translator.engines.kobold import KoboldEngine
from mc_translator.engines.openrouter import OpenRouterEngine, resolve_free_models
from mc_translator.engines.rate_limit import ModelRotator, RateLimiter
from mc_translator.text_processing import apply_smart_glue, is_only_placeholders, mask_protected_fragments
from mc_translator.usage_log import record_usage


class TranslationService:
    """Prepares strings, uses cache, delegates to a translation engine."""

    def __init__(
        self,
        engine_name: str,
        cache: TranslationCache,
        config: ConfigManager,
        *,
        google_mode: str = "single",
        ai_mode: str = "safe",
        ai_provider: str = "local",
        global_context: str = "",
        state: "JobState | None" = None,
        usage_log: dict | None = None,
        language_label: str = "",
    ) -> None:
        self.engine_name = engine_name
        self.cache = cache
        self.config = config
        self.google_mode = google_mode
        self.ai_mode = ai_mode
        self.ai_provider = ai_provider
        self.global_context = global_context
        self.state = state
        # Per-mc_dir, per-language record of which mods use which translated
        # string -- feeds the GUI's translation review screen. None (the
        # default) just disables recording, e.g. for callers that don't care.
        self.usage_log = usage_log
        # The display label (e.g. "Русский") this run was started with --
        # threaded through to callbacks.on_translation so the GUI's live edit
        # commits target the right cache entry even if the sidebar's language
        # dropdown gets changed mid-run for planning the *next* run.
        self.language_label = language_label
        # Lazily created on first OpenRouter use and reused for every
        # translate_dict() call made through this TranslationService (i.e.
        # for the service's whole lifetime -- one full translation run). See
        # _build_engine: without sharing this, a fresh RateLimiter was built
        # per file, so pacing never actually persisted across files.
        self._openrouter_rate_limiter: RateLimiter | None = None
        # Lazily created (once, guarded by _rate_limiter_lock below) the
        # first time _build_engine() is asked for an OpenRouter engine with
        # OPENROUTER/auto_cycle_free enabled. None means either auto-cycle
        # is off, or resolve_free_models() came back empty (shouldn't
        # normally happen -- it always falls back to FALLBACK_FREE_MODELS --
        # but _build_engine treats None defensively as "fall back to
        # single-model mode" either way). Shared across every per-file
        # OpenRouterEngine instance for the run, same reasoning as
        # _openrouter_rate_limiter above.
        self._openrouter_rotator: ModelRotator | None = None
        self._rotator_built = False
        # Shared RateLimiter for CustomApiEngine, same reasoning as
        # _openrouter_rate_limiter above -- no rotator here: unlike
        # OpenRouter's free models, NVIDIA NIM's rate limit is account-wide
        # rather than per-model, so CustomApiEngine just waits out a 429
        # against the single configured model instead of rotating.
        self._custom_rate_limiter: RateLimiter | None = None
        # Guards the lazy construction above -- job.py can now call
        # translate_dict() (and therefore _build_engine()) concurrently from
        # multiple file-processing threads for OpenRouter runs; without this,
        # two threads racing the "is None" check could each build their own
        # RateLimiter, briefly defeating the whole point of sharing one.
        self._rate_limiter_lock = threading.Lock()
        # Guards mutations of the shared usage_log dict (setdefault + list
        # append in usage_log.record_usage), called from translate_dict below
        # on both the cache-hit and freshly-translated paths -- also unsafe
        # once multiple files translate concurrently.
        self._usage_log_lock = threading.Lock()
        # Set once a BatchLlmEngine reports backend_seems_dead (see
        # llm_common.py) -- once the AI backend has proven unresponsive for
        # one file, stop calling it for every remaining file in this run too,
        # instead of repeating the same multi-minute timeout storm per file.
        # Untranslated strings are left as-is (not cached), so a later rerun
        # picks them back up once the backend recovers.
        self._ai_backend_given_up = False

    def _get_or_build_rotator(self) -> ModelRotator | None:
        """Resolves the auto-cycle free-model list at most once per run
        (resolve_free_models can hit the network) and builds the shared
        ModelRotator every subsequent OpenRouterEngine reuses -- same
        lazy-once-under-lock pattern as _openrouter_rate_limiter."""
        if not self._rotator_built:
            with self._rate_limiter_lock:
                if not self._rotator_built:
                    models = resolve_free_models(self.config)
                    self._openrouter_rotator = ModelRotator(models) if models else None
                    self._rotator_built = True
        return self._openrouter_rotator

    def _build_engine(self, context: str = "") -> TranslationEngine:
        if self.engine_name == "google":
            return GoogleEngine(
                workers=self.config.getint("GENERAL", "google_workers", 5),
                mode=self.google_mode,
            )
        if self.engine_name == "deepl":
            return DeepLEngine(self.config.get("API", "deepl_key"))
        final_ctx = context
        if self.global_context:
            final_ctx = f"{self.global_context} | {context}" if context else self.global_context

        if self.ai_provider == "openrouter":
            model = self.config.get("OPENROUTER", "model") or DEFAULT_OPENROUTER_MODEL
            rotator = self._get_or_build_rotator() if self.config.getboolean("OPENROUTER", "auto_cycle_free") else None
            if self._openrouter_rate_limiter is None:
                with self._rate_limiter_lock:
                    if self._openrouter_rate_limiter is None:
                        is_free_model = rotator is not None or ":free" in model.lower()
                        self._openrouter_rate_limiter = RateLimiter(min_interval=4.0 if is_free_model else 1.5)
            return OpenRouterEngine(
                api_key=self.config.get("OPENROUTER", "api_key"),
                model=model,
                mode=self.ai_mode,
                context=final_ctx,
                site_url=self.config.get("OPENROUTER", "site_url"),
                app_name=self.config.get("OPENROUTER", "app_name"),
                rate_limiter=self._openrouter_rate_limiter,
                rotator=rotator,
            )
        if self.ai_provider == "custom":
            if self._custom_rate_limiter is None:
                with self._rate_limiter_lock:
                    if self._custom_rate_limiter is None:
                        self._custom_rate_limiter = RateLimiter(min_interval=1.5)
            return CustomApiEngine(
                base_url=self.config.get("CUSTOM_AI", "base_url"),
                api_key=self.config.get("CUSTOM_AI", "api_key"),
                model=self.config.get("CUSTOM_AI", "model"),
                mode=self.ai_mode,
                context=final_ctx,
                rate_limiter=self._custom_rate_limiter,
            )
        if self.ai_provider == "anthropic":
            return AnthropicEngine(
                api_key=self.config.get("ANTHROPIC", "api_key"),
                model=self.config.get("ANTHROPIC", "model"),
                mode=self.ai_mode,
                context=final_ctx,
            )
        return KoboldEngine(mode=self.ai_mode, context=final_ctx)

    def translate_dict(
        self,
        strings: dict[str, str],
        target_lang: dict,
        callbacks: EngineCallbacks,
        *,
        context: str = "",
        usage_label: tuple[str, str] | None = None,
    ) -> dict[str, str]:
        """
        usage_label, if given, is (mod_name, kind) -- e.g. ("Applied Energistics 2",
        "Интерфейс") -- and every string this call resolves (cache hit or freshly
        translated) is recorded into self.usage_log for the GUI's translation
        review screen. Strings that pass through untouched (nothing to mask) or
        that the engine failed to translate are not recorded -- only strings that
        actually end up as a real translation in the output.

        This method is the SINGLE source of truth for self.state's
        translated_strings/cached_strings/untranslated_strings progress
        counters (see runtime/state.py). Callers (every processor) must NOT
        also increment translated_strings from the dict this method returns --
        that dict (`result`) deliberately also contains cache hits and
        English-fallback entries for callers that just want to write output,
        so counting len(result) there double-counts cache hits and silently
        counts untranslated fallbacks as if they were real translations (a
        real regression: it made a run where the AI backend gave up partway
        still report "100% translated").
        """
        if not strings:
            return {}

        smart_glue = self.config.getboolean("GENERAL", "smart_glue")
        result: dict[str, str] = {}
        pending: dict[str, EngineItem] = {}
        cached_count = 0

        for key, text in strings.items():
            if not callbacks.should_run():
                break
            callbacks.wait_if_paused()
            if smart_glue:
                text = apply_smart_glue(text)

            hit = self.cache.get(target_lang["api"], text)
            if hit is not None:
                result[key] = hit
                cached_count += 1
                if self.usage_log is not None and usage_label is not None:
                    with self._usage_log_lock:
                        record_usage(self.usage_log, text, usage_label[0], usage_label[1], self.engine_name)
                continue

            masked, mapping = mask_protected_fragments(text)
            if not masked or is_only_placeholders(masked):
                # Nothing real left to translate (either no fragments were
                # protected at all, or the whole string WAS protected
                # fragments with no prose around them, e.g. a bare "§f%s")
                # -- sending the latter to an AI engine anyway is pure
                # wasted work and a real failure risk: a small/quantized
                # model has no real sentence to anchor "leave this alone"
                # formatting around when the entire prompt IS the
                # placeholder, and is prone to "simplifying" it away
                # (live-verified: a masked "[#0#]" came back as a bare
                # "#0#" with the brackets stripped, failing the strict
                # unmask and counting as untranslated for no reason).
                result[key] = text
                continue
            pending[key] = EngineItem(key=key, original=text, masked=masked, mapping=mapping)

        if cached_count:
            callbacks.on_log(f"   🗃️ Из кэша: {cached_count} строк", "dim")
            if self.state is not None:
                self.state.increment_cached(cached_count)

        if not pending or not callbacks.should_run():
            return result

        is_ai_engine = self.engine_name not in ("google", "deepl")
        translated: dict[str, str] = {}
        if not (is_ai_engine and self._ai_backend_given_up):
            engine = self._build_engine(context)
            translated = engine.translate_batch(pending, target_lang, callbacks)
            if is_ai_engine and getattr(engine, "backend_seems_dead", False):
                self._ai_backend_given_up = True
                callbacks.on_log(
                    "⚠ ИИ-перевод отключён до конца этого запуска — сервер не отвечает. "
                    "Оставшиеся строки будут сохранены как есть (доперевод подхватит их позже).",
                    "red",
                )

        for key, text in translated.items():
            original = pending[key].original
            # mask_protected_fragments() strips the source's leading/
            # trailing whitespace before sending text to the translator, so
            # a semantically meaningful edge -- a trailing space that
            # separates a label from an in-game-appended value ("Energy: "),
            # a markdown list's leading indent -- would otherwise be lost
            # from the result. Restore it here, once, right before caching,
            # so a later cache hit returns the same padded value instead of
            # dropping it again.
            leading = original[: len(original) - len(original.lstrip())]
            trailing = original[len(original.rstrip()):]
            text = f"{leading}{text.strip()}{trailing}"
            result[key] = text
            self.cache.set(target_lang["api"], original, text)
            if self.usage_log is not None and usage_label is not None:
                with self._usage_log_lock:
                    record_usage(self.usage_log, original, usage_label[0], usage_label[1], self.engine_name)
            if callbacks.on_translation is not None:
                callbacks.on_translation(original, text, self.engine_name, self.language_label)
            else:
                callbacks.on_log(f" > {original[:40]} -> {text[:40]}", "dim")
        if self.state is not None:
            self.state.increment_translated(len(translated))

        # Anything the engine didn't return -- a genuine failure, or the
        # whole call skipped because the AI backend already gave up
        # (_ai_backend_given_up above) -- falls back to the untranslated
        # English original so the file still writes out something valid.
        # Tracked separately (not as "translated") so run_translation's
        # final summary can tell a real full success from a run that
        # silently left large portions of the pack in English.
        fallback_count = 0
        for key, item in pending.items():
            if key not in translated:
                result[key] = item.original
                fallback_count += 1
        if fallback_count and self.state is not None:
            self.state.increment_untranslated(fallback_count)

        self.cache.save_if_threshold()
        return result
