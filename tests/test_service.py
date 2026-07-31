"""Tests for mc_translator.engines.service.TranslationService -- the layer
every processor calls to mask, cache, and dispatch translation requests.
Network is stubbed via GoogleEngine._request (no real HTTP)."""
from mc_translator.cache import TranslationCache
from mc_translator.engines.anthropic import AnthropicEngine
from mc_translator.engines.base import EngineCallbacks
from mc_translator.engines.custom_api import CustomApiEngine
from mc_translator.engines.google import GoogleEngine
from mc_translator.engines.kobold import KoboldEngine
from mc_translator.engines.service import TranslationService
from mc_translator.runtime.state import JobState


class FakeConfig:
    def getboolean(self, section, key, fallback=False):
        return fallback

    def getint(self, section, key, fallback=0):
        return fallback

    def get(self, section, key):
        return ""


class CustomAiConfig(FakeConfig):
    """FakeConfig always returns "" from get() -- fine for engines that don't
    read config (google/kobold), but CustomApiEngine's base_url/api_key/model
    all come from the CUSTOM_AI section, so a real config stub is needed to
    verify _build_engine wires them through correctly."""

    def get(self, section, key):
        if section == "CUSTOM_AI":
            return {"base_url": "https://api.example.com/v1/chat/completions", "api_key": "sk-test", "model": "gpt-4o-mini"}[key]
        return ""


class AnthropicConfig(FakeConfig):
    """Same reasoning as CustomAiConfig above, for the ANTHROPIC section."""

    def get(self, section, key):
        if section == "ANTHROPIC":
            return {"api_key": "sk-ant-test", "model": "claude-sonnet-4-5-20250929"}[key]
        return ""


def make_callbacks() -> EngineCallbacks:
    return EngineCallbacks(
        should_run=lambda: True,
        wait_if_paused=lambda: None,
        on_log=lambda msg, tag="white": None,
        on_status=lambda msg: None,
    )


LANG = {"file": "ru_ru", "api": "ru", "deepl": "RU", "name": "Russian", "regex": r"[А-Яа-яЁё]"}


def test_cache_hit_returns_cached_value_without_calling_engine(tmp_path, monkeypatch):
    called = []

    def fake_request(self, text, api_code, timeout=10):
        called.append(text)
        return text

    monkeypatch.setattr(GoogleEngine, "_request", fake_request)
    cache = TranslationCache(str(tmp_path / "cache.json"))
    cache.set("ru", "Hello", "Привет")
    service = TranslationService("google", cache, FakeConfig())

    result = service.translate_dict({"k": "Hello"}, LANG, make_callbacks())

    assert result == {"k": "Привет"}
    assert called == []


def test_fresh_translation_is_cached_and_reused_on_next_call(tmp_path, monkeypatch):
    calls = []

    def fake_request(self, text, api_code, timeout=10):
        calls.append(text)
        return text.upper()

    monkeypatch.setattr(GoogleEngine, "_request", fake_request)
    cache = TranslationCache(str(tmp_path / "cache.json"))
    service = TranslationService("google", cache, FakeConfig())

    first = service.translate_dict({"k": "hello"}, LANG, make_callbacks())
    second = service.translate_dict({"k": "hello"}, LANG, make_callbacks())

    assert first == {"k": "HELLO"}
    assert second == {"k": "HELLO"}
    assert len(calls) == 1  # second call was a cache hit, no new request


def test_engine_failure_falls_back_to_original_without_caching(tmp_path, monkeypatch):
    """A key the engine omits (translation failure) must fall back to the
    English original for this run, and must NOT be cached -- so the next
    run retries it instead of permanently caching a failure."""
    monkeypatch.setattr(GoogleEngine, "_request", lambda self, text, api_code, timeout=10: None)
    cache = TranslationCache(str(tmp_path / "cache.json"))
    service = TranslationService("google", cache, FakeConfig())

    result = service.translate_dict({"k": "hello"}, LANG, make_callbacks())

    assert result == {"k": "hello"}
    assert cache.get("ru", "hello") is None


def test_fresh_translation_preserves_source_trailing_whitespace(tmp_path, monkeypatch):
    """Regression test (H2): mask_protected_fragments strips leading/
    trailing whitespace before sending text to the translator (the engine
    shouldn't have to deal with odd padding), but that whitespace can be
    semantically meaningful -- e.g. "Energy: " with the trailing space is
    what visually separates the label from an in-game-appended value
    ("Energy:5" if lost vs "Energy: 5" if kept). It must survive into both
    the returned result and what gets cached, so a later cache hit doesn't
    silently drop it again."""
    monkeypatch.setattr(GoogleEngine, "_request", lambda self, text, api_code, timeout=10: text.upper())
    cache = TranslationCache(str(tmp_path / "cache.json"))
    service = TranslationService("google", cache, FakeConfig())

    result = service.translate_dict({"k": "Energy: "}, LANG, make_callbacks())

    assert result["k"] == "ENERGY: "
    assert cache.get("ru", "Energy: ") == "ENERGY: "


def test_placeholder_only_string_skips_engine_call_entirely(tmp_path, monkeypatch):
    """Regression test: "§f%s" masks down to a bare "[#0#]" with no real
    prose (is_only_placeholders). Sending that to a small/quantized model
    risks it "simplifying away" the brackets (live-verified), so
    translate_dict must fall back to the original text without ever
    calling the engine -- same treatment as an empty mask."""
    called = []
    monkeypatch.setattr(GoogleEngine, "_request", lambda self, text, api_code, timeout=10: called.append(text) or text.upper())
    cache = TranslationCache(str(tmp_path / "cache.json"))
    service = TranslationService("google", cache, FakeConfig())

    result = service.translate_dict({"k": "§f%s"}, LANG, make_callbacks())

    assert result == {"k": "§f%s"}
    assert called == []
    assert cache.get("ru", "§f%s") is None


def test_ai_backend_given_up_stops_calling_engine_for_rest_of_run(tmp_path, monkeypatch):
    """Regression test: a real run kept retrying a hung KoboldCPP server on
    every remaining mod file (each attempt taking minutes) with no limit.
    Once one file's engine reports backend_seems_dead (llm_common's
    per-file circuit breaker), TranslationService must stop calling the
    engine for every later translate_dict() call in the same run."""
    call_count = {"n": 0}

    def fake_request(self, prompt, max_tokens):
        call_count["n"] += 1
        return None  # every request fails -- simulates a dead backend

    monkeypatch.setattr(KoboldEngine, "_request", fake_request)
    cache = TranslationCache(str(tmp_path / "cache.json"))
    service = TranslationService("ai", cache, FakeConfig(), ai_provider="local")

    # 60 items = 3 chunks of 20 ("safe" mode) -- enough to trip llm_common's
    # CONSECUTIVE_FAILURE_LIMIT within this single translate_dict() call.
    first_batch = {f"k{i}": f"v{i}" for i in range(60)}
    first = service.translate_dict(first_batch, LANG, make_callbacks())

    assert first == first_batch  # every key fell back to its original text
    assert service._ai_backend_given_up is True
    calls_after_first_file = call_count["n"]

    second = service.translate_dict({"other": "hello"}, LANG, make_callbacks())

    assert second == {"other": "hello"}  # falls back without ever calling the engine
    assert call_count["n"] == calls_after_first_file


def test_fresh_translation_preserves_source_leading_indent(tmp_path, monkeypatch):
    """Regression test (H2): a markdown/text line's leading indent (e.g. a
    nested list item) must survive translation instead of being flattened
    to top-level by mask_protected_fragments' internal strip()."""
    monkeypatch.setattr(GoogleEngine, "_request", lambda self, text, api_code, timeout=10: text.upper())
    cache = TranslationCache(str(tmp_path / "cache.json"))
    service = TranslationService("google", cache, FakeConfig())

    result = service.translate_dict({"k": "  - nested item"}, LANG, make_callbacks())

    assert result["k"] == "  - NESTED ITEM"


# ---------------------------------------------------------------------
# Progress counters (state.translated_strings/cached_strings/untranslated_strings)
#
# Regression coverage for the bug where every processor counted
# len(translate_dict()'s full return dict) as "translated" -- that dict
# also contains cache hits and English-fallback entries, so a run where the
# AI backend gave up partway through still reported translated_strings as
# if everything had succeeded. translate_dict() itself is now the single
# place these three counters get incremented (see engines/service.py).
# ---------------------------------------------------------------------


def test_cache_hit_increments_cached_strings_only(tmp_path, monkeypatch):
    monkeypatch.setattr(GoogleEngine, "_request", lambda self, text, api_code, timeout=10: text.upper())
    cache = TranslationCache(str(tmp_path / "cache.json"))
    cache.set("ru", "Hello", "Привет")
    state = JobState()
    service = TranslationService("google", cache, FakeConfig(), state=state)

    service.translate_dict({"k": "Hello"}, LANG, make_callbacks())

    assert state.cached_strings == 1
    assert state.translated_strings == 0
    assert state.untranslated_strings == 0


def test_fresh_translation_increments_translated_strings_only(tmp_path, monkeypatch):
    monkeypatch.setattr(GoogleEngine, "_request", lambda self, text, api_code, timeout=10: text.upper())
    cache = TranslationCache(str(tmp_path / "cache.json"))
    state = JobState()
    service = TranslationService("google", cache, FakeConfig(), state=state)

    service.translate_dict({"k": "hello"}, LANG, make_callbacks())

    assert state.translated_strings == 1
    assert state.cached_strings == 0
    assert state.untranslated_strings == 0


def test_engine_failure_increments_untranslated_strings_not_translated(tmp_path, monkeypatch):
    """The core bug this fix targets: a string the engine couldn't translate
    falls back to the original English text (translate_dict's documented
    contract), but that must NOT be counted as a successful translation."""
    monkeypatch.setattr(GoogleEngine, "_request", lambda self, text, api_code, timeout=10: None)
    cache = TranslationCache(str(tmp_path / "cache.json"))
    state = JobState()
    service = TranslationService("google", cache, FakeConfig(), state=state)

    result = service.translate_dict({"k": "hello"}, LANG, make_callbacks())

    assert result == {"k": "hello"}
    assert state.untranslated_strings == 1
    assert state.translated_strings == 0
    assert state.cached_strings == 0


def test_mixed_cache_fresh_and_failure_counters_dont_double_count(tmp_path, monkeypatch):
    """Regression test: a cache hit used to be counted in BOTH
    cached_strings (correctly) and translated_strings (via the processor's
    len(full result dict)) -- this asserts each of the three counters
    reflects exactly its own category, with no overlap."""

    def fake_request(self, text, api_code, timeout=10):
        if text == "fail":
            return None
        return text.upper()

    monkeypatch.setattr(GoogleEngine, "_request", fake_request)
    cache = TranslationCache(str(tmp_path / "cache.json"))
    cache.set("ru", "cached", "КЭШ")
    state = JobState()
    service = TranslationService("google", cache, FakeConfig(), state=state)

    result = service.translate_dict({"a": "cached", "b": "fresh", "c": "fail"}, LANG, make_callbacks())

    assert result == {"a": "КЭШ", "b": "FRESH", "c": "fail"}
    assert state.cached_strings == 1
    assert state.translated_strings == 1
    assert state.untranslated_strings == 1


def test_build_engine_wires_custom_provider_from_config(tmp_path):
    """ai_provider='custom' must build a CustomApiEngine using the
    CUSTOM_AI/base_url, api_key, and model config values -- not fall through
    to KoboldEngine (the ai_provider != 'openrouter' default branch)."""
    cache = TranslationCache(str(tmp_path / "cache.json"))
    service = TranslationService("ai", cache, CustomAiConfig(), ai_provider="custom")

    engine = service._build_engine()

    assert isinstance(engine, CustomApiEngine)
    assert engine.base_url == "https://api.example.com/v1/chat/completions"
    assert engine.api_key == "sk-test"
    assert engine.model == "gpt-4o-mini"


def test_build_engine_wires_anthropic_provider_from_config(tmp_path):
    """ai_provider='anthropic' must build an AnthropicEngine using the
    ANTHROPIC/api_key and model config values."""
    cache = TranslationCache(str(tmp_path / "cache.json"))
    service = TranslationService("ai", cache, AnthropicConfig(), ai_provider="anthropic")

    engine = service._build_engine()

    assert isinstance(engine, AnthropicEngine)
    assert engine.api_key == "sk-ant-test"
    assert engine.model == "claude-sonnet-4-5-20250929"
