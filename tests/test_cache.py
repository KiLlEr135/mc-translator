"""Tests for mc_translator.cache.TranslationCache -- the on-disk cache every
translated string round-trips through. load_and_polish() runs on every
program start, so any non-idempotent rewrite it performs compounds forever
across restarts (this is exactly how the real ATM10 cache got corrupted)."""
import json

import pytest

from mc_translator import text_processing
from mc_translator.cache import TranslationCache


@pytest.fixture(autouse=True)
def _isolate_terminology_fixes(monkeypatch):
    """TERMINOLOGY_FIXES is a module-level dict loaded once at import time;
    tests must not leak dictionary.json rules into each other or depend on
    whatever happens to be in the real dictionary.json on disk."""
    monkeypatch.setattr(text_processing, "TERMINOLOGY_FIXES", {}, raising=False)


def test_load_and_polish_is_idempotent_across_restarts_with_expanding_dictionary_rule(tmp_path, monkeypatch):
    """Regression test: a dictionary.json rule whose replacement CONTAINS the
    search term (e.g. "медь" -> "сырая медь") is not idempotent under
    polish_translation. load_and_polish used to run the full
    polish_translation (cosmetic fixes + terminology substitution) over every
    cached value on EVERY load and persist the result, so a correct cached
    value "сырая медь" grew by one word per program restart:
    "сырая медь" -> "сырая сырая медь" -> "сырая сырая сырая медь" -> ...
    This must not happen: reloading an already-polished cache must be a
    no-op."""
    monkeypatch.setitem(text_processing.TERMINOLOGY_FIXES, "медь", "сырая медь")

    cache_file = tmp_path / "cache.json"
    cache_file.write_text(
        json.dumps({"ru_Raw Copper": "сырая медь"}, ensure_ascii=False),
        encoding="utf-8",
    )

    # Simulate several program restarts: construct a fresh TranslationCache
    # (which runs load_and_polish in __init__) repeatedly against the same
    # on-disk file, the way the real app does every launch.
    for _ in range(4):
        cache = TranslationCache(str(cache_file))
        assert cache.get("ru", "Raw Copper") == "сырая медь", (
            "cached value must not grow additional 'сырая' prefixes across restarts"
        )


def test_load_and_polish_still_applies_default_cosmetic_fixes_on_load(tmp_path):
    """load_and_polish must keep fixing cosmetic damage (spacing around
    format codes/placeholders) on every load -- only the non-idempotent
    terminology substitution is restricted (see the test above)."""
    cache_file = tmp_path / "cache.json"
    cache_file.write_text(
        json.dumps({"ru_x": "[ %s ]"}, ensure_ascii=False),
        encoding="utf-8",
    )
    cache = TranslationCache(str(cache_file))
    assert cache.get("ru", "x") == "[%s]"
    assert cache.polish_changes == 1


def test_load_and_polish_recovers_from_non_dict_json(tmp_path):
    """Regression test: a cache.json that parses as VALID JSON but isn't a
    dict (a stray list, string, or number -- e.g. a manual edit gone wrong,
    or a corrupted partial write) used to crash TranslationCache.__init__
    with an uncaught AttributeError on self._data.items() -- only
    JSONDecodeError/OSError were guarded. Must fall back to an empty cache,
    same as unparseable JSON."""
    cache_file = tmp_path / "cache.json"
    cache_file.write_text("[1, 2, 3]", encoding="utf-8")
    cache = TranslationCache(str(cache_file))
    assert len(cache) == 0
    assert cache.get("ru", "anything") is None


def test_load_and_polish_applies_terminology_fix_once_when_value_is_stale(tmp_path, monkeypatch):
    """A stale cached value that still contains the wrong term ("сыромятная
    медь") must still get corrected on load -- only compounding on an
    ALREADY-correct value is the bug."""
    monkeypatch.setitem(text_processing.TERMINOLOGY_FIXES, "сыромятная медь", "сырая медь")
    cache_file = tmp_path / "cache.json"
    cache_file.write_text(
        json.dumps({"ru_x": "Кусок сыромятная медь тут"}, ensure_ascii=False),
        encoding="utf-8",
    )
    cache = TranslationCache(str(cache_file))
    assert cache.get("ru", "x") == "Кусок сырая медь тут"
