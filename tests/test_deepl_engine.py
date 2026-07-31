"""Tests for mc_translator.engines.deepl.DeepLEngine -- specifically the
shield-placeholder safety net shared with GoogleEngine/BatchLlmEngine (see
test_google_engine.py / test_llm_common.py for the same regression test
against those engines)."""
from mc_translator.engines.base import EngineCallbacks, EngineItem
from mc_translator.engines.deepl import DeepLEngine


class _FakeResponse:
    def __init__(self, translations):
        self._translations = translations

    def raise_for_status(self):
        pass

    def json(self):
        return {"translations": [{"text": t} for t in self._translations]}


def make_callbacks() -> EngineCallbacks:
    return EngineCallbacks(
        should_run=lambda: True,
        wait_if_paused=lambda: None,
        on_log=lambda msg, tag="white": None,
        on_status=lambda msg: None,
    )


def test_omits_key_when_translator_mangles_shield_placeholder(monkeypatch):
    """Regression test (C1): if DeepL's response mangles a [#n#] shield
    placeholder beyond recognition, the key must be omitted -- never
    cache/ship a response with a leaked placeholder or a silently dropped
    format code."""
    items = {
        "x": EngineItem(key="x", original="Use %s here", masked="Use [#0#] here", mapping={"[#0#]": "%s"}),
    }
    monkeypatch.setattr(
        "mc_translator.engines.deepl.requests.post",
        lambda *a, **k: _FakeResponse(["Use <<0>> here"]),
    )
    engine = DeepLEngine(api_key="fake:fx")
    result = engine.translate_batch(items, {"deepl": "RU"}, make_callbacks())
    assert result == {}


def test_keeps_clean_translation(monkeypatch):
    items = {
        "x": EngineItem(key="x", original="Use %s here", masked="Use [#0#] here", mapping={"[#0#]": "%s"}),
    }
    monkeypatch.setattr(
        "mc_translator.engines.deepl.requests.post",
        lambda *a, **k: _FakeResponse(["Используй [#0#] тут"]),
    )
    engine = DeepLEngine(api_key="fake:fx")
    result = engine.translate_batch(items, {"deepl": "RU"}, make_callbacks())
    assert result == {"x": "Используй %s тут"}
