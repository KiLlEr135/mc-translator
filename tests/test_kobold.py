"""Tests for mc_translator.engines.kobold.KoboldEngine -- the local KoboldCPP
engine (see llm_common.py's batch_size/max_tokens override params)."""
from mc_translator.engines.kobold import KoboldEngine


def test_safe_mode_batch_and_token_budget():
    """safe mode now matches the shared BatchLlmEngine batch_size default for
    non-context mode (20) with a doubled token budget (4096, vs the shared
    default's 2048) -- it used to be halved (10/1024), sized for a
    since-replaced, much slower 14B/partial-GPU model config. Live-verified
    on a real ATM10 run that even 2048 still truncated mid-string on
    content-heavy mods; see kobold.py's constructor comment for the full
    margin analysis and history."""
    engine = KoboldEngine(mode="safe")
    assert engine.batch_size == 20
    assert engine.max_tokens == 4096


def test_context_mode_uses_smaller_batch_than_the_shared_default():
    engine = KoboldEngine(mode="context", context="SomeMod")
    assert engine.batch_size == 20
    assert engine.max_tokens == 4096


def test_request_includes_repeat_penalty(monkeypatch):
    """Regression test: a small/quantized local model rambling past its
    natural stopping point (no repeat penalty) turns a request into a slow,
    doomed generation that eats the whole 300s timeout for nothing."""
    captured = {}

    class FakeResponse:
        ok = True

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, json, timeout):
        captured.update(json)
        return FakeResponse()

    monkeypatch.setattr("mc_translator.engines.kobold.requests.post", fake_post)
    engine = KoboldEngine(mode="safe")
    engine._request("prompt", 1024)

    assert captured["repeat_penalty"] == 1.1


def test_request_includes_json_schema_response_format(monkeypatch):
    """Regression test: without a grammar constraint, a small/quantized local
    model frequently emits syntactically-broken JSON (real observed errors:
    "Expecting ',' delimiter", "Expecting ':' delimiter"), which used to be
    caught only after the fact and recovered via llm_common.py's expensive
    split-the-batch-and-retry fallback -- each retry a full extra LLM
    generation call. response_format:{"type":"json_schema"} makes KoboldCPP's
    sampler structurally incapable of emitting invalid JSON syntax, instead
    of just detecting and recovering from it. Must NOT be
    {"type":"json_object"} -- verified live that KoboldCPP's built-in grammar
    for that mode forces a top-level JSON ARRAY, not the flat object this
    app's prompt/parser expects."""
    captured = {}

    class FakeResponse:
        ok = True

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, json, timeout):
        captured.update(json)
        return FakeResponse()

    monkeypatch.setattr("mc_translator.engines.kobold.requests.post", fake_post)
    engine = KoboldEngine(mode="safe")
    engine._request("prompt", 1024)

    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "translations",
            "schema": {"type": "object", "additionalProperties": {"type": "string"}},
        },
    }


def test_request_retries_with_doubled_max_tokens_on_truncated_response(monkeypatch):
    """Regression test: a real production incident found long Patchouli/
    quest strings truncating identically on every rerun -- OpenRouterEngine
    already retried a truncated (finish_reason='length') response with a
    doubled token budget, but KoboldEngine didn't, so the same local-model
    truncation repeated forever. See llm_common.post_with_truncation_retry."""
    calls = []

    class FakeResponse:
        def __init__(self, finish_reason, content):
            self.ok = True
            self._finish_reason = finish_reason
            self._content = content

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"finish_reason": self._finish_reason, "message": {"content": self._content}}]}

    def fake_post(url, json, timeout):
        calls.append(json["max_tokens"])
        if len(calls) == 1:
            return FakeResponse("length", "cut off")
        return FakeResponse("stop", "full response")

    monkeypatch.setattr("mc_translator.engines.kobold.requests.post", fake_post)
    engine = KoboldEngine(mode="safe")

    result = engine._request("prompt", 100)

    assert calls == [100, 200]  # doubled on retry
    assert result == "full response"
