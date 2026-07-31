"""Tests for mc_translator.engines.custom_api.CustomApiEngine -- the
user-configurable "any OpenAI-compatible API" engine (see kobold.py/
openrouter.py, which this mirrors: both are also just BatchLlmEngine
subclasses that implement _request). No real HTTP -- requests.post is
monkeypatched, same convention as test_kobold.py/test_openrouter.py.

429/402 handling waits out the rate limit and retries the SAME model
indefinitely (bounded only by Stop) rather than rotating across several
models or giving up after N retries -- see custom_api.py's docstring for
why: NVIDIA NIM's free-tier limit is account-wide, not per-model, so an
earlier model-rotation approach never actually avoided it."""
import requests

from mc_translator.engines.base import EngineCallbacks
from mc_translator.engines.custom_api import DEFAULT_COOLDOWN, CustomApiEngine


class FakeResponse:
    def __init__(self, json_data=None, status_code=200, headers=None):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.headers = headers or {}
        self.text = ""
        self.reason = ""

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._json_data


def make_callbacks(should_run=None, on_log=None, on_status=None):
    return EngineCallbacks(
        should_run=should_run or (lambda: True),
        wait_if_paused=lambda: None,
        on_log=on_log or (lambda msg, tag="white": None),
        on_status=on_status or (lambda msg: None),
    )


def test_defaults_to_moderate_concurrency_not_koboldlike_sequential():
    engine = CustomApiEngine("https://api.example.com/v1/chat/completions", "key", "some-model")
    assert engine.max_concurrent == 2


def test_request_sends_bearer_auth_header_when_api_key_present(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("mc_translator.engines.custom_api.requests.post", fake_post)
    engine = CustomApiEngine("https://api.example.com/v1/chat/completions", "sk-test", "gpt-4o-mini")
    engine._rate_limiter.min_interval = 0  # keep the test fast
    result = engine._request("prompt", 1024)

    assert result == "ok"
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["json"]["model"] == "gpt-4o-mini"
    assert captured["json"]["max_tokens"] == 1024


def test_request_omits_auth_header_when_api_key_is_blank(monkeypatch):
    """A local server (Ollama/LM Studio/vLLM) usually needs no API key --
    sending a stray 'Authorization: Bearer ' header could confuse a strict
    server, so it must be omitted entirely, not sent empty."""
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["headers"] = headers
        return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("mc_translator.engines.custom_api.requests.post", fake_post)
    engine = CustomApiEngine("http://localhost:1234/v1/chat/completions", "", "local-model")
    engine._rate_limiter.min_interval = 0
    engine._request("prompt", 1024)

    assert "Authorization" not in captured["headers"]


def test_request_returns_none_for_non_string_content(monkeypatch):
    def fake_post(url, headers, json, timeout):
        return FakeResponse({"choices": [{"message": {"content": None}}]})

    monkeypatch.setattr("mc_translator.engines.custom_api.requests.post", fake_post)
    engine = CustomApiEngine("https://api.example.com/v1/chat/completions", "key", "model")
    engine._rate_limiter.min_interval = 0

    assert engine._request("prompt", 1024) is None


def test_request_retries_with_doubled_max_tokens_on_truncated_response(monkeypatch):
    """Regression test: a real production incident found long Patchouli/
    quest strings truncating identically on every rerun regardless of AI
    provider -- OpenRouterEngine already retried a truncated (finish_
    reason='length') response with a doubled token budget, but
    CustomApiEngine (e.g. NVIDIA NIM) didn't, so the same truncation
    repeated forever. See llm_common.post_with_truncation_retry."""
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json["max_tokens"])
        if len(calls) == 1:
            return FakeResponse({"choices": [{"finish_reason": "length", "message": {"content": "cut off"}}]})
        return FakeResponse({"choices": [{"finish_reason": "stop", "message": {"content": "full response"}}]})

    monkeypatch.setattr("mc_translator.engines.custom_api.requests.post", fake_post)
    engine = CustomApiEngine("https://api.example.com/v1/chat/completions", "key", "model")
    engine._rate_limiter.min_interval = 0

    result = engine._request("prompt", 100)

    assert calls == [100, 200]  # doubled on retry
    assert result == "full response"


# ---------------------------------------------------------------------
# 429/402 handling -- waits out the SAME model's rate limit indefinitely
# instead of rotating across models or giving up after N retries.
# ---------------------------------------------------------------------


def test_waits_out_429_then_retries_same_model_and_succeeds(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json["model"])
        if len(calls) < 3:
            return FakeResponse(status_code=429)
        return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("mc_translator.engines.custom_api.requests.post", fake_post)
    monkeypatch.setattr("mc_translator.engines.custom_api.wait_for_model_cooldown", lambda *a, **kw: True)
    engine = CustomApiEngine("https://api.example.com/v1/chat/completions", "key", "model")
    engine._rate_limiter.min_interval = 0
    engine._active_callbacks = make_callbacks()

    result = engine._request("prompt", 1024)

    assert result == "ok"
    assert calls == ["model", "model", "model"]  # same model retried, never switched


def test_passes_retry_after_seconds_to_the_cooldown_wait(monkeypatch):
    waited = {}

    def fake_post(url, headers, json, timeout):
        return FakeResponse(status_code=429, headers={"Retry-After": "7"})

    def fake_wait(wait_seconds, model, label, callbacks):
        waited["seconds"] = wait_seconds
        waited["model"] = model
        waited["label"] = label
        return False  # pretend Stop was pressed so the test doesn't loop

    monkeypatch.setattr("mc_translator.engines.custom_api.requests.post", fake_post)
    monkeypatch.setattr("mc_translator.engines.custom_api.wait_for_model_cooldown", fake_wait)
    engine = CustomApiEngine("https://api.example.com/v1/chat/completions", "key", "model")
    engine._rate_limiter.min_interval = 0
    engine._active_callbacks = make_callbacks()

    assert engine._request("prompt", 1024) is None
    assert waited["seconds"] == 7.0
    assert waited["model"] == "model"
    assert waited["label"] == "Custom AI"


def test_falls_back_to_default_cooldown_without_retry_after_header(monkeypatch):
    waited = {}

    def fake_post(url, headers, json, timeout):
        return FakeResponse(status_code=429)

    def fake_wait(wait_seconds, model, label, callbacks):
        waited["seconds"] = wait_seconds
        return False

    monkeypatch.setattr("mc_translator.engines.custom_api.requests.post", fake_post)
    monkeypatch.setattr("mc_translator.engines.custom_api.wait_for_model_cooldown", fake_wait)
    engine = CustomApiEngine("https://api.example.com/v1/chat/completions", "key", "model")
    engine._rate_limiter.min_interval = 0
    engine._active_callbacks = make_callbacks()

    engine._request("prompt", 1024)

    assert waited["seconds"] == DEFAULT_COOLDOWN


def test_returns_none_when_stop_is_pressed_during_cooldown_wait(monkeypatch):
    """wait_for_model_cooldown returning False means Stop was pressed while
    waiting -- _request must propagate that as None, not keep retrying."""

    def fake_post(url, headers, json, timeout):
        return FakeResponse(status_code=429)

    monkeypatch.setattr("mc_translator.engines.custom_api.requests.post", fake_post)
    monkeypatch.setattr("mc_translator.engines.custom_api.wait_for_model_cooldown", lambda *a, **kw: False)
    engine = CustomApiEngine("https://api.example.com/v1/chat/completions", "key", "model")
    engine._rate_limiter.min_interval = 0
    engine._active_callbacks = make_callbacks()

    assert engine._request("prompt", 1024) is None


def test_returns_none_immediately_when_should_run_is_already_false(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(1)
        return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("mc_translator.engines.custom_api.requests.post", fake_post)
    engine = CustomApiEngine("https://api.example.com/v1/chat/completions", "key", "model")
    engine._rate_limiter.min_interval = 0
    engine._active_callbacks = make_callbacks(should_run=lambda: False)

    assert engine._request("prompt", 1024) is None
    assert calls == []  # never even posted


def test_non_rate_limit_error_raises_instead_of_waiting(monkeypatch):
    """A plain 500 is a real failure, not a rate-limit signal -- must
    propagate immediately, not be treated like a 429/402."""
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(1)
        return FakeResponse(status_code=500)

    monkeypatch.setattr("mc_translator.engines.custom_api.requests.post", fake_post)
    engine = CustomApiEngine("https://api.example.com/v1/chat/completions", "key", "model")
    engine._rate_limiter.min_interval = 0
    engine._active_callbacks = make_callbacks()

    try:
        engine._request("prompt", 1024)
        assert False, "expected HTTPError"
    except requests.HTTPError:
        pass
    assert len(calls) == 1
