"""Tests for mc_translator.engines.openrouter -- ModelRotator (thread-safe
round-robin over free OpenRouter models), Retry-After/X-RateLimit-Reset
parsing, free-model-list resolution, and OpenRouterEngine's request path
(both the original single-model 429 backoff and the auto-cycle rotation
layered on top of it). No real HTTP -- requests.post/get are monkeypatched,
same convention as test_kobold.py/test_cost_estimate.py."""
import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

from mc_translator.constants import FALLBACK_FREE_MODELS
from mc_translator.engines.base import EngineCallbacks, EngineItem
from mc_translator.engines.openrouter import OpenRouterEngine, fetch_free_models, resolve_free_models
from mc_translator.engines.rate_limit import ModelRotator, RateLimiter, parse_retry_after


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, headers=None, text="", reason=""):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._json_data = json_data if json_data is not None else {}
        self.headers = headers or {}
        self.text = text
        self.reason = reason

    def json(self):
        return self._json_data


class FakeConfig:
    def __init__(self, free_models=""):
        self._free_models = free_models

    def get(self, section, key):
        if section == "OPENROUTER" and key == "free_models":
            return self._free_models
        return ""


def make_callbacks(should_run=None, on_log=None, on_status=None):
    return EngineCallbacks(
        should_run=should_run or (lambda: True),
        wait_if_paused=lambda: None,
        on_log=on_log or (lambda msg, tag="white": None),
        on_status=on_status or (lambda msg: None),
    )


# ---------------------------------------------------------------------
# ModelRotator
# ---------------------------------------------------------------------


def test_model_rotator_requires_at_least_one_model():
    with pytest.raises(ValueError):
        ModelRotator([])


def test_model_rotator_stays_on_current_model_when_not_penalized():
    """Sequential (non-concurrent) re-acquisition: a caller that finished
    with a model (release(), no penalize()) gets the same model back next
    time -- switching away from a model that's working fine would be
    pointless churn."""
    rotator = ModelRotator(["a", "b", "c"])
    assert rotator.acquire() == ("a", 0.0)
    rotator.release("a")
    assert rotator.acquire() == ("a", 0.0)


def test_model_rotator_does_not_hand_the_same_model_to_two_concurrent_callers():
    """Regression test: a real 6h45m run showed 52 of 59 rotation events as
    exact back-to-back duplicates of the same model -- two callers racing
    acquire() before either had released/penalized got handed the SAME
    already-rate-limited model and both fired a doomed request at it. A
    second acquire() while the first is still in-flight (no release() yet)
    must get a DIFFERENT model when one is available."""
    rotator = ModelRotator(["a", "b", "c"])
    first = rotator.acquire()
    second = rotator.acquire()
    assert first[0] != second[0]
    assert first[1] == 0.0
    assert second[1] == 0.0


def test_model_rotator_releases_all_models_falls_back_to_reusing_one():
    """If every model is already claimed in-flight (more concurrent callers
    than free models), acquire() must still hand out a not-cooling-down
    model rather than stalling -- a rare duplicate hit is no worse than the
    pre-fix default behavior."""
    rotator = ModelRotator(["a", "b"])
    rotator.acquire()
    rotator.acquire()
    model, wait = rotator.acquire()
    assert model in ("a", "b")
    assert wait == 0.0


def test_model_rotator_release_after_penalize_is_a_harmless_no_op():
    rotator = ModelRotator(["a", "b"])
    rotator.acquire()
    rotator.penalize("a", 60.0)
    rotator.release("a")  # already cleared by penalize() -- must not raise
    model, wait = rotator.acquire()
    assert model == "b"
    assert wait == 0.0


def test_model_rotator_fallback_borrowed_model_refcounts_separately_from_original_holder():
    """Regression test: the all-claimed fallback path in acquire() (used
    when every model is already in-flight) used to hand out an
    already-claimed model WITHOUT incrementing its count, so the borrower's
    own release() wrongly cleared the ORIGINAL holder's still-active claim
    (a plain set can't distinguish "my claim" from someone else's -- a
    second holder's release() would silently free a model the first holder
    is still using, letting a third acquire() double up on it while the
    first request is still in flight). With a refcount, both claims are
    tracked and released independently."""
    rotator = ModelRotator(["a"])  # single model: every acquire() reuses "a"
    rotator.acquire()  # original holder: a's refcount -> 1
    rotator.acquire()  # borrower via the fallback path: a's refcount -> 2
    assert rotator._in_flight["a"] == 2
    rotator.release("a")  # borrower finishes first
    assert rotator._in_flight["a"] == 1  # original holder's claim must still stand
    rotator.release("a")  # original holder finishes
    assert "a" not in rotator._in_flight


def test_model_rotator_advances_to_next_after_penalize():
    rotator = ModelRotator(["a", "b", "c"])
    rotator.acquire()
    rotator.penalize("a", 60.0)
    model, wait = rotator.acquire()
    assert model == "b"
    assert wait == 0.0


def test_model_rotator_wraps_around_when_reaching_the_end():
    rotator = ModelRotator(["a", "b"])
    rotator.acquire()
    rotator.penalize("a", 60.0)
    rotator.acquire()  # now serving b
    rotator.penalize("b", 60.0)
    model, wait = rotator.acquire()
    # Both cooling down now -- serves whichever recovers first (a, penalized first).
    assert model == "a"
    assert wait > 0


def test_model_rotator_penalize_without_retry_after_uses_default_cooldown():
    rotator = ModelRotator(["a", "b"])
    rotator.penalize("a", None)
    model, wait = rotator.acquire()
    assert model == "b"
    assert wait == 0.0


def test_model_rotator_blacklist_removes_model_permanently():
    """Regression test: a real OpenRouter model (poolside/laguna-m.1:free)
    was listed as ":free" by OpenRouter's own catalog but 404'd on every
    actual request -- penalize()'s temporary cooldown just brought it back
    into rotation to fail the same way again, 354 times in one ~30-minute
    window. blacklist() must remove it for good, unlike penalize()."""
    rotator = ModelRotator(["a", "b", "c"])
    rotator.blacklist("b")
    assert "b" not in rotator.models
    assert rotator.models == ["a", "c"]
    # Never offered again, however many times acquire() is called.
    for _ in range(5):
        model, wait = rotator.acquire()
        assert model != "b"
        assert wait == 0.0
        rotator.release(model)


def test_model_rotator_blacklist_refuses_to_empty_the_rotation():
    """Blacklisting the last remaining model would leave acquire() with
    nothing to hand out (and crash on the empty-sequence min() call) -- must
    be a no-op instead, so the run always has at least one model to try."""
    rotator = ModelRotator(["a"])
    rotator.blacklist("a")
    assert rotator.models == ["a"]
    model, wait = rotator.acquire()
    assert model == "a"


def test_model_rotator_blacklist_releases_in_flight_claim():
    rotator = ModelRotator(["a", "b"])
    rotator.acquire()  # claims "a" (index starts at 0)
    rotator.blacklist("a")
    assert "a" not in rotator._in_flight


def test_model_rotator_blacklist_unknown_model_is_a_harmless_no_op():
    rotator = ModelRotator(["a", "b"])
    rotator.blacklist("not-in-the-list")
    assert rotator.models == ["a", "b"]


# ---------------------------------------------------------------------
# parse_retry_after
# ---------------------------------------------------------------------


def test_parse_retry_after_seconds_header():
    assert parse_retry_after(FakeResponse(headers={"Retry-After": "30"})) == 30.0


def test_parse_retry_after_http_date_header():
    future = datetime.now(timezone.utc) + timedelta(seconds=30)
    wait = parse_retry_after(FakeResponse(headers={"Retry-After": format_datetime(future)}))
    assert wait is not None
    assert 25 <= wait <= 35


def test_parse_retry_after_x_ratelimit_reset_ms_header():
    reset_ms = (time.time() + 40) * 1000
    wait = parse_retry_after(FakeResponse(headers={"X-RateLimit-Reset": str(reset_ms)}))
    assert wait is not None
    assert 35 <= wait <= 45


def test_parse_retry_after_returns_none_without_headers():
    assert parse_retry_after(FakeResponse(headers={})) is None


# ---------------------------------------------------------------------
# resolve_free_models / fetch_free_models
# ---------------------------------------------------------------------


def test_resolve_free_models_uses_manual_list_and_ignores_fetch(monkeypatch):
    monkeypatch.setattr("mc_translator.engines.openrouter.fetch_free_models", lambda: ["should:not-be-used"])
    config = FakeConfig(free_models="a:free\nb:free, c:free\n\n")
    assert resolve_free_models(config) == ["a:free", "b:free", "c:free"]


def test_resolve_free_models_dedupes_manual_list_preserving_order():
    config = FakeConfig(free_models="a:free\na:free\nb:free")
    assert resolve_free_models(config) == ["a:free", "b:free"]


def test_resolve_free_models_fetches_live_when_manual_list_empty(monkeypatch):
    monkeypatch.setattr("mc_translator.engines.openrouter.fetch_free_models", lambda: ["fetched:free"])
    assert resolve_free_models(FakeConfig(free_models="")) == ["fetched:free"]


def test_resolve_free_models_falls_back_to_builtin_list_when_fetch_fails(monkeypatch):
    monkeypatch.setattr("mc_translator.engines.openrouter.fetch_free_models", lambda: [])
    assert resolve_free_models(FakeConfig(free_models="")) == list(FALLBACK_FREE_MODELS)


def test_fetch_free_models_filters_to_free_suffix_only(monkeypatch):
    class FakeGetResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": "a:free"}, {"id": "b"}, {"id": "c:free"}]}

    monkeypatch.setattr("mc_translator.engines.openrouter.requests.get", lambda url, timeout=10: FakeGetResponse())
    assert fetch_free_models() == ["a:free", "c:free"]


def test_fetch_free_models_returns_empty_list_on_any_error(monkeypatch):
    def raise_error(url, timeout=10):
        raise RuntimeError("offline")

    monkeypatch.setattr("mc_translator.engines.openrouter.requests.get", raise_error)
    assert fetch_free_models() == []


# ---------------------------------------------------------------------
# OpenRouterEngine._request -- single-model mode (auto-cycle off, unchanged)
# ---------------------------------------------------------------------


def test_request_without_rotator_returns_content_on_success(monkeypatch):
    def fake_post(url, headers, json, timeout):
        return FakeResponse(json_data={"choices": [{"message": {"content": " hi "}}]})

    monkeypatch.setattr("mc_translator.engines.openrouter.requests.post", fake_post)
    engine = OpenRouterEngine("key", "some/model", rate_limiter=RateLimiter(min_interval=0.0))
    assert engine._request("prompt", 100) == "hi"


def test_request_without_rotator_retries_same_model_on_429_then_gives_up(monkeypatch):
    """Regression test: with auto-cycle off, behavior must stay exactly what
    it always was -- retry the SAME model up to 3 times on 429 with linear
    backoff (15s, 30s, 45s), then give up and return None."""
    posts = []
    sleeps = []

    def fake_post(url, headers, json, timeout):
        posts.append(json["model"])
        return FakeResponse(status_code=429)

    monkeypatch.setattr("mc_translator.engines.openrouter.requests.post", fake_post)
    monkeypatch.setattr("mc_translator.engines.openrouter.time.sleep", lambda s: sleeps.append(s))

    engine = OpenRouterEngine("key", "some/model:free", rate_limiter=RateLimiter(min_interval=0.0))
    assert engine._request("prompt", 100) is None
    assert posts == ["some/model:free"] * 3
    assert sleeps == [15, 30, 45]


def test_request_without_rotator_raises_on_other_http_errors(monkeypatch):
    def fake_post(url, headers, json, timeout):
        return FakeResponse(status_code=500, text="boom")

    monkeypatch.setattr("mc_translator.engines.openrouter.requests.post", fake_post)
    engine = OpenRouterEngine("key", "some/model", rate_limiter=RateLimiter(min_interval=0.0))
    with pytest.raises(Exception):
        engine._request("prompt", 100)


# ---------------------------------------------------------------------
# OpenRouterEngine -- auto-cycle mode (rotator supplied)
# ---------------------------------------------------------------------


def test_engine_switches_to_next_model_on_429_and_completes_translation(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        model = json["model"]
        calls.append(model)
        if model == "a:free":
            return FakeResponse(status_code=429, headers={"Retry-After": "0.05"})
        return FakeResponse(json_data={"choices": [{"message": {"content": '{"k": "translated"}'}}]})

    monkeypatch.setattr("mc_translator.engines.openrouter.requests.post", fake_post)

    rotator = ModelRotator(["a:free", "b:free"])
    engine = OpenRouterEngine("key", "a:free", rate_limiter=RateLimiter(min_interval=0.0), rotator=rotator)
    items = {"k": EngineItem(key="k", original="v", masked="v", mapping={})}
    logs = []
    callbacks = make_callbacks(on_log=lambda msg, tag="white": logs.append(msg))

    result = engine.translate_batch(items, {"name": "Russian"}, callbacks)

    assert result == {"k": "translated"}
    assert calls == ["a:free", "b:free"]
    assert any("🔁" in msg and "a:free" in msg for msg in logs)


def test_engine_penalizes_on_402_insufficient_quota_too(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        model = json["model"]
        calls.append(model)
        if model == "a:free":
            return FakeResponse(status_code=402)
        return FakeResponse(json_data={"choices": [{"message": {"content": '{"k": "translated"}'}}]})

    monkeypatch.setattr("mc_translator.engines.openrouter.requests.post", fake_post)

    rotator = ModelRotator(["a:free", "b:free"])
    engine = OpenRouterEngine("key", "a:free", rate_limiter=RateLimiter(min_interval=0.0), rotator=rotator)
    items = {"k": EngineItem(key="k", original="v", masked="v", mapping={})}

    result = engine.translate_batch(items, {"name": "Russian"}, make_callbacks())

    assert result == {"k": "translated"}
    assert calls == ["a:free", "b:free"]


def test_engine_blacklists_model_on_404_instead_of_cooling_down(monkeypatch):
    """Regression test for a real production incident: OpenRouter's own
    /models catalog listed "poolside/laguna-m.1:free" as available, but
    every actual chat-completions request to it 404'd with "No endpoints
    found" -- treating that like a rate limit (temporary cooldown) just
    brought it straight back into rotation to fail the same way again, 354
    times in a single ~30-minute window of a real run. A 404 must remove
    the model from rotation for good, not just cool it down."""
    calls = []

    def fake_post(url, headers, json, timeout):
        model = json["model"]
        calls.append(model)
        if model == "a:free":
            return FakeResponse(status_code=404, text='{"error":{"message":"No endpoints found for a:free.","code":404}}')
        return FakeResponse(json_data={"choices": [{"message": {"content": '{"k": "translated"}'}}]})

    monkeypatch.setattr("mc_translator.engines.openrouter.requests.post", fake_post)

    rotator = ModelRotator(["a:free", "b:free"])
    engine = OpenRouterEngine("key", "a:free", rate_limiter=RateLimiter(min_interval=0.0), rotator=rotator)
    items = {"k": EngineItem(key="k", original="v", masked="v", mapping={})}
    logs = []
    callbacks = make_callbacks(on_log=lambda msg, tag="white": logs.append(msg))

    result = engine.translate_batch(items, {"name": "Russian"}, callbacks)

    assert result == {"k": "translated"}
    assert calls == ["a:free", "b:free"]
    assert any("🚫" in msg and "a:free" in msg for msg in logs)
    assert "a:free" not in rotator.models  # gone for good, not just cooling down

    # A later batch (e.g. the next file in the same run) must never try the
    # blacklisted model again -- unlike a 429 cooldown, which eventually
    # expires and brings the model back.
    second_items = {"k2": EngineItem(key="k2", original="v2", masked="v2", mapping={})}
    calls.clear()
    engine.translate_batch(second_items, {"name": "Russian"}, make_callbacks())
    assert "a:free" not in calls


def test_engine_waits_for_cooldown_when_every_model_is_exhausted_then_recovers(monkeypatch):
    attempt = {"n": 0}
    calls = []

    def fake_post(url, headers, json, timeout):
        attempt["n"] += 1
        calls.append(json["model"])
        if attempt["n"] <= 2:
            return FakeResponse(status_code=429, headers={"Retry-After": "0.05"})
        return FakeResponse(json_data={"choices": [{"message": {"content": '{"k": "translated"}'}}]})

    monkeypatch.setattr("mc_translator.engines.openrouter.requests.post", fake_post)

    rotator = ModelRotator(["a:free", "b:free"])
    engine = OpenRouterEngine("key", "a:free", rate_limiter=RateLimiter(min_interval=0.0), rotator=rotator)
    items = {"k": EngineItem(key="k", original="v", masked="v", mapping={})}
    logs = []
    callbacks = make_callbacks(on_log=lambda msg, tag="white": logs.append(msg))

    result = engine.translate_batch(items, {"name": "Russian"}, callbacks)

    assert result == {"k": "translated"}
    # Both models take one failed hit each, then whichever cooled down first
    # (a, penalized first) is retried once its cooldown elapses and succeeds.
    assert calls == ["a:free", "b:free", "a:free"]


def test_post_with_truncation_retry_retries_once_with_doubled_max_tokens(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json["max_tokens"])
        if len(calls) == 1:
            return FakeResponse(json_data={"choices": [{"finish_reason": "length", "message": {"content": "cut off"}}]})
        return FakeResponse(json_data={"choices": [{"finish_reason": "stop", "message": {"content": "full"}}]})

    monkeypatch.setattr("mc_translator.engines.openrouter.requests.post", fake_post)
    engine = OpenRouterEngine("key", "some/model", rate_limiter=RateLimiter(min_interval=0.0))

    response = engine._post_with_truncation_retry("some/model", "prompt", 100)

    assert calls == [100, 200]
    assert response.json()["choices"][0]["finish_reason"] == "stop"


def test_post_with_truncation_retry_does_not_retry_a_complete_response(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json["max_tokens"])
        return FakeResponse(json_data={"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]})

    monkeypatch.setattr("mc_translator.engines.openrouter.requests.post", fake_post)
    engine = OpenRouterEngine("key", "some/model", rate_limiter=RateLimiter(min_interval=0.0))

    engine._post_with_truncation_retry("some/model", "prompt", 100)

    assert calls == [100]


def test_post_with_truncation_retry_respects_the_token_cap(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json["max_tokens"])
        return FakeResponse(json_data={"choices": [{"finish_reason": "length", "message": {"content": "x"}}]})

    monkeypatch.setattr("mc_translator.engines.openrouter.requests.post", fake_post)
    engine = OpenRouterEngine("key", "some/model", rate_limiter=RateLimiter(min_interval=0.0))

    engine._post_with_truncation_retry("some/model", "prompt", OpenRouterEngine.MAX_TOKENS_CAP)

    assert calls == [OpenRouterEngine.MAX_TOKENS_CAP]  # already at the cap -- no retry


def test_post_with_truncation_retry_leaves_rate_limit_responses_untouched(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json["max_tokens"])
        return FakeResponse(status_code=429)

    monkeypatch.setattr("mc_translator.engines.openrouter.requests.post", fake_post)
    engine = OpenRouterEngine("key", "some/model", rate_limiter=RateLimiter(min_interval=0.0))

    response = engine._post_with_truncation_retry("some/model", "prompt", 100)

    assert calls == [100]
    assert response.status_code == 429


def test_engine_stop_interrupts_a_cooldown_wait_instantly(monkeypatch):
    """Regression guard: a free-tier daily cap can mean a many-hour real
    cooldown (ModelRotator.acquire()'s reported wait). The wait loop must
    check should_run() before every sleep so Stop interrupts it immediately
    instead of blocking until the real cooldown elapses."""

    def fake_post(url, headers, json, timeout):
        return FakeResponse(status_code=429, headers={"Retry-After": "5"})

    monkeypatch.setattr("mc_translator.engines.openrouter.requests.post", fake_post)

    rotator = ModelRotator(["a:free"])
    engine = OpenRouterEngine("key", "a:free", rate_limiter=RateLimiter(min_interval=0.0), rotator=rotator)
    items = {"k": EngineItem(key="k", original="v", masked="v", mapping={})}

    call_count = {"n": 0}

    def should_run():
        call_count["n"] += 1
        return call_count["n"] <= 4  # let it reach the cooldown wait, then stop

    callbacks = make_callbacks(should_run=should_run)

    start = time.monotonic()
    result = engine.translate_batch(items, {"name": "Russian"}, callbacks)
    elapsed = time.monotonic() - start

    assert result == {}
    assert elapsed < 1.0  # never actually waited out the 5-second cooldown
