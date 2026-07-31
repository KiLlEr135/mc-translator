"""Tests for mc_translator.engines.llm_common — prompt building and batched
JSON translation shared by every AI engine (rate limiting/model rotation
moved to engines/rate_limit.py -- see test_rate_limit.py)."""
import json

import pytest

from mc_translator.engines.base import EngineCallbacks, EngineItem
from mc_translator.engines.llm_common import (
    BatchLlmEngine,
    build_glossary_hint,
    build_translation_prompt,
    parse_llm_json_response,
    post_with_truncation_retry,
)


def test_build_glossary_hint_empty_glossary():
    assert build_glossary_hint({}) == ""


def test_build_glossary_hint_includes_wrong_and_right_terms():
    hint = build_glossary_hint({"полуслой": "плита"})
    assert "полуслой" in hint
    assert "плита" in hint


def test_build_glossary_hint_respects_limit():
    glossary = {f"wrong{i}": f"right{i}" for i in range(100)}
    hint = build_glossary_hint(glossary, limit=5)
    assert hint.count("→") == 5


def test_build_translation_prompt_includes_glossary_hint_in_safe_mode():
    prompt = build_translation_prompt(
        {"k": "v"}, "Russian", mode="safe", context="", glossary_hint=" HINT_MARKER"
    )
    assert "HINT_MARKER" in prompt


def test_build_translation_prompt_includes_glossary_hint_in_context_mode():
    prompt = build_translation_prompt(
        {"k": "v"}, "Russian", mode="context", context="SomeMod", glossary_hint=" HINT_MARKER"
    )
    assert "HINT_MARKER" in prompt
    assert "SomeMod" in prompt


def test_build_translation_prompt_no_hint_by_default():
    prompt = build_translation_prompt({"k": "v"}, "Russian", mode="safe", context="")
    assert "→" not in prompt


def test_parse_llm_json_response_repairs_trailing_comma_before_brace():
    """Regression test: a real run against a small local model (Qwen2.5-3B)
    produced 'Illegal trailing comma before end of object' -- a harmless
    formatting slip that shouldn't sink an otherwise-valid translation."""
    content = '{"a": "1", "b": "2",}'
    assert parse_llm_json_response(content) == {"a": "1", "b": "2"}


def test_parse_llm_json_response_repairs_trailing_comma_before_bracket():
    content = '{"a": ["x", "y",]}'
    assert parse_llm_json_response(content) == {"a": ["x", "y"]}


def test_parse_llm_json_response_does_not_touch_a_comma_inside_a_string():
    """The trailing-comma repair must never fire on a comma that's part of
    an actual translated value, only on a real syntax error."""
    content = '{"a": "hello, world"}'
    assert parse_llm_json_response(content) == {"a": "hello, world"}


def test_parse_llm_json_response_still_raises_on_genuinely_broken_json():
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json_response('{"a": "unterminated')


# ---------------------------------------------------------------------
# post_with_truncation_retry -- shared by OpenRouterEngine/KoboldEngine/
# CustomApiEngine. Regression coverage: a real production incident found
# long Patchouli/quest strings truncating identically on every provider
# (OpenRouter already retried; Kobold/CustomApi didn't until this was
# factored out and shared).
# ---------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, ok=True, finish_reason="stop"):
        self.ok = ok
        self._finish_reason = finish_reason

    def json(self):
        return {"choices": [{"finish_reason": self._finish_reason}]}


def test_post_with_truncation_retry_retries_once_on_length_finish_reason():
    calls = []

    def post(max_tokens):
        calls.append(max_tokens)
        if len(calls) == 1:
            return _FakeResponse(finish_reason="length")
        return _FakeResponse(finish_reason="stop")

    result = post_with_truncation_retry(post, 100, 1000)

    assert calls == [100, 200]  # doubled on retry
    assert result.json()["choices"][0]["finish_reason"] == "stop"


def test_post_with_truncation_retry_does_not_retry_a_complete_response():
    calls = []

    def post(max_tokens):
        calls.append(max_tokens)
        return _FakeResponse(finish_reason="stop")

    post_with_truncation_retry(post, 100, 1000)

    assert calls == [100]


def test_post_with_truncation_retry_respects_the_cap():
    calls = []

    def post(max_tokens):
        calls.append(max_tokens)
        return _FakeResponse(finish_reason="length")

    post_with_truncation_retry(post, 1000, 1000)  # already at the cap

    assert calls == [1000]  # no retry attempted past the cap


def test_post_with_truncation_retry_leaves_error_responses_untouched():
    calls = []

    def post(max_tokens):
        calls.append(max_tokens)
        return _FakeResponse(ok=False)

    result = post_with_truncation_retry(post, 100, 1000)

    assert calls == [100]  # a 429/402/error response is never retried here
    assert result.ok is False


def test_post_with_truncation_retry_keeps_original_if_retry_also_fails():
    calls = []

    def post(max_tokens):
        calls.append(max_tokens)
        if len(calls) == 1:
            return _FakeResponse(finish_reason="length")
        return _FakeResponse(ok=False)

    result = post_with_truncation_retry(post, 100, 1000)

    assert calls == [100, 200]
    assert result.ok is True  # falls back to the original (truncated) response


def make_callbacks() -> EngineCallbacks:
    return EngineCallbacks(
        should_run=lambda: True,
        wait_if_paused=lambda: None,
        on_log=lambda msg, tag="white": None,
        on_status=lambda msg: None,
    )


def test_batch_llm_engine_omits_key_when_translator_mangles_shield_placeholder():
    """Regression test (C1): same safety net as GoogleEngine/DeepLEngine --
    if the LLM's JSON response contains a value where a [#n#] shield
    placeholder survives in a form unmask can't recognize, that key must be
    omitted (not cached/shipped with a leaked placeholder or a silently
    dropped format code). Also regression coverage for a real production
    mystery (161107, then ~650, then 394 strings staying untranslated with
    zero explanation in the log): this omission used to be completely
    silent -- only a per-run aggregate count, no way to tell which of the
    three omission reasons fired for a given string -- now logged."""
    items = {
        "x": EngineItem(key="x", original="Use %s here", masked="Use [#0#] here", mapping={"[#0#]": "%s"}),
    }

    def call_api(prompt: str, max_tokens: int) -> str:
        return json.dumps({"x": "Используй <<0>> тут"})

    logs = []
    callbacks = make_callbacks()
    callbacks.on_log = lambda msg, tag="white": logs.append(msg)
    engine = BatchLlmEngine(call_api=call_api)
    result = engine.translate_batch(items, {"name": "Russian"}, callbacks)

    assert result == {}
    assert any("защищённый код" in msg and "Use %s here" in msg for msg in logs)


def test_batch_llm_engine_omits_key_when_translation_has_foreign_script_contamination():
    """A free/low-quality OpenRouter model occasionally answers with a script
    that has no business in any supported language (observed for real: a
    Russian-target translation with a stray Arabic word mid-sentence) --
    that key must be omitted (falls back to the English original) rather
    than cached/shipped with the contamination. Now logged per-key (see
    the shield-placeholder test above for why)."""
    items = {
        "x": EngineItem(key="x", original="Some launchers like CurseForge", masked="Some launchers like CurseForge", mapping={}),
    }

    def call_api(prompt: str, max_tokens: int) -> str:
        return json.dumps({"x": "Некоторые лаунчеры, مثل CurseForge"})

    logs = []
    callbacks = make_callbacks()
    callbacks.on_log = lambda msg, tag="white": logs.append(msg)
    engine = BatchLlmEngine(call_api=call_api)
    result = engine.translate_batch(items, {"name": "Russian"}, callbacks)

    assert result == {}
    assert any("посторонний алфавит" in msg for msg in logs)


def test_batch_llm_engine_omits_key_when_translation_is_chinese_for_a_non_cjk_target():
    """Regression test: user reported once seeing the tool output Chinese
    characters instead of a real translation. CJK/Kana/Hangul are excluded
    from the universal foreign-script check because they're legitimate
    zh_cn/ja_jp/ko_kr TARGETS -- but that exclusion applied even when the
    actual target was Russian, so a model answering in Chinese passed
    silently. has_foreign_script_contamination now takes target_lang and
    only allows CJK when the target's own regex expects it."""
    items = {
        "x": EngineItem(key="x", original="A small hatchet", masked="A small hatchet", mapping={}),
    }

    def call_api(prompt: str, max_tokens: int) -> str:
        return json.dumps({"x": "手斧"})

    logs = []
    callbacks = make_callbacks()
    callbacks.on_log = lambda msg, tag="white": logs.append(msg)
    engine = BatchLlmEngine(call_api=call_api)
    result = engine.translate_batch(items, {"name": "Russian", "regex": r"[А-Яа-яЁё]"}, callbacks)

    assert result == {}
    assert any("посторонний алфавит" in msg for msg in logs)


def test_batch_llm_engine_omits_key_and_logs_when_response_missing_the_key():
    """The third silent-omission path: the model's JSON response simply
    doesn't include a requested key at all (or returns None/empty for it).
    Previously indistinguishable in the log from the other two omission
    reasons -- now logged with its own distinct message."""
    items = {
        "x": EngineItem(key="x", original="Hello there", masked="Hello there", mapping={}),
    }

    def call_api(prompt: str, max_tokens: int) -> str:
        return json.dumps({})  # "x" is simply missing from the response

    logs = []
    callbacks = make_callbacks()
    callbacks.on_log = lambda msg, tag="white": logs.append(msg)
    engine = BatchLlmEngine(call_api=call_api)
    result = engine.translate_batch(items, {"name": "Russian"}, callbacks)

    assert result == {}
    assert any("пустой/отсутствующий ответ" in msg and "Hello there" in msg for msg in logs)


def test_batch_llm_engine_keeps_clean_translation():
    items = {
        "x": EngineItem(key="x", original="Use %s here", masked="Use [#0#] here", mapping={"[#0#]": "%s"}),
    }

    def call_api(prompt: str, max_tokens: int) -> str:
        return json.dumps({"x": "Используй [#0#] тут"})

    engine = BatchLlmEngine(call_api=call_api)
    result = engine.translate_batch(items, {"name": "Russian"}, make_callbacks())
    assert result == {"x": "Используй %s тут"}


def test_batch_llm_engine_gives_up_after_consecutive_full_failures():
    """Regression test: a KoboldCPP hang once made a real run retry every
    remaining chunk of a file forever (each attempt taking minutes), because
    nothing capped consecutive fully-failed batches. After
    CONSECUTIVE_FAILURE_LIMIT fully-failed chunks in a row, translate_batch
    must give up on the rest of this file and flag backend_seems_dead."""
    items = {
        f"k{i}": EngineItem(key=f"k{i}", original=f"v{i}", masked=f"v{i}", mapping={})
        for i in range(100)  # 5 chunks of 20 (default "safe" batch_size)
    }
    attempted_keys: set[str] = set()
    logs = []

    def call_api(prompt: str, max_tokens: int) -> str | None:
        payload = json.loads(prompt.split("Data: ", 1)[1])
        attempted_keys.update(payload.keys())
        return None  # every request "succeeds" at the HTTP layer but returns
        # nothing usable -- same effective symptom as a request that times
        # out and gets logged as a failure.

    callbacks = EngineCallbacks(
        should_run=lambda: True,
        wait_if_paused=lambda: None,
        on_log=lambda msg, tag="white": logs.append(msg),
        on_status=lambda msg: None,
    )

    engine = BatchLlmEngine(call_api=call_api)
    result = engine.translate_batch(items, {"name": "Russian"}, callbacks)

    assert result == {}
    assert engine.backend_seems_dead is True
    assert any("не отвечает" in msg for msg in logs)
    # Gave up after 3 consecutive fully-failed chunks (60 items) -- the last
    # 2 chunks (40 items) must never have been attempted.
    assert attempted_keys == {f"k{i}" for i in range(60)}


def test_batch_llm_engine_circuit_breaker_does_not_trip_on_isolated_failure():
    """A single failed chunk that recovers via its split-retry (or is simply
    followed by successful chunks) is normal, transient behavior -- it must
    not be mistaken for a dead backend."""
    items = {
        f"k{i}": EngineItem(key=f"k{i}", original=f"v{i}", masked=f"v{i}", mapping={})
        for i in range(60)  # 3 chunks of 20
    }
    attempt = {"n": 0}

    def call_api(prompt: str, max_tokens: int) -> str | None:
        attempt["n"] += 1
        if attempt["n"] == 1:
            return None  # only the very first request (chunk 1's full attempt) fails
        payload = json.loads(prompt.split("Data: ", 1)[1])
        return json.dumps({k: f"t-{k}" for k in payload})

    engine = BatchLlmEngine(call_api=call_api)
    result = engine.translate_batch(items, {"name": "Russian"}, make_callbacks())

    assert engine.backend_seems_dead is False
    assert len(result) == 60  # every key recovered (chunk 1 via its split retry)


def test_batch_llm_engine_concurrent_branch_trips_circuit_breaker_too():
    """Regression test: the CONSECUTIVE_FAILURE_LIMIT/backend_seems_dead
    circuit breaker used to be implemented only in the sequential
    (max_concurrent<=1) branch -- OpenRouterEngine always uses
    max_concurrent=2 or 4, so a dead OpenRouter backend could be retried
    once per file for an entire run with nothing ever tripping the breaker."""
    items = {
        f"k{i}": EngineItem(key=f"k{i}", original=f"v{i}", masked=f"v{i}", mapping={})
        for i in range(100)  # 5 chunks of 20
    }
    logs = []

    def call_api(prompt: str, max_tokens: int) -> str | None:
        return None  # every request fails

    callbacks = EngineCallbacks(
        should_run=lambda: True,
        wait_if_paused=lambda: None,
        on_log=lambda msg, tag="white": logs.append(msg),
        on_status=lambda msg: None,
    )

    engine = BatchLlmEngine(call_api=call_api, max_concurrent=2)
    result = engine.translate_batch(items, {"name": "Russian"}, callbacks)

    assert result == {}
    assert engine.backend_seems_dead is True
    assert any("не отвечает" in msg for msg in logs)


def test_batch_llm_engine_fallback_recursively_isolates_a_single_bad_item():
    """Regression test: the fallback split used to go exactly ONE level deep
    and never retry a half that still failed, so a single item whose content
    genuinely broke the model's JSON output (not a transient blip) sank
    every other item sharing its half too -- e.g. one bad string in a
    20-item batch could lose up to 10 lines instead of just itself. A real
    6h45m run showed 73 lines lost from only ~10 malformed-JSON responses,
    matching this exact blast radius. The split must keep recursing past the
    first level for a genuinely malformed (retryable) response, isolating
    down to the single culprit."""
    items = {
        f"k{i}": EngineItem(key=f"k{i}", original=f"v{i}", masked=f"v{i}", mapping={})
        for i in range(20)  # one full "safe"-mode batch
    }

    def call_api(prompt: str, max_tokens: int) -> str | None:
        payload = json.loads(prompt.split("Data: ", 1)[1])
        if "k7" in payload:
            return "not valid json {"  # malformed response, not a bare failure
        return json.dumps({k: f"t-{k}" for k in payload})

    engine = BatchLlmEngine(call_api=call_api)
    result = engine.translate_batch(items, {"name": "Russian"}, make_callbacks())

    assert "k7" not in result
    assert result == {f"k{i}": f"t-k{i}" for i in range(20) if i != 7}


def test_batch_llm_engine_fallback_budget_caps_worst_case_requests_for_a_systemically_broken_backend():
    """Regression test: recursing on retryable (malformed-JSON) failures is
    only safe because it's bounded. Without a cap, a backend that returns
    unparsable JSON for EVERY sub-chunk (not just one poisoned item, but a
    systemically broken/incompatible model) would recurse every branch all
    the way to single items -- 2*len(chunk)-1 requests in the worst case
    (15 for this 8-item chunk), multiplying CONSECUTIVE_FAILURE_LIMIT's
    per-chunk cost far past the chunk's own size. The shared extra-attempt
    budget (== the chunk's own size) must cap total requests well below
    that uncapped worst case."""
    items = {
        f"k{i}": EngineItem(key=f"k{i}", original=f"v{i}", masked=f"v{i}", mapping={})
        for i in range(8)
    }
    call_count = {"n": 0}

    def call_api(prompt: str, max_tokens: int) -> str | None:
        call_count["n"] += 1
        return "not valid json {"  # every request is malformed (retryable), no exceptions

    engine = BatchLlmEngine(call_api=call_api, batch_size=8)
    result = engine.translate_batch(items, {"name": "Russian"}, make_callbacks())

    assert result == {}
    # Budget = len(chunk) extra attempts + the guaranteed first one = 9 max,
    # well below the uncapped worst case of 2*8-1 = 15.
    assert call_count["n"] <= 9


def test_batch_llm_engine_fallback_does_not_recurse_past_first_split_on_none_response():
    """Counterpart to the recursive-isolation test above: when the failure
    is a bare None/network response (not malformed JSON), splitting past the
    first level must NOT happen -- the backend itself is the problem there,
    and recursing deeper into a hung/dead server would multiply the wasted
    wait instead of isolating anything (this is exactly the runaway-retry
    shape CONSECUTIVE_FAILURE_LIMIT exists to cap)."""
    items = {
        f"k{i}": EngineItem(key=f"k{i}", original=f"v{i}", masked=f"v{i}", mapping={})
        for i in range(20)
    }
    call_count = {"n": 0}

    def call_api(prompt: str, max_tokens: int) -> str | None:
        call_count["n"] += 1
        return None  # every request fails at the HTTP/content layer

    engine = BatchLlmEngine(call_api=call_api)
    result = engine.translate_batch(items, {"name": "Russian"}, make_callbacks())

    assert result == {}
    # Exactly the pre-fix request count: 1 (full chunk) + 2 (first-level
    # split halves) = 3 -- never recurses into the halves a second time.
    assert call_count["n"] == 3


def test_batch_llm_engine_fallback_split_isolates_bad_item_at_batch_size_10():
    """Regression test: the fallback's sub-chunk size used to be hardcoded
    to 10, which equals KoboldEngine's default "safe"-mode batch_size --
    the "split and retry" fallback degenerated into a no-op resubmission of
    the SAME already-failed 10-item chunk, so one bad string sank all 10
    (9 genuinely translatable ones included) instead of being isolated."""
    items = {
        f"k{i}": EngineItem(key=f"k{i}", original=f"v{i}", masked=f"v{i}", mapping={})
        for i in range(10)  # exactly one Kobold "safe"-mode chunk
    }

    def call_api(prompt: str, max_tokens: int) -> str | None:
        payload = json.loads(prompt.split("Data: ", 1)[1])
        if len(payload) >= 10:
            return None  # the full 10-item batch always fails
        return json.dumps({k: f"t-{k}" for k in payload})  # any smaller sub-batch succeeds

    engine = BatchLlmEngine(call_api=call_api, batch_size=10)
    result = engine.translate_batch(items, {"name": "Russian"}, make_callbacks())

    assert result == {f"k{i}": f"t-k{i}" for i in range(10)}
