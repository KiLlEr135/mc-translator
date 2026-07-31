"""Tests for mc_translator.engines.google's GoogleEngine._translate_batch_mode
-- chunk-boundary logic, the empty-first-item guard, and the per-item
fallback when a chunk's response can't be split back into the right number
of parts. All exercised via a subclass overriding _request (no network)."""
from mc_translator.engines.base import EngineCallbacks, EngineItem
from mc_translator.engines.google import GoogleEngine


class RecordingGoogleEngine(GoogleEngine):
    """Records every _request call's text instead of hitting the network."""

    def __init__(self, *args, respond=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.requests: list[str] = []
        self._respond = respond or (lambda text: text)

    def _request(self, text, api_code, timeout=10):
        self.requests.append(text)
        return self._respond(text)


def make_items(n: int, char_len: int = 1) -> dict[str, EngineItem]:
    return {
        str(i): EngineItem(key=str(i), original=f"orig{i}", masked="x" * char_len, mapping={})
        for i in range(n)
    }


def make_callbacks() -> EngineCallbacks:
    return EngineCallbacks(
        should_run=lambda: True,
        wait_if_paused=lambda: None,
        on_log=lambda msg, tag="white": None,
        on_status=lambda msg: None,
    )


def test_empty_items_returns_empty_without_any_request():
    engine = RecordingGoogleEngine(mode="batch")
    result = engine.translate_batch({}, {"api": "ru"}, make_callbacks())
    assert result == {}
    assert engine.requests == []


def test_batch_mode_splits_at_20_item_boundary():
    """25 one-char items must split into two chunks (20 + 5), not one
    oversized chunk and not 25 individual requests."""
    engine = RecordingGoogleEngine(mode="batch", workers=1)
    items = make_items(25, char_len=1)
    result = engine.translate_batch(items, {"api": "ru"}, make_callbacks())
    assert len(engine.requests) == 2
    assert len(result) == 25
    for key in items:
        assert result[key] == "x"  # echoed back unchanged (no masking applied here)


def test_batch_mode_first_oversized_item_does_not_produce_empty_chunk():
    """Regression test: a first item alone longer than 2000 chars used to
    trip the flush condition on the very first loop iteration (keys still
    empty), appending an empty ([], "") chunk that sent a blank query to the
    API for no reason. The `if keys and (...)` guard prevents this -- the
    first item always joins the (empty) current chunk regardless of its own
    length; the flush only fires once a SECOND item would overflow it."""
    engine = RecordingGoogleEngine(mode="batch", workers=1)
    items = {
        "big": EngineItem(key="big", original="orig", masked="y" * 2500, mapping={}),
        "small": EngineItem(key="small", original="orig2", masked="z", mapping={}),
    }
    result = engine.translate_batch(items, {"api": "ru"}, make_callbacks())
    # No request text should ever be empty.
    assert all(text for text in engine.requests), f"an empty chunk was sent: {engine.requests}"
    assert len(result) == 2


def test_batch_mode_falls_back_to_single_requests_on_part_count_mismatch():
    """If a chunk's response can't be split into exactly as many parts as
    keys went in (e.g. the translator dropped/merged a separator), the
    engine must fall back to individual per-item requests rather than
    silently mis-mapping translations to the wrong keys."""

    def respond(text: str) -> str:
        if "|~|" in text:
            # Simulate a garbled multi-item response that won't split evenly.
            return "cannot-split-this-properly"
        return text  # single-item fallback requests still succeed

    engine = RecordingGoogleEngine(mode="batch", workers=1, respond=respond)
    items = make_items(3, char_len=5)
    result = engine.translate_batch(items, {"api": "ru"}, make_callbacks())
    assert len(result) == 3
    for key in items:
        assert result[key] == "xxxxx"


def test_single_mode_omits_keys_with_no_response():
    """translate_batch (mode='single') must omit a key whose _request
    returned None, rather than caching an empty/garbage translation --
    service.py's fallback then supplies the original text for that run."""

    def respond(text):
        return None if text == "fail" else text

    engine = RecordingGoogleEngine(mode="single", workers=2, respond=respond)
    items = {
        "ok": EngineItem(key="ok", original="orig", masked="succeed", mapping={}),
        "bad": EngineItem(key="bad", original="orig2", masked="fail", mapping={}),
    }
    result = engine.translate_batch(items, {"api": "ru"}, make_callbacks())
    assert result == {"ok": "succeed"}
    assert "bad" not in result


def test_single_mode_omits_key_when_translator_mangles_shield_placeholder():
    """Regression test (C1): if the translator returns a response where a
    [#n#] shield placeholder survives in some unrecognizable/mangled form
    (so unmask can't restore the protected format code/tag it stood for),
    the key must be omitted -- never cache/ship a response with a leaked
    placeholder or a silently-dropped format code."""

    def respond(text):
        # Simulate a translator that garbles the placeholder brackets into
        # something unmask_translation cannot recognize at all.
        return text.replace("[#0#]", "<<0>>")

    engine = RecordingGoogleEngine(mode="single", workers=1, respond=respond)
    items = {
        "x": EngineItem(key="x", original="Use %s here", masked="Use [#0#] here", mapping={"[#0#]": "%s"}),
    }
    result = engine.translate_batch(items, {"api": "ru"}, make_callbacks())
    assert result == {}
