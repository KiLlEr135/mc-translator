"""Tests for mc_translator.engines.anthropic.AnthropicEngine -- Claude via
Anthropic's native Messages API. Unlike CustomApiEngine's OpenAI-compatible
shape, this uses x-api-key auth (not Bearer), a required anthropic-version
header, and a `content` list-of-blocks response instead of `choices[0].
message.content`. No real HTTP -- requests.post is monkeypatched, same
convention as test_custom_api.py."""
from mc_translator.engines.anthropic import AnthropicEngine


class FakeResponse:
    def __init__(self, json_data):
        self._json_data = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


def test_request_uses_x_api_key_header_not_bearer(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResponse({"content": [{"type": "text", "text": "ok"}]})

    monkeypatch.setattr("mc_translator.engines.anthropic.requests.post", fake_post)
    engine = AnthropicEngine("sk-ant-test", "claude-sonnet-4-5-20250929")
    engine._rate_limiter.min_interval = 0  # keep the test fast
    result = engine._request("prompt", 1024)

    assert result == "ok"
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "sk-ant-test"
    assert "Authorization" not in captured["headers"]
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["json"]["model"] == "claude-sonnet-4-5-20250929"
    assert captured["json"]["max_tokens"] == 1024
    assert captured["json"]["messages"] == [{"role": "user", "content": "prompt"}]


def test_request_extracts_text_from_first_content_block(monkeypatch):
    def fake_post(url, headers, json, timeout):
        return FakeResponse({"content": [{"type": "text", "text": "  Привет  "}]})

    monkeypatch.setattr("mc_translator.engines.anthropic.requests.post", fake_post)
    engine = AnthropicEngine("key", "model")
    engine._rate_limiter.min_interval = 0

    assert engine._request("prompt", 1024) == "Привет"


def test_request_returns_none_for_empty_content_list(monkeypatch):
    def fake_post(url, headers, json, timeout):
        return FakeResponse({"content": []})

    monkeypatch.setattr("mc_translator.engines.anthropic.requests.post", fake_post)
    engine = AnthropicEngine("key", "model")
    engine._rate_limiter.min_interval = 0

    assert engine._request("prompt", 1024) is None


def test_request_returns_none_when_content_is_missing(monkeypatch):
    def fake_post(url, headers, json, timeout):
        return FakeResponse({})

    monkeypatch.setattr("mc_translator.engines.anthropic.requests.post", fake_post)
    engine = AnthropicEngine("key", "model")
    engine._rate_limiter.min_interval = 0

    assert engine._request("prompt", 1024) is None
