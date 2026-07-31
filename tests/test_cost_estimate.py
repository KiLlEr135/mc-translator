"""Tests for mc_translator.runtime.cost_estimate — pre-flight volume/cost estimate."""
from unittest.mock import MagicMock, patch

from mc_translator.runtime.cost_estimate import (
    estimate_openrouter_cost,
    estimate_translation_chars,
    format_count,
)


def test_estimate_translation_chars_uses_average():
    assert estimate_translation_chars(10) == 10 * 25


def test_format_count_uses_space_separator():
    assert format_count(1234567) == "1 234 567"


def _fake_models_response():
    return {
        "data": [
            {
                "id": "anthropic/claude-3.5-sonnet",
                "pricing": {"prompt": "0.000003", "completion": "0.000015"},
            },
            {
                "id": "google/gemma-2-9b-it:free",
                "pricing": {"prompt": "0", "completion": "0"},
            },
        ]
    }


def test_estimate_openrouter_cost_for_paid_model():
    def fake_get(url, timeout=None):
        resp = MagicMock()
        resp.raise_for_status = lambda: None
        resp.json = lambda: _fake_models_response()
        return resp

    with patch("requests.get", side_effect=fake_get):
        cost = estimate_openrouter_cost("anthropic/claude-3.5-sonnet", total_chars=10000)
    assert cost is not None
    assert cost.startswith("$")


def test_estimate_openrouter_cost_returns_none_for_free_model():
    def fake_get(url, timeout=None):
        resp = MagicMock()
        resp.raise_for_status = lambda: None
        resp.json = lambda: _fake_models_response()
        return resp

    with patch("requests.get", side_effect=fake_get):
        cost = estimate_openrouter_cost("google/gemma-2-9b-it:free", total_chars=10000)
    assert cost is None


def test_estimate_openrouter_cost_returns_none_for_unknown_model():
    def fake_get(url, timeout=None):
        resp = MagicMock()
        resp.raise_for_status = lambda: None
        resp.json = lambda: _fake_models_response()
        return resp

    with patch("requests.get", side_effect=fake_get):
        cost = estimate_openrouter_cost("nonexistent/model", total_chars=10000)
    assert cost is None


def test_estimate_openrouter_cost_returns_none_on_network_failure():
    def fake_get_fail(url, timeout=None):
        raise ConnectionError("no network")

    with patch("requests.get", side_effect=fake_get_fail):
        cost = estimate_openrouter_cost("anthropic/claude-3.5-sonnet", total_chars=10000)
    assert cost is None
