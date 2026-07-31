"""Tests for mc_translator.engines.rate_limit -- RateLimiter and
parse_model_list. ModelRotator/parse_retry_after are covered in
test_openrouter.py (their original home before the module split), and the
rotation-request-path behavior is covered per-engine in test_openrouter.py/
test_custom_api.py."""
import time

from mc_translator.engines.rate_limit import RateLimiter, parse_model_list


def test_rate_limiter_spaces_out_calls():
    limiter = RateLimiter(min_interval=0.1)
    start = time.monotonic()
    limiter.wait()
    limiter.wait()
    limiter.wait()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.2  # 2 waits of >=0.1s between 3 calls


def test_parse_model_list_splits_lines_and_commas():
    assert parse_model_list("a\nb, c\n\n") == ["a", "b", "c"]


def test_parse_model_list_dedupes_preserving_order():
    assert parse_model_list("a\na\nb") == ["a", "b"]


def test_parse_model_list_empty_input_returns_empty_list():
    assert parse_model_list("") == []
    assert parse_model_list(None) == []
