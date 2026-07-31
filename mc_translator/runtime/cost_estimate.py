"""
Rough pre-flight volume/cost estimate for paid translation engines (DeepL,
paid OpenRouter models), shown in the log before a run starts so the user can
hit Stop early if the estimate looks too expensive for their plan.

Character counts here are an approximation (average string length * string
count), not an exact count from re-reading every file — the estimator already
does one pass to count strings, and a byte-exact character count would need a
much larger rewrite across every processor's estimation helper for marginal
accuracy gain over a documented average.
"""
from __future__ import annotations

import requests

from mc_translator.constants import OPENROUTER_MODELS_URL

AVG_CHARS_PER_STRING = 25
CHARS_PER_TOKEN = 4  # rough rule of thumb for English/mixed text


def estimate_translation_chars(total_strings: int) -> int:
    return total_strings * AVG_CHARS_PER_STRING


def format_count(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def estimate_openrouter_cost(model: str, total_chars: int) -> str | None:
    """Best-effort estimate using OpenRouter's live public pricing. Returns
    None (never raises) on any failure — network issue, unknown model id, or
    a free/zero-priced model — so callers fall back to a plain char count."""
    try:
        resp = requests.get(OPENROUTER_MODELS_URL, timeout=5)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        info = next((m for m in models if m.get("id") == model), None)
        if not info:
            return None
        pricing = info.get("pricing", {})
        prompt_price = float(pricing.get("prompt", 0) or 0)
        completion_price = float(pricing.get("completion", 0) or 0)
        if prompt_price <= 0 and completion_price <= 0:
            return None
        input_tokens = total_chars / CHARS_PER_TOKEN
        output_tokens = input_tokens  # translations are roughly similar length to source
        cost = input_tokens * prompt_price + output_tokens * completion_price
        return f"${cost:.2f}"
    except Exception:
        return None
