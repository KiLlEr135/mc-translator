import requests

from mc_translator.constants import KOBOLD_API
from mc_translator.engines.llm_common import BatchLlmEngine, post_with_truncation_retry

# Constrains KoboldCPP's grammar sampler (supported on this OpenAI-compatible
# endpoint since v1.90 -- confirmed against the user's installed v1.117.1 and
# live-verified via /api/extra/json_to_grammar) to only ever emit a flat JSON
# object of string values -- makes the "Expecting ',' delimiter"/"Expecting
# ':' delimiter"-class syntax errors that used to trigger the expensive
# split-and-retry fallback (see llm_common.py's _translate_chunk_with_fallback)
# structurally impossible to generate, instead of just detecting and
# recovering from them after the fact. additionalProperties (not a fixed
# `properties`/`required` key list) because _request only receives the
# already-built prompt, not the batch's key set -- this constrains VALID
# JSON SHAPE, not the exact key names. response_format:{"type":"json_object"}
# was considered and rejected: on this server it forces a top-level JSON
# ARRAY, not the object this app's prompt/parser expects.
_JSON_OBJECT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "translations",
        "schema": {"type": "object", "additionalProperties": {"type": "string"}},
    },
}


class KoboldEngine(BatchLlmEngine):
    # Upper bound for the truncation-retry doubling in
    # post_with_truncation_retry -- matches OpenRouterEngine.MAX_TOKENS_CAP;
    # no evidence the installed local model needs a different cap.
    MAX_TOKENS_CAP = 8192

    def __init__(self, mode: str = "safe", context: str = "") -> None:
        super().__init__(
            mode=mode,
            context=context,
            call_api=self._request,
            label="KoboldCPP",
            # Both modes share the same budget: 20 items / 4096 output
            # tokens. The old halved values (10/1024) were sized for a
            # since-replaced 14B model with only 6 of ~48 layers on GPU
            # (~218s for a 20-string "safe" batch, measured against that
            # config) -- stale now that the local model is a fully
            # GPU-resident 3B with grammar-constrained JSON output: a
            # 20-item grammared batch is ~35-70s against the 300s timeout
            # (4-9x headroom, accounting for grammar sampling's own
            # throughput cost), even at this doubled token cap -- raising
            # the cap doesn't slow down a normal-length batch at all (the
            # model still stops at its own end-of-message token), it only
            # gives headroom to the rare long one.
            #
            # First went 1024 -> 2048 -- still not enough. Live-verified on
            # a real ATM10 run (Actuallyadditions, a content-heavy mod):
            # even at 2048, several batches were cut off mid-string around
            # ~6000 characters (with grammar guaranteeing valid JSON *syntax*
            # but unable to stop generation being truncated at the token
            # cap), logged as "Unterminated string"/"Invalid \uXXXX escape"
            # parse errors -- distinct from the delimiter/comma syntax
            # errors the grammar fix already eliminated -- each one forcing
            # the same expensive split-and-retry. 4096 gives real headroom
            # above the observed ~2000-2500 token shortfall instead of just
            # creeping past it.
            batch_size=20,
            max_tokens=4096,
        )

    def _post(self, prompt: str, max_tokens: int) -> requests.Response:
        return requests.post(
            KOBOLD_API,
            json={
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": max_tokens,
                # Mild repetition penalty as a cheap guard against a small/
                # quantized local model rambling past its natural stopping
                # point instead of emitting an end-of-message token -- that
                # failure mode is what turns a request into a slow, doomed
                # generation that eats the whole request timeout for nothing.
                "repeat_penalty": 1.1,
                "response_format": _JSON_OBJECT_RESPONSE_FORMAT,
            },
            timeout=300,
        )

    def _request(self, prompt: str, max_tokens: int) -> str | None:
        # Long Patchouli/quest text can legitimately need more than the
        # batch's default token budget -- without this, a truncated
        # response failed identically on every single rerun (the source
        # text's own length is the problem, not model quality). See
        # llm_common.post_with_truncation_retry.
        response = post_with_truncation_retry(
            lambda mt: self._post(prompt, mt), max_tokens, self.MAX_TOKENS_CAP, label="KoboldCPP"
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return content.strip() if isinstance(content, str) else None
