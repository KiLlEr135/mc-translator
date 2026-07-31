import time

import requests

from mc_translator.constants import FALLBACK_FREE_MODELS, OPENROUTER_API, OPENROUTER_MODELS_URL
from mc_translator.engines.base import EngineCallbacks
from mc_translator.engines.llm_common import BatchLlmEngine, post_with_truncation_retry
from mc_translator.engines.rate_limit import (
    ModelRotator,
    RateLimiter,
    parse_model_list,
    parse_retry_after,
    wait_for_model_cooldown,
)


def fetch_free_models() -> list[str]:
    """Best-effort live fetch of every currently free (":free" suffix) model
    ID from OpenRouter's public /models endpoint. Never raises -- returns []
    on any failure (network issue, unexpected response shape), same
    error-handling contract as estimate_openrouter_cost in
    runtime/cost_estimate.py, so callers can fall back to a built-in list."""
    try:
        resp = requests.get(OPENROUTER_MODELS_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return [m["id"] for m in data if isinstance(m.get("id"), str) and m["id"].endswith(":free")]
    except Exception:
        return []


def resolve_free_models(config) -> list[str]:
    """Resolves the model list for auto-cycle mode. Priority: the user's own
    OPENROUTER/free_models setting (one model ID per line and/or
    comma-separated) if non-empty; otherwise every free model fetched live
    from OpenRouter; otherwise the small built-in FALLBACK_FREE_MODELS list,
    so the feature still works offline or if OpenRouter's /models endpoint
    is down."""
    manual = parse_model_list(config.get("OPENROUTER", "free_models") or "")
    if manual:
        return manual

    fetched = fetch_free_models()
    if fetched:
        return fetched

    return list(FALLBACK_FREE_MODELS)


class OpenRouterEngine(BatchLlmEngine):
    # Upper bound for the one-shot truncation-retry bump in
    # _post_with_truncation_retry -- prevents an already-huge batch from
    # doubling indefinitely (e.g. across repeated split-retry sub-chunks)
    # and keeps a single request within a sane cost/latency envelope.
    MAX_TOKENS_CAP = 8192

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        mode: str = "safe",
        context: str = "",
        site_url: str = "",
        app_name: str = "MC Translator",
        rate_limiter: RateLimiter | None = None,
        rotator: ModelRotator | None = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.site_url = site_url.strip()
        self.app_name = app_name.strip() or "MC Translator"
        # Free-tier OpenRouter models have tight per-minute quotas; paid models
        # can sustain more concurrent requests and a shorter pacing interval.
        # This only bounds the actual outgoing request RATE — concurrency
        # overlaps waiting on a slow response, it never fires more requests
        # per second than a fully sequential run would.
        # A rotator (auto-cycle mode, see resolve_free_models) only ever
        # contains ":free" models by construction, so its presence alone
        # means "treat this engine as free-tier" for pacing/concurrency.
        is_free_model = rotator is not None or ":free" in self.model.lower()
        # A caller (TranslationService) can supply a shared RateLimiter so
        # pacing persists across the many short-lived OpenRouterEngine
        # instances built over one run (translate_dict() -- and therefore
        # _build_engine() -- is called roughly once per file). Without this,
        # each new instance started its own RateLimiter with _next_slot=0.0,
        # so the first request of every new file fired immediately
        # regardless of how recently the previous file's last request went
        # out, bursting past the intended per-minute rate.
        self._rate_limiter = rate_limiter or RateLimiter(min_interval=4.0 if is_free_model else 1.5)
        # Also shared across every per-file OpenRouterEngine instance for the
        # same reason as _rate_limiter above -- None means auto-cycle is off
        # and _request() behaves exactly as it always has (single model,
        # linear 429 backoff).
        self._rotator = rotator
        # Stashed by translate_batch() (below) on every call so
        # _request_rotating() -- reached indirectly via the call_api
        # callback from a worker thread inside BatchLlmEngine -- can log,
        # update status, and honor Stop/Pause. _call_api's contract is
        # (prompt, max_tokens) -> str | None (see llm_common.py), with no
        # room for EngineCallbacks, unlike _translate_chunk which receives
        # them directly.
        self._active_callbacks: EngineCallbacks | None = None
        super().__init__(
            mode=mode,
            context=context,
            call_api=self._request,
            label="OpenRouter",
            max_concurrent=2 if is_free_model else 4,
        )

    def translate_batch(self, items, target_lang, callbacks):
        self._active_callbacks = callbacks
        return super().translate_batch(items, target_lang, callbacks)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        return headers

    def _post(self, model: str, prompt: str, max_tokens: int) -> requests.Response:
        return requests.post(
            OPENROUTER_API,
            headers=self._headers(),
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": max_tokens,
            },
            timeout=300,
        )

    def _post_with_truncation_retry(self, model: str, prompt: str, max_tokens: int) -> requests.Response:
        """See llm_common.post_with_truncation_retry for the full
        rationale -- this is just OpenRouter's _post bound into the shared
        implementation."""
        return post_with_truncation_retry(
            lambda mt: self._post(model, prompt, mt),
            max_tokens,
            self.MAX_TOKENS_CAP,
            label="OpenRouter",
            rate_limiter=self._rate_limiter,
        )

    @staticmethod
    def _extract_content(response: requests.Response) -> str | None:
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        if not isinstance(content, str):
            # None for a filtered/empty response; some models can also
            # return a non-string content shape (e.g. a list of parts).
            print("\n[Предупреждение] OpenRouter вернул пустой ответ (возможно, сработал фильтр модели).")
            return None
        return content.strip()

    def _request(self, prompt: str, max_tokens: int) -> str | None:
        if self._rotator is None:
            return self._request_single(prompt, max_tokens)
        return self._request_rotating(prompt, max_tokens)

    def _request_single(self, prompt: str, max_tokens: int) -> str | None:
        """Original single-model behavior (auto-cycle off): on HTTP 429,
        wait longer and retry the SAME model a bounded number of times
        before giving up. Left byte-for-byte equivalent to how this engine
        always behaved, so existing single-model runs are unaffected."""
        max_retries = 3

        for attempt in range(max_retries):
            self._rate_limiter.wait()
            response = self._post_with_truncation_retry(self.model, prompt, max_tokens)

            # Если словили лимит (429), ждем дольше и пробуем снова
            if response.status_code == 429:
                wait_time = 15 * (attempt + 1)
                print(f"\n[OpenRouter] Поймали лимит 429. Ждем {wait_time} сек. (Попытка {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue

            if not response.ok:
                detail = response.text[:200] if response.text else response.reason
                raise requests.HTTPError(f"{response.status_code}: {detail}", response=response)

            return self._extract_content(response)

        print("\n[Ошибка] Не удалось получить ответ: бесплатная модель слишком перегружена.")
        return None

    def _request_rotating(self, prompt: str, max_tokens: int) -> str | None:
        """Auto-cycle mode: instead of retrying the same model on a 429
        (rate limit) or 402 (insufficient quota/credits), switch to the next
        free model in self._rotator. When every model is cooling down, sleep
        interruptibly until the earliest one resets, then keep going -- this
        can legitimately run for hours against a free-tier daily cap, so it
        must never block Stop (should_run) or Pause."""
        callbacks = self._active_callbacks

        while True:
            if callbacks is not None:
                if not callbacks.should_run():
                    return None
                callbacks.wait_if_paused()
                if not callbacks.should_run():
                    return None

            model, wait = self._rotator.acquire()
            if wait > 0:
                if not wait_for_model_cooldown(wait, model, "OpenRouter", callbacks):
                    return None  # Stop was pressed while waiting
                continue

            # acquire() marks `model` in-flight so a concurrent caller (see
            # OpenRouterEngine's max_concurrent=2 for free models) won't be
            # handed the same not-yet-penalized model -- release() (always,
            # via finally) is what clears that claim again, whether this
            # request succeeded, got penalized, or raised.
            try:
                self._rate_limiter.wait()
                response = self._post_with_truncation_retry(model, prompt, max_tokens)

                if response.status_code in (429, 402):
                    retry_after = parse_retry_after(response)
                    self._rotator.penalize(model, retry_after)
                    message = f"🔁 {model}: лимит исчерпан, переключаюсь на следующую модель..."
                    if callbacks is not None:
                        callbacks.on_log(message, "yellow")
                    else:
                        print(f"\n[OpenRouter] {message}")
                    continue

                if response.status_code == 404:
                    # A real, live-observed case: OpenRouter's own /models
                    # catalog listed "poolside/laguna-m.1:free" as available,
                    # but every actual request to it 404'd with "No endpoints
                    # found" -- a temporary penalize() cooldown just brings a
                    # permanently dead model straight back into rotation to
                    # fail the exact same way again (354 wasted requests to
                    # this one model in a single ~30-minute window before
                    # this fix). Remove it from rotation for the rest of the
                    # run instead of cooling it down.
                    self._rotator.blacklist(model)
                    message = f"🚫 {model}: модель недоступна (404), исключаю из перебора..."
                    if callbacks is not None:
                        callbacks.on_log(message, "yellow")
                    else:
                        print(f"\n[OpenRouter] {message}")
                    continue

                if not response.ok:
                    detail = response.text[:200] if response.text else response.reason
                    raise requests.HTTPError(f"{response.status_code}: {detail}", response=response)

                return self._extract_content(response)
            finally:
                self._rotator.release(model)
