import requests

from mc_translator.engines.base import EngineCallbacks
from mc_translator.engines.llm_common import BatchLlmEngine, post_with_truncation_retry
from mc_translator.engines.rate_limit import RateLimiter, parse_retry_after, wait_for_model_cooldown

# Used when a 429/402 response doesn't carry a usable Retry-After header --
# a conservative guess so the wait isn't too short, but also isn't an
# unreasonably long stall when the server gives no better information.
DEFAULT_COOLDOWN = 60.0


class CustomApiEngine(BatchLlmEngine):
    """Any OpenAI-compatible chat-completions endpoint the user points it at
    -- OpenAI, Groq, DeepSeek, Mistral, Together, NVIDIA NIM, Gemini's
    OpenAI-compat endpoint, a local Ollama/LM Studio/vLLM server, etc. Same
    request/response shape as OpenRouterEngine/KoboldEngine (both already
    just BatchLlmEngine subclasses), minus OpenRouter-specific headers and
    Kobold's local-only grammar constraint -- an arbitrary third-party API
    isn't guaranteed to support either.

    On a 429 (rate limit) or 402 (quota), waits for the limit to reset and
    retries the SAME model indefinitely (bounded only by Stop) -- an earlier
    version instead rotated across several configured model IDs on 429
    (mirroring OpenRouter's free-model rotation), but for NVIDIA NIM that
    turned out to be pointless: its free-tier rate limit is account-wide,
    shared across every model behind one API key, not per-model, so
    switching models never actually dodged the limit. Waiting for the
    shared limit to reset is the only thing that works."""

    # Upper bound for the truncation-retry doubling in
    # post_with_truncation_retry -- matches OpenRouterEngine.MAX_TOKENS_CAP;
    # nothing here confirms a given third-party model's real context
    # window, so this is a conservative shared default, not a per-provider
    # tuned value.
    MAX_TOKENS_CAP = 8192

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        mode: str = "safe",
        context: str = "",
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.base_url = base_url.strip()
        self.api_key = api_key.strip()
        self.model = model.strip()
        # Stashed by translate_batch() on every call so _request() -- reached
        # indirectly via the call_api callback from a worker thread inside
        # BatchLlmEngine -- can log, update status, and honor Stop/Pause
        # while waiting out a rate limit.
        self._active_callbacks: EngineCallbacks | None = None
        super().__init__(
            mode=mode,
            context=context,
            call_api=self._request,
            label="Custom AI",
            # Unknown provider, so a moderate default rather than
            # OpenRouterEngine's paid-tier max_concurrent=4/min_interval=1.5:
            # cloud APIs generally tolerate a couple of concurrent requests
            # (unlike Kobold's single local GPU, max_concurrent=1), but
            # nothing here confirms the account is on a generous paid tier.
            max_concurrent=2,
        )
        # A caller (TranslationService) can supply a shared RateLimiter so
        # pacing persists across the many short-lived CustomApiEngine
        # instances built over one run, same reasoning as OpenRouterEngine.
        self._rate_limiter = rate_limiter or RateLimiter(min_interval=1.5)

    def translate_batch(self, items, target_lang, callbacks):
        self._active_callbacks = callbacks
        return super().translate_batch(items, target_lang, callbacks)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, prompt: str, max_tokens: int) -> requests.Response:
        return requests.post(
            self.base_url,
            headers=self._headers(),
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": max_tokens,
            },
            timeout=300,
        )

    @staticmethod
    def _extract_content(response: requests.Response) -> str | None:
        content = response.json().get("choices", [{}])[0].get("message", {}).get("content")
        return content.strip() if isinstance(content, str) else None

    def _request(self, prompt: str, max_tokens: int) -> str | None:
        """Real-world regression this replaced: a live NVIDIA NIM free-tier
        run hit 429 on nearly every request (203 in one ~1h run, live-
        verified via logs/mc_translator.log). A first fix added a bounded
        3-retry backoff, but that still gave up and marked the whole AI
        backend dead once NVIDIA's shared limit stayed hot past 3 tries --
        this waits it out for as long as it takes instead, same as
        OpenRouterEngine's rotator does when every model is cooling down."""
        callbacks = self._active_callbacks

        while True:
            if callbacks is not None:
                if not callbacks.should_run():
                    return None
                callbacks.wait_if_paused()
                if not callbacks.should_run():
                    return None

            self._rate_limiter.wait()
            response = post_with_truncation_retry(
                lambda mt: self._post(prompt, mt),
                max_tokens,
                self.MAX_TOKENS_CAP,
                label="Custom AI",
                rate_limiter=self._rate_limiter,
            )

            if response.status_code in (429, 402):
                retry_after = parse_retry_after(response)
                wait_time = retry_after if retry_after and retry_after > 0 else DEFAULT_COOLDOWN
                if not wait_for_model_cooldown(wait_time, self.model, "Custom AI", callbacks):
                    return None  # Stop was pressed while waiting
                continue

            response.raise_for_status()
            return self._extract_content(response)
