import requests

from mc_translator.constants import ANTHROPIC_API, ANTHROPIC_VERSION
from mc_translator.engines.llm_common import BatchLlmEngine
from mc_translator.engines.rate_limit import RateLimiter


class AnthropicEngine(BatchLlmEngine):
    """Claude via Anthropic's native Messages API -- unlike NVIDIA/OpenAI/
    Gemini/Grok (all OpenAI-compatible, covered by CustomApiEngine),
    Anthropic uses a different auth header (x-api-key, not Bearer), a
    required api-version header, and a different response shape (a
    `content` list of blocks instead of `choices[0].message.content`), so
    it needs its own _request rather than just another base_url."""

    def __init__(self, api_key: str, model: str, *, mode: str = "safe", context: str = "") -> None:
        self.api_key = api_key.strip()
        self.model = model.strip()
        super().__init__(
            mode=mode,
            context=context,
            call_api=self._request,
            label="Claude",
            max_concurrent=2,
        )
        self._rate_limiter = RateLimiter(min_interval=1.5)

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def _request(self, prompt: str, max_tokens: int) -> str | None:
        self._rate_limiter.wait()
        response = requests.post(
            ANTHROPIC_API,
            headers=self._headers(),
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=300,
        )
        response.raise_for_status()
        blocks = response.json().get("content")
        if not isinstance(blocks, list) or not blocks:
            return None
        text = blocks[0].get("text")
        return text.strip() if isinstance(text, str) else None
