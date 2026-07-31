"""Request pacing and multi-model rotation, shared by every AI engine that
talks to a real rate-limited API (OpenRouterEngine, CustomApiEngine) --
split out of llm_common.py (which stays focused on the batched-JSON-
translation-over-chat-completions concern) purely to keep both files under
the project's 500-line guideline."""
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests

from mc_translator.engines.base import EngineCallbacks


class RateLimiter:
    """Spaces requests at least `min_interval` seconds apart across however
    many concurrent worker threads are calling `wait()`. Raising an engine's
    concurrency then overlaps *waiting on a slow response* instead of
    increasing the actual outgoing request rate — important for providers
    with tight per-minute quotas (e.g. free OpenRouter models)."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start_at = max(now, self._next_slot)
            self._next_slot = start_at + self.min_interval
        delay = start_at - now
        if delay > 0:
            time.sleep(delay)


class ModelRotator:
    """Thread-safe round-robin over a list of model IDs from a single
    provider, shared across every short-lived engine instance built over one
    run (mirrors RateLimiter above -- see TranslationService._build_engine,
    which builds a fresh engine roughly once per file). When a model's
    rate/quota limit is hit, penalize() puts it on a cooldown; acquire()
    then serves the next model in the list that isn't cooling down, wrapping
    back to the start once it reaches the end -- this is what makes
    consecutive requests actually switch models instead of retrying the one
    that just failed. If every model is cooling down, acquire() reports the
    one with the earliest cooldown expiry and how long to wait for it, so
    the caller can sleep (interruptibly) and try again.

    Originally OpenRouter-specific (engines/openrouter.py); generalized once
    CustomApiEngine needed the exact same rotation behavior for an
    arbitrary OpenAI-compatible provider (e.g. NVIDIA NIM) -- nothing in
    this class actually depends on OpenRouter."""

    # Used when the server didn't send a usable Retry-After/X-RateLimit-Reset
    # header alongside the 429/402 -- a conservative guess so a model isn't
    # retried immediately, but also isn't benched for an unreasonably long time.
    DEFAULT_COOLDOWN = 60.0

    def __init__(self, models: list[str]) -> None:
        if not models:
            raise ValueError("ModelRotator requires at least one model")
        self.models = list(models)
        self._lock = threading.Lock()
        self._index = 0
        self._cooldown_until: dict[str, float] = {}
        # Reference count of in-flight requests per model (between an
        # acquire() that returned it and that same caller's matching
        # release()). Without this, two threads racing acquire() before
        # either has called penalize()/release() both see the same
        # not-yet-penalized model and fire duplicate requests at it -- under
        # OpenRouterEngine's max_concurrent=2 for free models, this happened
        # on ~88% of real rotation events (52/59 groups were exact
        # back-to-back duplicates in a real 6h45m run), wasting roughly half
        # the requests made while cycling through a rate-limited storm.
        #
        # A plain set (rather than a refcount) doesn't work here: the
        # all-claimed fallback branch below can hand the same model to a
        # SECOND caller while a first caller still holds it, and that second
        # caller's release() would then wrongly clear the first caller's
        # still-active claim (both call plain release() with no way to tell
        # "my claim" from "someone else's"). Each acquire() that returns a
        # model increments its count; release()/penalize() decrement it, so
        # a model is only actually free again once every holder has let go.
        self._in_flight: dict[str, int] = {}

    def acquire(self) -> tuple[str, float]:
        """Returns (model, wait_seconds). wait_seconds is 0.0 when a model is
        immediately usable; otherwise it's how long until the earliest model
        recovers -- the caller is expected to wait and call acquire() again,
        not to fire a request at the returned model anyway.

        A model returned with wait_seconds == 0.0 has its in-flight count
        incremented, and the caller must pass it to release() exactly once
        (always, in a finally block, whether the request succeeded, was
        penalized, or raised) -- this is what stops a second concurrent
        acquire() from being handed the same not-yet-penalized model."""
        with self._lock:
            now = time.monotonic()
            n = len(self.models)
            for offset in range(n):
                idx = (self._index + offset) % n
                model = self.models[idx]
                if self._cooldown_until.get(model, 0.0) <= now and not self._in_flight.get(model, 0):
                    self._index = idx
                    self._in_flight[model] = self._in_flight.get(model, 0) + 1
                    return model, 0.0
            # Every not-cooling-down model is already claimed by another
            # in-flight request. Rather than stall the whole run waiting for
            # one to free up (the free-model list is normally much larger
            # than the handful of concurrent callers), fall back to handing
            # out a not-cooling-down model even if it's already in-flight --
            # still increments its count like the normal path, so this
            # caller's own release() only ever removes its own claim, never
            # the other holder's.
            for offset in range(n):
                idx = (self._index + offset) % n
                model = self.models[idx]
                if self._cooldown_until.get(model, 0.0) <= now:
                    self._index = idx
                    self._in_flight[model] = self._in_flight.get(model, 0) + 1
                    return model, 0.0
            soonest = min(self.models, key=lambda m: self._cooldown_until.get(m, 0.0))
            wait = max(0.0, self._cooldown_until[soonest] - now)
            return soonest, wait

    def release(self, model: str) -> None:
        """Removes exactly one in-flight claim acquire() placed on `model`.
        Safe to call even if the model has no claim left (e.g. penalize()
        already cleared it) -- a no-op in that case."""
        with self._lock:
            count = self._in_flight.get(model, 0)
            if count <= 1:
                self._in_flight.pop(model, None)
            else:
                self._in_flight[model] = count - 1

    def penalize(self, model: str, retry_after: float | None) -> None:
        """Marks `model` unavailable for retry_after seconds (or
        DEFAULT_COOLDOWN if the server didn't say) and advances the shared
        index past it, so the next acquire() call looks at the following
        model first instead of re-offering this one. Also releases this
        caller's own in-flight claim, same as release() -- it never touches
        another concurrent holder's claim on the same model."""
        with self._lock:
            cooldown = retry_after if retry_after and retry_after > 0 else self.DEFAULT_COOLDOWN
            self._cooldown_until[model] = time.monotonic() + cooldown
            count = self._in_flight.get(model, 0)
            if count <= 1:
                self._in_flight.pop(model, None)
            else:
                self._in_flight[model] = count - 1
            if model in self.models:
                self._index = (self.models.index(model) + 1) % len(self.models)

    def blacklist(self, model: str) -> None:
        """Permanently removes `model` from rotation for the rest of this
        run -- unlike penalize()'s temporary cooldown, this is for errors
        that mean the model is fundamentally broken (e.g. a real OpenRouter
        case: HTTP 404 "No endpoints found" for a model its own /models
        catalog still listed as ":free") rather than rate-limited. A
        temporary cooldown is the wrong tool there: the model would just
        come back into rotation and fail the exact same way again, forever
        -- live-verified as 354 wasted requests to one such dead model in a
        single ~30-minute window. No-op if it would empty the rotation
        entirely (falls back to leaving it in, still cooling down normally,
        so the run always has at least one model to try). Also releases
        this caller's own in-flight claim, same as release()/penalize()."""
        with self._lock:
            if model not in self.models or len(self.models) <= 1:
                count = self._in_flight.get(model, 0)
                if count <= 1:
                    self._in_flight.pop(model, None)
                else:
                    self._in_flight[model] = count - 1
                return
            self.models.remove(model)
            self._cooldown_until.pop(model, None)
            self._in_flight.pop(model, None)
            if self._index >= len(self.models):
                self._index = 0


def parse_retry_after(response: requests.Response) -> float | None:
    """Best-effort extraction of how long to wait before a rate limit
    resets, from whichever header the response actually carries. Returns
    None if nothing usable is present (caller falls back to a default
    cooldown) -- never raises."""
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            try:
                parsed = parsedate_to_datetime(retry_after)
            except (TypeError, ValueError):
                parsed = None
            if parsed is not None:
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())

    reset = response.headers.get("X-RateLimit-Reset")
    if reset:
        try:
            # OpenRouter documents this header as a Unix epoch in milliseconds.
            return max(0.0, float(reset) / 1000.0 - time.time())
        except ValueError:
            pass

    return None


def parse_model_list(raw: str) -> list[str]:
    """Parses a user-supplied model-ID list: one per line and/or
    comma-separated, deduplicated, order-preserved. Shared by OpenRouter's
    free_models setting and CUSTOM_AI's auto-cycle models setting."""
    manual: list[str] = []
    seen: set[str] = set()
    for line in (raw or "").splitlines():
        for part in line.split(","):
            model_id = part.strip()
            if model_id and model_id not in seen:
                seen.add(model_id)
                manual.append(model_id)
    return manual


def wait_for_model_cooldown(
    wait_seconds: float, model: str, label: str, callbacks: EngineCallbacks | None
) -> bool:
    """Sleeps up to wait_seconds in short (<=1s) increments so a Stop click
    interrupts even a multi-hour wait for a rate limit to reset. Returns
    False if interrupted by Stop, True once the wait completes normally.
    `label` is just the provider name (e.g. "OpenRouter", "Custom AI") for
    the status message -- shared by every rotation-capable engine."""
    deadline = time.monotonic() + wait_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        if callbacks is not None:
            if not callbacks.should_run():
                return False
            callbacks.wait_if_paused()
            callbacks.on_status(
                f"⏳ Все модели {label} исчерпаны — жду сброс лимита ({model}): {int(remaining)} с"
            )
        time.sleep(min(1.0, remaining))
