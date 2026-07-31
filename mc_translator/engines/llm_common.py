import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from mc_translator import text_processing
from mc_translator.engines.base import EngineCallbacks, EngineItem, TranslationEngine
from mc_translator.engines.rate_limit import RateLimiter
from mc_translator.text_processing import polish_translation, unmask_translation_strict

# Cap on how many dictionary.json entries get injected into a single prompt,
# so a large personal glossary doesn't blow up every request's token cost.
MAX_GLOSSARY_HINTS = 30

# After this many fully-failed batches in a row (the batch AND both of its
# split-retry halves produced zero translations), the backend is presumably
# hung or down -- give up on the rest of this file instead of retrying every
# remaining chunk forever (a real KoboldCPP hang once burned 2+ hours on a
# single file, each failed attempt taking up to 300s x 3, with no limit).
CONSECUTIVE_FAILURE_LIMIT = 3


def post_with_truncation_retry(
    post: Callable[[int], requests.Response],
    max_tokens: int,
    max_tokens_cap: int,
    *,
    label: str = "",
    rate_limiter: RateLimiter | None = None,
) -> requests.Response:
    """Calls post(max_tokens) once; if the response succeeded but the
    model's own finish_reason says the completion was cut off by the token
    cap (as opposed to stopping naturally), doubles max_tokens (capped at
    max_tokens_cap) and POSTs once more before returning. 429/402/error
    responses are returned exactly as received -- the caller's own
    rate-limit/error handling is unchanged.

    Long Patchouli guide pages / KubeJS quest text batched in "context"
    mode (batch_size=40, max_tokens=4096 -- see BatchLlmEngine defaults
    below) can legitimately need more than a batch's default completion
    budget; when that happens the model's JSON gets cut off mid-string,
    which otherwise only ever surfaces as a generic "Unterminated string"
    json.JSONDecodeError -- and _translate_chunk_with_fallback's split-and-
    retry fallback never changes max_tokens per sub-chunk, so the SAME
    truncation repeats on every retry no matter how small the sub-chunk
    gets. Worse: since the problem is the SOURCE TEXT's own length against
    a fixed budget, not any one model's quality, the identical string fails
    identically on every future rerun AND on every other provider/engine --
    live-verified as a real cause of specific strings staying untranslated
    forever regardless of whether OpenRouter, a custom API, or local
    KoboldCPP answers.

    Originally OpenRouter-only (its own free-tier truncation issues surfaced
    this first); shared here once Kobold/CustomApiEngine turned out to need
    the identical fix for the identical symptom, rather than duplicating
    this a third time."""
    response = post(max_tokens)
    if response.ok and max_tokens < max_tokens_cap:
        try:
            finish_reason = response.json().get("choices", [{}])[0].get("finish_reason")
        except ValueError:
            finish_reason = None
        if finish_reason == "length":
            bigger = min(max_tokens * 2, max_tokens_cap)
            print(
                f"\n[{label}] Ответ модели обрезан по лимиту токенов "
                f"({max_tokens}). Повторяю с лимитом {bigger}..."
            )
            if rate_limiter is not None:
                rate_limiter.wait()
            retried = post(bigger)
            if retried.ok:
                return retried
    return response


def build_glossary_hint(glossary: dict[str, str], limit: int = MAX_GLOSSARY_HINTS) -> str:
    """dictionary.json stores target-language corrections ('полуслой' -> 'плита'),
    not English-source-to-target pairs — it can't drive DeepL's native glossary
    API (which requires the entries match source_lang). But telling the LLM
    "if you were about to write X, write Y instead" works with this exact data
    shape and steers it away from known-bad terms up front, instead of relying
    purely on the post-hoc polish_translation regex fixups."""
    if not glossary:
        return ""
    pairs = list(glossary.items())[:limit]
    formatted = "; ".join(f'"{wrong}" → "{right}"' for wrong, right in pairs)
    return (
        f" Если бы ты хотел написать один из следующих вариантов слева, "
        f"напиши вместо этого вариант справа: {formatted}."
    )


def build_translation_prompt(
    payload: dict[str, str],
    lang_name: str,
    *,
    mode: str,
    context: str,
    glossary_hint: str = "",
) -> str:
    blob = json.dumps(payload, ensure_ascii=False)
    if mode == "context" and context:
        return (
            f"Ты локализатор Minecraft. Переведи строки мода/квеста «{context}» на {lang_name}. "
            f"Сохраняй игровой стиль и лор. Не переводи JSON-ключи. Теги [#0#] не менять."
            f"{glossary_hint} "
            f"Верни ТОЛЬКО строго валидный компактный JSON с теми же ключами: без висячих "
            f"запятых, с экранированием кавычек и переносов строк внутри значений. Данные: {blob}"
        )
    return (
        f"Translate JSON string values from English to {lang_name}. "
        f"Do not translate keys. Preserve [#0#] tags exactly."
        f"{glossary_hint} "
        f"Return ONLY strictly valid, compact JSON with the same keys: no trailing commas, "
        f"and escape any quotes or newlines inside string values. Data: {blob}"
    )


# Small/quantized local models occasionally leave a harmless trailing comma
# before a closing }/] -- strict json.loads rejects it even though no actual
# translated value is affected, so it's worth one lenient retry before
# treating an otherwise-well-formed response as a failure.
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def parse_llm_json_response(content: str) -> dict:
    text = re.sub(r"^```json\s*|^```\s*|```$", "", content.strip(), flags=re.IGNORECASE).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = json.loads(_TRAILING_COMMA_RE.sub(r"\1", text))
    if not isinstance(data, dict):
        raise TypeError("LLM response is not a JSON object")
    return data


class BatchLlmEngine(TranslationEngine):
    """Batched JSON translation via any chat-completions API."""

    def __init__(
        self,
        *,
        mode: str = "safe",
        context: str = "",
        call_api: Callable[[str, int], str | None],
        label: str = "ИИ",
        max_concurrent: int = 1,
        batch_size: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.mode = mode
        self.context = context
        self._call_api = call_api
        self.label = label
        # batch_size/max_tokens let a caller (e.g. KoboldEngine) scale these
        # down from the provider-agnostic defaults below for hardware that
        # needs a bigger safety margin under the network timeout.
        self.batch_size = batch_size if batch_size is not None else (40 if mode == "context" else 20)
        self.max_tokens = max_tokens if max_tokens is not None else (4096 if mode == "context" else 2048)
        # 1 = fully sequential (e.g. local KoboldCPP: one GPU, no benefit from
        # concurrent requests, real risk of VRAM thrashing). >1 overlaps
        # waiting on slow responses for providers that can actually take it
        # (see OpenRouterEngine, which also rate-limits the real request rate).
        self.max_concurrent = max(1, max_concurrent)
        # Set once translate_batch gives up early after CONSECUTIVE_FAILURE_LIMIT
        # fully-failed chunks -- TranslationService reads this to stop calling
        # this engine for the rest of the run (see service.py's translate_dict).
        self.backend_seems_dead = False

    def translate_batch(
        self,
        items: dict[str, EngineItem],
        target_lang: dict,
        callbacks: EngineCallbacks,
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        keys = list(items.keys())
        chunks = [keys[i : i + self.batch_size] for i in range(0, len(keys), self.batch_size)]

        if self.max_concurrent <= 1:
            consecutive_failures = 0
            for chunk in chunks:
                if not callbacks.should_run():
                    break
                callbacks.wait_if_paused()
                ok = self._translate_chunk_with_fallback(chunk, items, target_lang, result, callbacks)
                if ok:
                    consecutive_failures = 0
                    continue
                consecutive_failures += 1
                if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                    self.backend_seems_dead = True
                    callbacks.on_log(
                        f"❌ {self.label}: {CONSECUTIVE_FAILURE_LIMIT} пакета подряд "
                        f"полностью не удались — похоже, сервер не отвечает. "
                        f"Пропускаем остаток файла.",
                        "red",
                    )
                    break
            return result

        # Same consecutive-failure circuit breaker as the sequential branch
        # above, adapted for concurrency: _translate_chunk_with_fallback's
        # own return value (not a before/after len(result) snapshot) is the
        # per-chunk success signal, since `result` is a single dict mutated
        # from multiple worker threads at once -- a len() delta taken around
        # one future's fut.result() call could be corrupted by unrelated
        # concurrent writes from other still-running futures. "Consecutive"
        # here means consecutive in COMPLETION order, not submission order
        # (true concurrency has no single well-defined order) -- still
        # correctly catches the "backend is completely dead, every chunk is
        # failing" case this breaker exists for. Without this, OpenRouterEngine
        # (the only engine that ever uses max_concurrent > 1) could retry a
        # dead backend once per file for an entire run with no way to stop.
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            futures = []
            for chunk in chunks:
                if not callbacks.should_run():
                    break
                callbacks.wait_if_paused()
                if not callbacks.should_run():
                    break
                futures.append(
                    pool.submit(
                        self._translate_chunk_with_fallback, chunk, items, target_lang, result, callbacks
                    )
                )
            consecutive_failures = 0
            for fut in as_completed(futures):
                if not callbacks.should_run():
                    break
                if fut.result():
                    consecutive_failures = 0
                    continue
                consecutive_failures += 1
                if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                    self.backend_seems_dead = True
                    callbacks.on_log(
                        f"❌ {self.label}: {CONSECUTIVE_FAILURE_LIMIT} пакета подряд "
                        f"полностью не удались — похоже, сервер не отвечает. "
                        f"Пропускаем остаток файла.",
                        "red",
                    )
                    break
        return result

    def _translate_chunk_with_fallback(
        self,
        chunk: list[str],
        items: dict[str, EngineItem],
        target_lang: dict,
        result: dict[str, str],
        callbacks: EngineCallbacks,
        *,
        _depth: int = 0,
        _budget: list[int] | None = None,
    ) -> bool:
        """Returns True if the chunk (or at least one of its fallback splits)
        produced a translation.

        Regression: this used to split a failing chunk exactly ONE level
        deep and never retry a half that still failed, so a single item
        whose content genuinely broke the model's JSON output (e.g. an
        unescaped control character) sank every other item sharing its half
        too -- not just itself (a real run lost 73 lines this way from only
        ~10 malformed-JSON responses). Beyond the first split, this now keeps
        recursively halving a still-failing (sub-)chunk down to a single
        item -- but ONLY when the failure is `retryable` (a malformed/
        unparsable response -- the server replied, just badly, so retrying
        smaller is cheap and can actually isolate the culprit). A `None`
        response or network error is never retryable beyond the first split:
        the backend itself is the problem there, and recursing deeper into a
        hung/dead server would multiply the wasted wait per CONSECUTIVE_FAILURE_LIMIT's
        original incident (a KoboldCPP hang once burned 2+ hours on one file).

        `_budget` bounds how many EXTRA (recursive) attempts one top-level
        chunk's fallback can spend in total, shared across the whole
        recursion tree via the same mutable list -- without it, a backend
        that returns malformed (retryable) JSON for essentially every
        sub-chunk, not just one poisoned item, could recurse all the way to
        single items on every branch: up to 2*len(chunk)-1 requests instead
        of the isolated-single-item case's ~2*log2(len(chunk)), multiplying
        CONSECUTIVE_FAILURE_LIMIT's per-chunk cost 10x+ in exactly the
        hung/misbehaving-backend scenario that limit exists to cap. Capping
        the extra-attempt budget at the chunk's own size bounds the total to
        at most 2*len(chunk)+1 requests while leaving plenty of headroom for
        the realistic case (isolating one or a few genuinely bad items)."""
        if _budget is None:
            _budget = [len(chunk)]
        ok, retryable = self._translate_chunk(chunk, items, target_lang, result, callbacks)
        if ok:
            return True
        if len(chunk) <= 1:
            return False
        if _depth > 0 and not retryable:
            return False
        callbacks.on_log(f"❌ Ошибка {self.label}. Дробим пакет...", "yellow")
        # Half of THIS chunk's size, not a fixed constant: a fixed size equal
        # to or larger than the caller's batch_size (e.g. KoboldEngine's
        # "safe" mode, batch_size=10) made the "split" a no-op that just
        # resubmitted the identical already-failed chunk unchanged, instead
        # of isolating whichever single item actually caused the failure.
        sub_size = max(1, len(chunk) // 2)
        recovered = False
        for j in range(0, len(chunk), sub_size):
            if not callbacks.should_run():
                break
            callbacks.wait_if_paused()
            if _budget[0] <= 0:
                break
            _budget[0] -= 1
            sub = chunk[j : j + sub_size]
            if self._translate_chunk_with_fallback(
                sub, items, target_lang, result, callbacks, _depth=_depth + 1, _budget=_budget
            ):
                recovered = True
        return recovered

    def _translate_chunk(
        self,
        chunk_keys: list[str],
        items: dict[str, EngineItem],
        target_lang: dict,
        result: dict[str, str],
        callbacks: EngineCallbacks,
    ) -> tuple[bool, bool]:
        payload = {k: items[k].masked for k in chunk_keys}
        # Read text_processing.TERMINOLOGY_FIXES fresh each call (module-qualified,
        # not imported by name) so a mid-session dictionary reload (see
        # gui/app.py's _start_translation) takes effect on the next chunk.
        glossary_hint = build_glossary_hint(text_processing.TERMINOLOGY_FIXES)
        prompt = build_translation_prompt(
            payload,
            target_lang["name"],
            mode=self.mode,
            context=self.context,
            glossary_hint=glossary_hint,
        )
        callbacks.on_status(f"⏳ {self.label}: пакет {len(chunk_keys)} строк...")
        try:
            content = self._call_api(prompt, self.max_tokens)
            if not content:
                # No response at all (empty/filtered/exhausted retries) --
                # not a malformed-JSON problem a smaller chunk can isolate,
                # so not retryable (see _translate_chunk_with_fallback).
                return False, False
            translated = parse_llm_json_response(content)
            for key in chunk_keys:
                value = translated.get(key)
                original_preview = items[key].original[:40]
                if isinstance(value, str) and value:
                    text = unmask_translation_strict(value, items[key].mapping)
                    if text is None:
                        # The LLM mangled or fully dropped a shield
                        # placeholder (e.g. reformatted or omitted [#n#])
                        # badly enough that unmask couldn't restore it --
                        # omit the key like any other failure below, rather
                        # than shipping a corrupted/incomplete result. Logged
                        # (not just silently dropped) since this was
                        # previously invisible except as an aggregate
                        # per-run "не переведено" count with no way to tell
                        # WHICH of the three omission reasons below actually
                        # fired for a given string.
                        callbacks.on_log(
                            f"⚠ {self.label}: потерян защищённый код в «{original_preview}» "
                            f"-> «{value[:60]}»",
                            "yellow",
                        )
                        continue
                    if text_processing.has_foreign_script_contamination(text, target_lang):
                        # A free/low-quality model occasionally answers with
                        # a script that has no business in any supported
                        # language (e.g. a stray Arabic word inside an
                        # otherwise-Russian sentence) -- omit the key like
                        # any other failure so service.py falls back to the
                        # English original instead of caching/shipping it.
                        callbacks.on_log(
                            f"⚠ {self.label}: посторонний алфавит в переводе «{original_preview}» "
                            f"-> «{text[:60]}»",
                            "yellow",
                        )
                        continue
                    result[key] = polish_translation(text)
                else:
                    # Omit the key (matches TranslationEngine.translate_batch's
                    # "may omit keys on failure" contract) so service.py's own
                    # non-caching fallback supplies the original text instead of
                    # us permanently caching a bogus "None"/original value.
                    callbacks.on_log(
                        f"⚠ {self.label}: пустой/отсутствующий ответ для «{original_preview}»",
                        "yellow",
                    )
            return True, False
        except (json.JSONDecodeError, TypeError, KeyError, IndexError, AttributeError) as exc:
            # AttributeError covers a malformed API response reaching
            # `.strip()` on a non-string/None `content` inside _call_api
            # (e.g. KoboldEngine/OpenRouterEngine._request) -- without it,
            # this used to propagate out of the worker thread and abort the
            # whole run (see job.py's except Exception, which used to skip
            # cache.save() before that was moved into finally).
            # Retryable: the server DID respond, just with something that
            # doesn't parse -- a smaller chunk can isolate which single item
            # actually confused the model.
            callbacks.on_log(f"❌ {self.label}: неверный ответ — {exc}", "red")
            return False, True
        except requests.RequestException as exc:
            # Not retryable: a network/timeout failure means the backend
            # itself is the problem, not this chunk's content -- retrying
            # ever-smaller chunks against a hung/dead server would just
            # multiply the wasted wait (see CONSECUTIVE_FAILURE_LIMIT above).
            callbacks.on_log(f"❌ {self.label}: сеть — {exc}", "red")
            return False, False
