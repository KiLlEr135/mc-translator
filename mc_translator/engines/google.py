import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from mc_translator.engines.base import EngineCallbacks, EngineItem, TranslationEngine
from mc_translator.text_processing import polish_translation, unmask_translation_strict


class GoogleEngine(TranslationEngine):
    API_URL = "https://translate.googleapis.com/translate_a/single"
    BATCH_SEP = " |~| "

    def __init__(self, workers: int = 5, mode: str = "single") -> None:
        self.workers = max(1, min(workers, 10))
        self.mode = mode

    def _request(self, text: str, api_code: str, timeout: int = 10) -> str | None:
        for _ in range(3):
            try:
                r = requests.get(
                    self.API_URL,
                    params={"client": "gtx", "sl": "en", "tl": api_code, "dt": "t", "q": text},
                    timeout=timeout,
                )
                if r.status_code == 429:
                    time.sleep(3)
                    continue
                if r.ok:
                    data = r.json()
                    # Google returns [null, ...] for empty/blocked/untranslatable
                    # input -- data[0] is None, and iterating/subscripting it
                    # raises TypeError (not caught below), which used to
                    # propagate out of the ThreadPoolExecutor worker and abort
                    # the whole batch/run.
                    if not data or not data[0]:
                        return None
                    return "".join(part[0] for part in data[0] if part and part[0])
            except (requests.RequestException, KeyError, IndexError, ValueError, TypeError):
                time.sleep(1)
        return None

    def _finalize(self, raw: str, item: EngineItem) -> str | None:
        text = unmask_translation_strict(raw, item.mapping)
        if text is None:
            # The translator mangled or fully dropped a shield placeholder
            # (e.g. turned our ASCII [#n#] into a full-width bracket variant,
            # or omitted it outright) badly enough that a protected format
            # code/tag was lost -- treat like any other engine failure: omit
            # the key so service.py's non-caching fallback supplies the
            # English original instead of shipping a corrupted/incomplete
            # result.
            return None
        return polish_translation(text)

    def translate_batch(
        self,
        items: dict[str, EngineItem],
        target_lang: dict,
        callbacks: EngineCallbacks,
    ) -> dict[str, str]:
        if not items:
            return {}
        api_code = target_lang["api"]
        if self.mode == "batch":
            return self._translate_batch_mode(items, api_code, callbacks)
        return self._translate_single_mode(items, api_code, callbacks)

    def _translate_single_mode(
        self, items: dict[str, EngineItem], api_code: str, callbacks: EngineCallbacks
    ) -> dict[str, str]:
        result: dict[str, str] = {}

        def work(key: str, masked: str) -> tuple[str, str | None]:
            if not callbacks.should_run():
                return key, None
            callbacks.wait_if_paused()
            if not callbacks.should_run():
                return key, None
            return key, self._request(masked, api_code)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(work, k, v.masked): k for k, v in items.items()}
            for fut in as_completed(futures):
                callbacks.wait_if_paused()
                if not callbacks.should_run():
                    break
                key, raw = fut.result()
                if raw:
                    finalized = self._finalize(raw, items[key])
                    if finalized is not None:
                        result[key] = finalized
                # else: omit the key so service.py's non-caching fallback
                # supplies the original text instead of us permanently
                # caching it as if it were a real translation.
        return result

    def _translate_batch_mode(
        self, items: dict[str, EngineItem], api_code: str, callbacks: EngineCallbacks
    ) -> dict[str, str]:
        chunks: list[tuple[list[str], str]] = []
        keys: list[str] = []
        current = ""

        for key, item in items.items():
            if keys and (len(current) + len(item.masked) > 2000 or len(keys) >= 20):
                # `keys` guard above: without it, a first item alone longer
                # than 2000 chars would trip this on the very first
                # iteration (keys still empty) and append an empty ([], "")
                # chunk, which then sends a blank query to the API for no
                # reason and always falls through the len(parts)==len(keys)
                # check.
                chunks.append((keys, current))
                keys = [key]
                current = item.masked
            else:
                keys.append(key)
                current = current + self.BATCH_SEP + item.masked if current else item.masked
        if keys:
            chunks.append((keys, current))

        result: dict[str, str] = {}

        def translate_chunk(chunk_keys: list[str], text: str) -> tuple[list[str], list[str] | None]:
            if not callbacks.should_run():
                return chunk_keys, None
            callbacks.wait_if_paused()
            if not callbacks.should_run():
                return chunk_keys, None
            raw = self._request(text, api_code)
            if not raw:
                return chunk_keys, None
            parts = re.split(r"\s*\|\s*~\s*\|\s*", raw)
            if len(parts) == len(chunk_keys):
                return chunk_keys, parts
            return chunk_keys, None

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(translate_chunk, ck, ct) for ck, ct in chunks]
            for fut in as_completed(futures):
                callbacks.wait_if_paused()
                if not callbacks.should_run():
                    break
                chunk_keys, parts = fut.result()
                if parts:
                    for idx, key in enumerate(chunk_keys):
                        finalized = self._finalize(parts[idx].strip(), items[key])
                        if finalized is not None:
                            result[key] = finalized
                else:
                    for key in chunk_keys:
                        if not callbacks.should_run():
                            break
                        callbacks.wait_if_paused()
                        single = self._request(items[key].masked, api_code, timeout=5)
                        if single:
                            finalized = self._finalize(single, items[key])
                            if finalized is not None:
                                result[key] = finalized
                        # else: omit the key so service.py's non-caching
                        # fallback supplies the original instead of caching it.
                        time.sleep(0.3)
        return result
