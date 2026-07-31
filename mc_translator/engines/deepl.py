from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from mc_translator.engines.base import EngineCallbacks, EngineItem, TranslationEngine
from mc_translator.text_processing import polish_translation, unmask_translation_strict


class DeepLEngine(TranslationEngine):
    def __init__(self, api_key: str, workers: int = 4) -> None:
        self.api_key = api_key.strip()
        self.workers = max(1, min(workers, 4))
        self.url = (
            "https://api-free.deepl.com/v2/translate"
            if self.api_key.endswith(":fx")
            else "https://api.deepl.com/v2/translate"
        )

    def translate_batch(
        self,
        items: dict[str, EngineItem],
        target_lang: dict,
        callbacks: EngineCallbacks,
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        keys = list(items.keys())
        chunks = [keys[i : i + 40] for i in range(0, len(keys), 40)]

        def translate_chunk(chunk_keys: list[str]) -> dict[str, str]:
            try:
                response = requests.post(
                    self.url,
                    headers={"Authorization": f"DeepL-Auth-Key {self.api_key}"},
                    json={
                        "text": [items[k].masked for k in chunk_keys],
                        "target_lang": target_lang["deepl"],
                    },
                    timeout=60,
                )
                response.raise_for_status()
                translations = response.json()["translations"]
                chunk_result: dict[str, str] = {}
                for idx, key in enumerate(chunk_keys):
                    raw = translations[idx]["text"]
                    raw = unmask_translation_strict(raw, items[key].mapping)
                    if raw is None:
                        # DeepL mangled or fully dropped a shield placeholder
                        # badly enough that unmask couldn't restore it --
                        # omit the key, same as an empty/failed translation.
                        continue
                    val = polish_translation(raw)
                    if val:
                        chunk_result[key] = val
                    # else: omit the key (matches Google/LLM engines' contract)
                    # so service.py's non-caching fallback supplies the
                    # original text instead of permanently caching an empty
                    # string as if it were a real translation.
                return chunk_result
            except (requests.RequestException, KeyError, IndexError) as exc:
                callbacks.on_log(f"❌ Ошибка DeepL: {exc}", "red")
                # Omit these keys rather than caching the untranslated original
                # (service.py's non-caching fallback supplies it for this run).
                return {}

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {}
            for chunk_keys in chunks:
                if not callbacks.should_run():
                    break
                callbacks.wait_if_paused()
                if not callbacks.should_run():
                    break
                futures[pool.submit(translate_chunk, chunk_keys)] = chunk_keys

            for fut in as_completed(futures):
                if not callbacks.should_run():
                    break
                result.update(fut.result())

        return result
