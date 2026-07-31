import json
import os
import threading

from mc_translator.constants import CACHE_FILE_AI, CACHE_FILE_STD
from mc_translator.text_processing import polish_translation
from mc_translator.utils.atomic_write import atomic_write


class TranslationCache:
    """Thread-safe translation cache with optional auto-save."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self._data: dict[str, str] = {}
        self._lock = threading.RLock()
        self._dirty = False
        self._entries_since_flush = 0
        self.polish_changes = self.load_and_polish()

    def load_and_polish(self) -> int:
        changes = 0
        with self._lock:
            if not os.path.exists(self.filepath):
                self._data = {}
                return 0
            try:
                with open(self.filepath, encoding="utf-8") as f:
                    loaded = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._data = {}
                return 0
            # Valid JSON but the wrong shape (a stray list/string/number --
            # a manual edit gone wrong, or a corrupted partial write) used
            # to crash here with an uncaught AttributeError on .items()
            # below, since only JSONDecodeError/OSError were guarded.
            self._data = loaded if isinstance(loaded, dict) else {}
            if not isinstance(loaded, dict):
                return 0

            for key, value in list(self._data.items()):
                polished = polish_translation(value)
                if polished != value:
                    self._data[key] = polished
                    changes += 1
            if changes:
                self._dirty = True
                self._flush_unlocked()
        return changes

    def make_key(self, api_code: str, source_text: str) -> str:
        return f"{api_code}_{source_text}"

    def get(self, api_code: str, source_text: str) -> str | None:
        key = self.make_key(api_code, source_text)
        with self._lock:
            return self._data.get(key)

    def set(self, api_code: str, source_text: str, translated: str) -> None:
        key = self.make_key(api_code, source_text)
        with self._lock:
            self._data[key] = translated
            self._dirty = True
            self._entries_since_flush += 1

    def discard(self, api_code: str, source_text: str) -> bool:
        """Remove one cached entry (used by the GUI's selective cache-clear —
        by mod or by file type — as opposed to clear(), which wipes everything).
        Returns True if an entry was actually removed."""
        key = self.make_key(api_code, source_text)
        with self._lock:
            if key in self._data:
                del self._data[key]
                self._dirty = True
                return True
        return False

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def save(self) -> None:
        with self._lock:
            if self._dirty:
                self._flush_unlocked()

    def save_if_threshold(self, every: int = 500) -> None:
        with self._lock:
            if self._dirty and self._entries_since_flush >= every:
                self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        payload = json.dumps(self._data, ensure_ascii=False, indent=2)
        atomic_write(self.filepath, payload.encode("utf-8"))
        self._dirty = False
        self._entries_since_flush = 0

    def clear(self) -> None:
        """Clear the cache data and mark as clean."""
        with self._lock:
            self._data = {}
            self._dirty = False
            self._entries_since_flush = 0


def load_both_caches() -> tuple[TranslationCache, TranslationCache, int]:
    std = TranslationCache(CACHE_FILE_STD)
    ai = TranslationCache(CACHE_FILE_AI)
    return std, ai, std.polish_changes + ai.polish_changes

