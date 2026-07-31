import threading
import time
from dataclasses import dataclass, field


@dataclass
class JobState:
    is_running: bool = False
    is_paused: bool = False
    total_strings: int = 0
    translated_strings: int = 0
    cached_strings: int = 0
    # Strings TranslationService.translate_dict() (engines/service.py)
    # couldn't translate this run and fell back to the untranslated English
    # original for -- e.g. the AI backend gave up after repeated failures.
    # Never subtracted from "remaining" in eta_text() below: these strings
    # still need translating, a later run (doперевод) will pick them up.
    untranslated_strings: int = 0
    start_time: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def wait_if_paused(self) -> None:
        while self.is_paused and self.is_running:
            time.sleep(0.5)

    def should_run(self) -> bool:
        with self._lock:
            return self.is_running

    def stop(self) -> None:
        with self._lock:
            self.is_running = False
            self.is_paused = False

    def try_start(self) -> bool:
        """Atomically transition from not-running to running, returning
        False (state left untouched) if it was already running. pywebview
        dispatches every js_api call (e.g. start_analysis/start_translation)
        on its own freshly-spawned thread with no built-in serialization, so
        a plain "if self.job_state.is_running: return" check followed by a
        separate "self.job_state.is_running = True" assignment is a
        check-then-act race: two near-simultaneous calls can both pass the
        check before either sets the flag, starting two concurrent jobs that
        share (and clobber) the same in-memory cache/usage-log state."""
        with self._lock:
            if self.is_running:
                return False
            self.is_running = True
            self.is_paused = False
            return True

    def increment_translated(self, count: int = 1) -> None:
        with self._lock:
            self.translated_strings += count

    def increment_cached(self, count: int = 1) -> None:
        with self._lock:
            self.cached_strings += count

    def increment_untranslated(self, count: int = 1) -> None:
        with self._lock:
            self.untranslated_strings += count

    def eta_text(self) -> str:
        # translated_strings and cached_strings together are the actual
        # "resolved" volume (fresh AI/engine translations plus cache hits);
        # untranslated_strings (fallback-to-English) is deliberately excluded
        # -- those strings are still pending, not done, so they must not
        # shrink "remaining" or count toward the rate.
        resolved = self.translated_strings + self.cached_strings
        if not self.start_time or resolved == 0:
            return "расчёт..."
        elapsed = time.time() - self.start_time
        if elapsed < 5:
            return "расчёт..."
        remaining = self.total_strings - resolved
        if remaining <= 0:
            return "готово"
        rate = resolved / elapsed
        seconds = remaining / rate
        if seconds < 60:
            return f"≈ {int(seconds)} сек"
        if seconds < 3600:
            return f"≈ {int(seconds // 60)} мин {int(seconds % 60)} сек"
        return f"≈ {int(seconds // 3600)} ч {int((seconds % 3600) // 60)} мин"
