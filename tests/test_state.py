"""Tests for mc_translator.runtime.state.JobState -- progress counters and
the ETA calculation that depends on them (see engines/service.py's
translate_dict, the single place these counters are incremented)."""
import time

from mc_translator.runtime.state import JobState


def test_increment_translated_cached_and_untranslated_are_independent():
    state = JobState()
    state.increment_translated(3)
    state.increment_cached(2)
    state.increment_untranslated(1)
    assert state.translated_strings == 3
    assert state.cached_strings == 2
    assert state.untranslated_strings == 1


def test_eta_text_before_any_progress_is_calculating():
    state = JobState()
    assert state.eta_text() == "расчёт..."
    state.start_time = time.time()
    assert state.eta_text() == "расчёт..."


def test_eta_text_uses_combined_translated_and_cached_for_the_rate():
    """Regression test: eta_text() used to divide only by translated_strings,
    completely ignoring cached_strings -- even though a cache hit is real,
    already-resolved work that shrinks how much is actually left. Ignoring
    it understated the resolution rate and could make "remaining" look
    stuck even while cache hits were rapidly closing out a run."""
    state = JobState()
    state.total_strings = 100
    state.start_time = time.time() - 10  # comfortably over the 5s floor
    state.translated_strings = 5
    state.cached_strings = 5
    text = state.eta_text()
    assert text not in ("расчёт...", "готово")


def test_eta_text_untranslated_strings_do_not_count_as_resolved():
    """A string that fell back to English is NOT done -- it must not make
    eta_text() think there's nothing left, or that any progress was made."""
    state = JobState()
    state.total_strings = 10
    state.start_time = time.time() - 10
    state.untranslated_strings = 10
    assert state.eta_text() == "расчёт..."  # translated+cached still 0


def test_eta_text_reports_done_once_resolved_covers_total():
    state = JobState()
    state.total_strings = 10
    state.start_time = time.time() - 10
    state.translated_strings = 6
    state.cached_strings = 4
    assert state.eta_text() == "готово"
