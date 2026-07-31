"""Tests for mc_translator.runtime.state.JobState.try_start() -- closes the
check-then-act race between bridge.py's start_analysis/start_translation and
a near-simultaneous second call: pywebview dispatches every js_api call on
its own freshly-spawned thread with no built-in serialization, so a plain
"if is_running: return" followed by a separate "is_running = True" could let
two calls both pass the check before either set the flag."""
import threading

from mc_translator.runtime.state import JobState


def test_try_start_returns_true_and_sets_running_when_idle():
    state = JobState()
    assert state.try_start() is True
    assert state.is_running is True
    assert state.is_paused is False


def test_try_start_returns_false_when_already_running():
    state = JobState()
    state.is_running = True
    assert state.try_start() is False


def test_try_start_is_atomic_under_concurrent_calls():
    """Fire many threads at try_start() simultaneously -- exactly one must
    win, matching the real start_analysis/start_translation race scenario."""
    state = JobState()
    results = []
    lock = threading.Lock()
    barrier = threading.Barrier(20)

    def attempt():
        barrier.wait()
        ok = state.try_start()
        with lock:
            results.append(ok)

    threads = [threading.Thread(target=attempt) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(True) == 1
    assert results.count(False) == 19
