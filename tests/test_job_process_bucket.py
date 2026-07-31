"""Tests for TranslationJob._process_bucket -- the concurrent/sequential
per-file-type loop runner used by run_translation for all ~14 file buckets.
Covers: sequential/parallel equivalence, per-item exception isolation (one
bad file must not abort the rest of the bucket, unlike the previous
behavior where any uncaught exception jumped straight to run_translation's
outer except and skipped every remaining bucket), on_item_done semantics
(only fires on success, and only if Stop wasn't clicked mid-item), and that
the parallel path genuinely overlaps execution rather than just not
crashing."""
import threading
import time

from mc_translator.runtime.job import TranslationJob


def make_job(job_state):
    logs = []
    statuses = []
    job = TranslationJob(
        config=object(),  # unused by _process_bucket / AiLauncher.__init__
        cache_std=None,
        cache_ai=None,
        state=job_state,
        on_log=lambda msg, tag="white": logs.append((msg, tag)),
        on_status=lambda msg, progress: statuses.append((msg, progress)),
        on_row=lambda *a: None,
    )
    return job, logs, statuses


def test_empty_items_returns_immediately(job_state):
    job, logs, statuses = make_job(job_state)
    done, failed = job._process_bucket(
        [], lambda p: None, label="X", done_start=5, total_items=10, parallel=False
    )
    assert (done, failed) == (5, 0)
    assert statuses == []


def test_sequential_processes_all_items(job_state):
    job, logs, statuses = make_job(job_state)
    seen = []
    done, failed = job._process_bucket(
        ["a", "b", "c"], lambda p: seen.append(p),
        label="X", done_start=0, total_items=3, parallel=False,
    )
    assert seen == ["a", "b", "c"]
    assert (done, failed) == (3, 0)


def test_sequential_isolates_per_item_exception(job_state):
    job, logs, statuses = make_job(job_state)

    def worker(path):
        if path == "bad":
            raise RuntimeError("boom")

    done, failed = job._process_bucket(
        ["a", "bad", "c"], worker, label="X", done_start=0, total_items=3, parallel=False,
    )
    assert done == 3  # all three attempted/counted
    assert failed == 1
    assert any("Ошибка обработки" in msg for msg, tag in logs)


def test_on_item_done_only_fires_on_success(job_state):
    job, logs, statuses = make_job(job_state)
    succeeded = []

    def worker(path):
        if path == "bad":
            raise RuntimeError("boom")

    job._process_bucket(
        ["a", "bad", "c"], worker, label="X", done_start=0, total_items=3,
        parallel=False, on_item_done=lambda p: succeeded.append(p),
    )
    assert succeeded == ["a", "c"]


def test_on_item_done_does_not_fire_if_stopped_mid_item(job_state):
    """Matches the original jar-loop semantics: a processor can complete
    normally (no exception) after Stop was clicked mid-way through its own
    internal work -- that item must not be recorded as "fully done" (e.g.
    the jar manifest must not mark a Stop-interrupted jar as safe to skip
    next run)."""
    job, logs, statuses = make_job(job_state)
    succeeded = []

    def worker(path):
        if path == "b":
            job_state.is_running = False  # simulate Stop clicked mid-item

    job._process_bucket(
        ["a", "b", "c"], worker, label="X", done_start=0, total_items=3,
        parallel=False, on_item_done=lambda p: succeeded.append(p),
    )
    assert "b" not in succeeded
    assert "a" in succeeded


def test_parallel_path_actually_overlaps_execution(job_state):
    job, logs, statuses = make_job(job_state)
    active = {"count": 0, "max": 0}
    lock = threading.Lock()

    def worker(path):
        with lock:
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
        time.sleep(0.05)
        with lock:
            active["count"] -= 1

    items = [f"file{i}" for i in range(9)]
    done, failed = job._process_bucket(
        items, worker, label="X", done_start=0, total_items=9, parallel=True,
    )
    assert (done, failed) == (9, 0)
    assert active["max"] > 1, "parallel=True never actually ran items concurrently"


def test_parallel_path_isolates_per_item_exception(job_state):
    job, logs, statuses = make_job(job_state)

    def worker(path):
        if path == "file3":
            raise ValueError("boom")

    items = [f"file{i}" for i in range(6)]
    done, failed = job._process_bucket(
        items, worker, label="X", done_start=0, total_items=6, parallel=True,
    )
    assert done == 6
    assert failed == 1
    assert any("Ошибка обработки" in msg for msg, tag in logs)


def test_single_item_uses_sequential_path_even_when_parallel_true(job_state):
    """len(items) <= 1 always takes the sequential branch -- no point
    spinning up a thread pool for a single file."""
    job, logs, statuses = make_job(job_state)
    seen = []
    done, failed = job._process_bucket(
        ["only"], lambda p: seen.append(p), label="X", done_start=0, total_items=1, parallel=True,
    )
    assert seen == ["only"]
    assert (done, failed) == (1, 0)


def test_stopping_before_bucket_starts_skips_all_items(job_state):
    job_state.is_running = False
    job, logs, statuses = make_job(job_state)
    seen = []
    done, failed = job._process_bucket(
        ["a", "b"], lambda p: seen.append(p), label="X", done_start=0, total_items=2, parallel=False,
    )
    assert seen == []
    assert (done, failed) == (0, 0)
