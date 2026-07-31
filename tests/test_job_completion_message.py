"""Regression tests for runtime/job.py's run_translation() completion
message. Before this fix, "✅ ПЕРЕВОД УСПЕШНО ЗАВЕРШЕН!" was gated only by
total_failed (a count of processor-level EXCEPTIONS) -- a string that
silently fell back to the untranslated English original (engines/service.py's
translate_dict() fallback path, e.g. because the AI backend gave up after
repeated failures) never raises, so a run that left large portions of a
pack in English still reported full success with no indication anything
was wrong. run_translation() must now check state.untranslated_strings and
report "⚠ ПЕРЕВОД ЗАВЕРШЁН ЧАСТИЧНО" instead in that case."""
import json
import zipfile

from mc_translator.cache import TranslationCache
from mc_translator.engines.google import GoogleEngine
from mc_translator.runtime.job import TranslationJob, TranslationOptions
from mc_translator.runtime.state import JobState


class _FakeConfig:
    def getboolean(self, section, key, fallback=False):
        return fallback

    def getint(self, section, key, fallback=0):
        return fallback

    def get(self, section, key):
        return ""


def _make_jar(path):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("assets/testmod/lang/en_us.json", json.dumps({"key": "Hello World"}))


def _base_options(**overrides):
    defaults = dict(
        language_label="Русский",
        mc_version="1.21.1",
        output_mode="inplace",
        pack_name="pack",
        engine="google",
        google_mode="single",
        ai_mode="safe",
        ai_provider="local",
        global_context="",
        process_mode="force",
        translate_mods=True,
        translate_books=False,
        translate_quests=False,
    )
    defaults.update(overrides)
    return TranslationOptions(**defaults)


def _run_job(tmp_path, monkeypatch, fake_request):
    monkeypatch.setattr(GoogleEngine, "_request", fake_request)
    mods_dir = tmp_path / "mods"
    mods_dir.mkdir()
    _make_jar(mods_dir / "mod.jar")

    logs = []
    job = TranslationJob(
        config=_FakeConfig(),
        cache_std=TranslationCache(str(tmp_path / "cache.json")),
        cache_ai=TranslationCache(str(tmp_path / "cache_ai.json")),
        state=JobState(),
        on_log=lambda msg, tag="white": logs.append(msg),
        on_status=lambda msg, progress: None,
        on_row=lambda *a: None,
    )
    job.state.is_running = True
    job.run_translation(_base_options(mc_dir=str(tmp_path)))
    return job, logs


def test_full_success_reports_uspeshno_zavershen(tmp_path, monkeypatch):
    job, logs = _run_job(tmp_path, monkeypatch, lambda self, text, api_code, timeout=10: text.upper())

    assert any("УСПЕШНО ЗАВЕРШЕН" in msg for msg in logs)
    assert not any("ЧАСТИЧНО" in msg for msg in logs)
    assert job.state.untranslated_strings == 0


def test_engine_failure_reports_partial_completion_not_full_success(tmp_path, monkeypatch):
    job, logs = _run_job(tmp_path, monkeypatch, lambda self, text, api_code, timeout=10: None)

    assert any("ЧАСТИЧНО" in msg for msg in logs)
    assert not any("УСПЕШНО ЗАВЕРШЕН" in msg for msg in logs)
    assert job.state.untranslated_strings > 0
