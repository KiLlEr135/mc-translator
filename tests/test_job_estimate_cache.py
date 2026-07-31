"""Job-level integration test: run_translation must load StringEstimator's
per-jar count cache before the "Подсчёт строк" pass and persist it
afterward (pruned to jars still present), so a later run over an unchanged
jar can skip re-parsing it entirely. The actual cache-hit/skip-reparse
mechanics are covered directly in tests/test_estimator_cache.py; this test
only exercises job.py's own wiring (load -> pass through -> prune -> save)
against a real jar on disk, following the pattern of
test_job_run_translation_mc_version.py."""
import json
import os
import zipfile

from mc_translator.runtime import job as job_module
from mc_translator.runtime.job import TranslationJob, TranslationOptions
from mc_translator.runtime.manifest import load_estimate_cache


class _FakeConfig:
    def getboolean(self, section, key, fallback=False):
        return fallback

    def getint(self, section, key, fallback=0):
        return fallback

    def get(self, section, key):
        return ""


class _FakeCache:
    def save(self):
        pass


class _NoOpJarProcessor:
    """Stand-in for JarProcessor -- avoids any real translation/network
    activity. job.py's own estimate-cache wiring is what's under test here,
    not JarProcessor's translation behavior (covered elsewhere)."""

    def __init__(self, service, state, callbacks):
        pass

    def process(self, path, **kwargs):
        pass


def _make_jar(path):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("assets/mod/lang/en_us.json", json.dumps({"key": "Hello World"}))


def _run(tmp_path):
    job = TranslationJob(
        config=_FakeConfig(),
        cache_std=_FakeCache(),
        cache_ai=_FakeCache(),
        state=job_module.JobState(),
        on_log=lambda msg, tag="white": None,
        on_status=lambda msg, progress: None,
        on_row=lambda *a: None,
    )
    job.state.is_running = True
    options = TranslationOptions(
        mc_dir=str(tmp_path),
        language_label="Русский",
        mc_version="1.21.1",
        output_mode="inplace",
        pack_name="pack",
        engine="google",
        google_mode="single",
        ai_mode="safe",
        ai_provider="local",
        global_context="",
        process_mode="append",
        translate_mods=True,
        translate_books=False,
        translate_quests=False,
    )
    job.run_translation(options)


def test_run_translation_persists_estimate_cache_for_jars(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module, "JarProcessor", _NoOpJarProcessor)
    mods_dir = tmp_path / "mods"
    mods_dir.mkdir()
    jar_path = mods_dir / "mod.jar"
    _make_jar(jar_path)

    _run(tmp_path)

    cache = load_estimate_cache(str(tmp_path), "ru_ru")
    assert str(jar_path) in cache
    assert cache[str(jar_path)]["count"] == 1


def test_run_translation_prunes_estimate_cache_for_removed_jars(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module, "JarProcessor", _NoOpJarProcessor)
    mods_dir = tmp_path / "mods"
    mods_dir.mkdir()
    jar_path = mods_dir / "mod.jar"
    _make_jar(jar_path)
    _run(tmp_path)
    assert str(jar_path) in load_estimate_cache(str(tmp_path), "ru_ru")

    os.remove(jar_path)
    # A second jar keeps the run from hitting the early "Нечего переводить"
    # return, so we actually reach the finally block's pruning logic.
    jar_path2 = mods_dir / "mod2.jar"
    _make_jar(jar_path2)
    _run(tmp_path)

    cache = load_estimate_cache(str(tmp_path), "ru_ru")
    assert str(jar_path) not in cache
    assert str(jar_path2) in cache
