"""Tests for StringEstimator's optional estimate_cache -- a per-jar count
cache (see mc_translator.runtime.manifest.load_estimate_cache's docstring).
A jar's translatable-string count is a pure function of its own bytes plus
(mode, translate_mods, translate_books) -- _count_lang compares the English
source against the SAME jar's bundled translation, never against external
pack/sidecar state -- so a cache hit must return exactly what a fresh
re-parse would, without ever reopening the zip."""
import json
import zipfile

from mc_translator.processors import estimator as estimator_module
from mc_translator.processors.estimator import StringEstimator


def _make_jar(path, entries):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, json.dumps(data) if isinstance(data, dict) else data)


def test_estimate_jar_cache_hit_skips_reparsing_and_matches_fresh_count(
    tmp_path, job_state, lang, monkeypatch
):
    jar_path = tmp_path / "mod.jar"
    _make_jar(jar_path, {"assets/mod/lang/en_us.json": {"key": "Hello World"}})

    est = StringEstimator(job_state)
    cache: dict = {}
    count1 = est._estimate_jar(
        str(jar_path), f"{lang['file']}.json", lang, "append", True, True, False,
        estimate_cache=cache,
    )
    assert count1 == 1
    assert str(jar_path) in cache
    assert cache[str(jar_path)]["count"] == 1

    calls = []
    real_zipfile_cls = estimator_module.zipfile.ZipFile

    def spy(*args, **kwargs):
        calls.append(args)
        return real_zipfile_cls(*args, **kwargs)

    monkeypatch.setattr(estimator_module.zipfile, "ZipFile", spy)

    count2 = est._estimate_jar(
        str(jar_path), f"{lang['file']}.json", lang, "append", True, True, False,
        estimate_cache=cache,
    )
    assert count2 == count1
    assert calls == []  # cache hit -- the zip was never reopened


def test_estimate_jar_cache_invalidated_when_jar_content_changes(tmp_path, job_state, lang):
    jar_path = tmp_path / "mod.jar"
    _make_jar(jar_path, {"assets/mod/lang/en_us.json": {"key": "Hello World"}})

    est = StringEstimator(job_state)
    cache: dict = {}
    count1 = est._estimate_jar(
        str(jar_path), f"{lang['file']}.json", lang, "append", True, True, False,
        estimate_cache=cache,
    )
    assert count1 == 1

    _make_jar(jar_path, {
        "assets/mod/lang/en_us.json": {"key": "Hello World", "key2": "Another string"},
    })

    count2 = est._estimate_jar(
        str(jar_path), f"{lang['file']}.json", lang, "append", True, True, False,
        estimate_cache=cache,
    )
    assert count2 == 2


def test_estimate_jar_cache_invalidated_by_mode_change(tmp_path, job_state, lang):
    jar_path = tmp_path / "mod.jar"
    _make_jar(jar_path, {
        "assets/mod/lang/en_us.json": {"key": "Hello World"},
        "assets/mod/lang/ru_ru.json": {"key": "Привет мир"},
    })

    est = StringEstimator(job_state)
    cache: dict = {}
    count_append = est._estimate_jar(
        str(jar_path), f"{lang['file']}.json", lang, "append", True, True, False,
        estimate_cache=cache,
    )
    assert count_append == 0  # already translated inside the jar -- append mode counts nothing

    count_force = est._estimate_jar(
        str(jar_path), f"{lang['file']}.json", lang, "force", True, True, False,
        estimate_cache=cache,
    )
    assert count_force == 1  # different mode -- cache entry for "append" must not be reused


def test_estimate_without_cache_arg_behaves_as_before(tmp_path, job_state, lang):
    """estimate_cache defaults to None -- callers that don't pass it (e.g.
    the Analyze screen's estimator usage, if any) keep working unchanged."""
    jar_path = tmp_path / "mod.jar"
    _make_jar(jar_path, {"assets/mod/lang/en_us.json": {"key": "Hello World"}})

    est = StringEstimator(job_state)
    total = est.estimate(
        [str(jar_path)], [], [],
        target_lang=lang, mode="append", translate_mods=True, translate_books=True,
        translate_quests=False, smart_glue=False,
    )
    assert total == 1
