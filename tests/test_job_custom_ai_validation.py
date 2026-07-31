"""Regression test for runtime/job.py: ai_provider='custom' must validate
against CUSTOM_AI/base_url + model, not silently fall through to the
'local' KoboldCPP branch's AI/model_path (.gguf) check -- a file the
custom-API path has no use for. Mirrors test_job_finally_robustness.py's
_FakeConfig/_base_options convention."""
from mc_translator.runtime.job import TranslationJob, TranslationOptions
from mc_translator.runtime.state import JobState


class _FakeConfig:
    def __init__(self, base_url="", model=""):
        self._base_url = base_url
        self._model = model

    def getboolean(self, section, key, fallback=False):
        return fallback

    def getint(self, section, key, fallback=0):
        return fallback

    def get(self, section, key):
        if section == "CUSTOM_AI" and key == "base_url":
            return self._base_url
        if section == "CUSTOM_AI" and key == "model":
            return self._model
        return ""


def _options(**overrides):
    defaults = dict(
        mc_dir="",
        language_label="Русский",
        mc_version="1.21.1",
        output_mode="inplace",
        pack_name="pack",
        engine="ai",
        google_mode="single",
        ai_mode="safe",
        ai_provider="custom",
        global_context="",
        process_mode="append",
        translate_mods=True,
        translate_books=False,
        translate_quests=False,
    )
    defaults.update(overrides)
    return TranslationOptions(**defaults)


def _run(config):
    logs = []
    job = TranslationJob(
        config=config,
        cache_std=None,
        cache_ai=None,
        state=JobState(),
        on_log=lambda msg, tag="white": logs.append(msg),
        on_status=lambda msg, progress=None: None,
        on_row=lambda *a: None,
    )
    job.run_translation(_options())
    return logs


def test_missing_base_url_blocks_run_with_custom_ai_error():
    logs = _run(_FakeConfig(base_url="", model="gpt-4o-mini"))
    assert any("URL эндпоинта" in msg for msg in logs)
    assert not any("gguf" in msg for msg in logs)


def test_missing_model_blocks_run_with_custom_ai_error():
    logs = _run(_FakeConfig(base_url="https://api.example.com/v1/chat/completions", model=""))
    assert any("ID модели" in msg for msg in logs)
    assert not any("gguf" in msg for msg in logs)
