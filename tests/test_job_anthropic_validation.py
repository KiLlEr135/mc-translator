"""Regression test for runtime/job.py: ai_provider='anthropic' must validate
against ANTHROPIC/api_key + model, not silently fall through to the
'local' KoboldCPP branch's AI/model_path (.gguf) check. Mirrors
test_job_custom_ai_validation.py's convention for the 'custom' provider."""
from mc_translator.runtime.job import TranslationJob, TranslationOptions
from mc_translator.runtime.state import JobState


class _FakeConfig:
    def __init__(self, api_key="", model=""):
        self._api_key = api_key
        self._model = model

    def getboolean(self, section, key, fallback=False):
        return fallback

    def getint(self, section, key, fallback=0):
        return fallback

    def get(self, section, key):
        if section == "ANTHROPIC" and key == "api_key":
            return self._api_key
        if section == "ANTHROPIC" and key == "model":
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
        ai_provider="anthropic",
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


def test_missing_api_key_blocks_run_with_anthropic_error():
    logs = _run(_FakeConfig(api_key="", model="claude-sonnet-4-5-20250929"))
    assert any("API-ключ Claude" in msg for msg in logs)
    assert not any("gguf" in msg for msg in logs)


def test_missing_model_blocks_run_with_anthropic_error():
    logs = _run(_FakeConfig(api_key="sk-ant-test", model=""))
    assert any("ID модели Claude" in msg for msg in logs)
    assert not any("gguf" in msg for msg in logs)
