"""Regression test: TranslationJob.run_translation must thread
options.mc_version through to LangProcessor.process for the lang_files
bucket (LangProcessor uses it to pick legacy ru_RU vs modern ru_ru casing
via is_legacy_version/legacy_lang_code) -- a previous version of this call
was missing the mc_version kwarg entirely, silently falling back to
LangProcessor.process's default of "1.12.2" regardless of the modpack's
actual mc_version.

Everything else discover_* returns is monkeypatched to empty so only the
lang_files bucket is exercised, and LangProcessor itself is swapped for a
recording stand-in -- this keeps the test focused on job.py's own wiring
rather than re-testing LangProcessor's internals (covered elsewhere).

The discover_* functions are called from runtime/translation_pipeline.py
(job.py delegates file discovery there -- see discover_translation_files),
so they're monkeypatched on that module, not on job_module."""
from mc_translator.runtime import job as job_module
from mc_translator.runtime import translation_pipeline as pipeline_module
from mc_translator.runtime.job import TranslationJob, TranslationOptions


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


class _RecordingLangProcessor:
    """Stand-in for LangProcessor that just records the kwargs process() was
    called with, so the test can assert mc_version made it through."""

    instances: list["_RecordingLangProcessor"] = []

    def __init__(self, service, state, callbacks):
        self.service = service
        self.state = state
        self.callbacks = callbacks
        self.calls = []
        _RecordingLangProcessor.instances.append(self)

    def process(self, path, **kwargs):
        self.calls.append((path, kwargs))


def test_run_translation_passes_mc_version_to_lang_processor(tmp_path, monkeypatch):
    _RecordingLangProcessor.instances = []

    lang_file = tmp_path / "en_us.lang"
    lang_file.write_text("item.foo.name=Hello World\n", encoding="utf-8")

    monkeypatch.setattr(job_module, "LangProcessor", _RecordingLangProcessor)
    monkeypatch.setattr(pipeline_module, "discover_jar_files", lambda *a, **k: [])
    monkeypatch.setattr(pipeline_module, "discover_loose_lang_files", lambda *a, **k: [])
    monkeypatch.setattr(pipeline_module, "discover_snbt_files", lambda *a, **k: [])
    monkeypatch.setattr(pipeline_module, "discover_mcfunction_files", lambda *a, **k: [])
    monkeypatch.setattr(
        pipeline_module,
        "discover_mod_content_files",
        lambda mc_dir, **kwargs: {"lang": [str(lang_file)]},
    )

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
        mc_version="1.16.5",
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

    assert len(_RecordingLangProcessor.instances) == 1
    proc = _RecordingLangProcessor.instances[0]
    assert len(proc.calls) == 1
    path, kwargs = proc.calls[0]
    assert path == str(lang_file)
    assert kwargs.get("mc_version") == "1.16.5"
