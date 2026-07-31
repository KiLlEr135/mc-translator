"""Job-level regression tests for two runtime/job.py robustness fixes:

- run_translation() used to call ai_launcher.ensure_running() unconditionally
  after the (stoppable) string-counting phase, with no should_run() guard --
  a Stop click landing during estimation could still spawn a fresh local
  koboldcpp.exe process afterward, with no way for the GUI's Stop button to
  reach and terminate it (see runtime/ai_launcher.py's own terminate() no-op
  when self.process is still None).
- The finally block's "always persist" guarantee (cache.save() /
  pack_writer.close()) used to be unguarded, so an OSError there (disk full,
  AV/OneDrive lock) propagated out of run_translation() entirely, skipping
  jar/estimate-manifest persistence and the final status/summary logging --
  unlike the sibling save_usage_log/save_jar_manifest/save_estimate_cache
  calls in the same block, which already protect themselves internally.
"""
import json
import zipfile

from mc_translator.runtime import job as job_module
from mc_translator.runtime.job import TranslationJob, TranslationOptions


class _FakeConfig:
    def getboolean(self, section, key, fallback=False):
        return fallback

    def getint(self, section, key, fallback=0):
        return fallback

    def get(self, section, key):
        if section == "AI" and key == "model_path":
            return "model.gguf"
        return ""


class _RaisingCache:
    def save(self):
        raise OSError("disk full")


class _NoOpJarProcessor:
    def __init__(self, service, state, callbacks):
        pass

    def process(self, path, **kwargs):
        pass


class _AssertNotCalledAiLauncher:
    def ensure_running(self, *a, **kw):
        raise AssertionError("ensure_running must not be called once should_run() is False")

    def terminate(self, on_log=None):
        pass


def _make_jar(path):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("assets/mod/lang/en_us.json", json.dumps({"key": "Hello World"}))


def _base_options(**overrides):
    defaults = dict(
        language_label="Русский",
        mc_version="1.21.1",
        output_mode="inplace",
        pack_name="pack",
        engine="ai",
        google_mode="single",
        ai_mode="safe",
        ai_provider="local",
        global_context="",
        process_mode="append",
        translate_mods=True,
        translate_books=False,
        translate_quests=False,
    )
    defaults.update(overrides)
    return TranslationOptions(**defaults)


def test_stopped_before_ai_launch_never_spawns_local_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module, "JarProcessor", _NoOpJarProcessor)
    mods_dir = tmp_path / "mods"
    mods_dir.mkdir()
    _make_jar(mods_dir / "mod.jar")

    job = TranslationJob(
        config=_FakeConfig(),
        cache_std=_RaisingCache(),
        cache_ai=_RaisingCache(),
        state=job_module.JobState(),
        on_log=lambda msg, tag="white": None,
        on_status=lambda msg, progress: None,
        on_row=lambda *a: None,
    )
    # Simulate Stop landing before the local-AI launch step: should_run()
    # is False, but nothing has raised/returned early yet (a non-empty
    # modpack is discovered, so the run reaches the ai_provider=="local"
    # branch).
    job.state.is_running = False
    job.ai_launcher = _AssertNotCalledAiLauncher()

    options = _base_options(mc_dir=str(tmp_path))
    job.run_translation(options)  # must not raise (proves ensure_running was skipped)


def test_finally_block_survives_cache_save_oserror(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module, "JarProcessor", _NoOpJarProcessor)
    mods_dir = tmp_path / "mods"
    mods_dir.mkdir()
    _make_jar(mods_dir / "mod.jar")

    logs = []
    job = TranslationJob(
        config=_FakeConfig(),
        cache_std=_RaisingCache(),
        cache_ai=_RaisingCache(),
        state=job_module.JobState(),
        on_log=lambda msg, tag="white": logs.append(msg),
        on_status=lambda msg, progress: None,
        on_row=lambda *a: None,
    )
    job.state.is_running = True

    options = _base_options(mc_dir=str(tmp_path), engine="google", ai_provider="local")
    # Must not raise: cache.save()'s OSError is caught, and execution must
    # reach the post-finally status/summary logging below it.
    job.run_translation(options)

    assert any("кэш" in msg.lower() for msg in logs)
    # The run must still reach its final status line after the guarded
    # OSError, not die silently inside the finally block.
    assert any("ОСТАНОВЛЕНО" in msg or "ЗАВЕРШЁН" in msg or "завершен" in msg.lower() for msg in logs)
