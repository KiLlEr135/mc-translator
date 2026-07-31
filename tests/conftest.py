"""Shared pytest fixtures for processor/engine tests -- a no-network fake
TranslationService and EngineCallbacks, so processor tests exercise the real
processing/orchestration logic without hitting Google/DeepL/OpenRouter/Kobold."""
import pytest

from mc_translator.engines.base import EngineCallbacks
from mc_translator.runtime.state import JobState


class FakeConfig:
    def getboolean(self, section, key, fallback=False):
        return fallback

    def getint(self, section, key, fallback=0):
        return fallback

    def get(self, section, key):
        return ""


class FakeService:
    """translate_dict() echoes each value with a "TR[...]" marker -- no
    network, no shield masking (tests that care about shield/escaping
    behavior exercise text_processing/snbt_extract directly instead)."""

    def __init__(self):
        self.config = FakeConfig()
        self.calls = []  # [(strings_dict, usage_label), ...] for assertions

    def translate_dict(self, strings, target_lang, callbacks, *, context="", usage_label=None):
        self.calls.append((dict(strings), usage_label))
        return {k: f"TR[{v}]" for k, v in strings.items()}


@pytest.fixture
def fake_service():
    return FakeService()


@pytest.fixture
def fake_callbacks():
    logs = []

    def on_log(msg, tag="white"):
        logs.append((msg, tag))

    cb = EngineCallbacks(
        should_run=lambda: True,
        wait_if_paused=lambda: None,
        on_log=on_log,
        on_status=lambda msg: None,
    )
    cb.logs = logs  # dataclass instances allow extra attrs; convenience for assertions
    return cb


@pytest.fixture
def job_state():
    state = JobState()
    state.is_running = True
    return state


@pytest.fixture
def lang():
    return {"file": "ru_ru", "api": "ru", "deepl": "RU", "name": "Russian", "regex": r"[А-Яа-яЁё]"}
