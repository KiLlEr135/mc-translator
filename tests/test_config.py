"""Tests for mc_translator.config.ConfigManager — settings.ini persistence,
including recovery from a corrupted file and safe fallback-returning getters.

Each test monkeypatches mc_translator.config.SETTINGS_FILE to an isolated
tmp_path location, since SETTINGS_FILE now resolves to a fixed path next to
the application (see mc_translator/utils/app_paths.py) rather than a bare
relative filename -- it must not touch the real project's settings.ini.
"""
import glob
import os

import pytest

from mc_translator.config import ConfigManager


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    path = str(tmp_path / "settings.ini")
    monkeypatch.setattr("mc_translator.config.SETTINGS_FILE", path)
    return path


def test_fresh_config_applies_defaults(isolated_settings):
    cfg = ConfigManager()
    assert cfg.get("GENERAL", "theme") == "Dark"
    assert cfg.getboolean("GENERAL", "smart_glue") is True
    assert cfg.get("AI", "gpu_backend") == "cuda"
    assert os.path.exists(isolated_settings)


def test_existing_settings_ini_without_gpu_backend_gets_it_injected(isolated_settings):
    """A settings.ini written before gpu_backend existed must gain the key
    (defaulted to "cuda", preserving today's --usecublas launch behavior)
    on next load, not crash or silently leave AiLauncher reading a missing
    option."""
    with open(isolated_settings, "w", encoding="utf-8") as f:
        f.write("[AI]\nexe_path = koboldcpp.exe\nmodel_path = model.gguf\n")

    cfg = ConfigManager()

    assert cfg.get("AI", "gpu_backend") == "cuda"
    # And it's actually persisted, not just defaulted in memory.
    cfg2 = ConfigManager()
    assert cfg2.get("AI", "gpu_backend") == "cuda"


def test_existing_settings_ini_without_auto_cycle_keys_gets_them_injected(isolated_settings):
    """Same backfill guarantee as gpu_backend above, for the two OPENROUTER
    keys added by the "auto-cycle free models" feature -- an install's
    settings.ini written before this feature existed must gain both keys
    (defaulted to off / empty, preserving today's single-model behavior) on
    next load, not crash or silently leave the GUI/engine reading missing
    options."""
    with open(isolated_settings, "w", encoding="utf-8") as f:
        f.write("[OPENROUTER]\napi_key = \nmodel = google/gemma-2-9b-it:free\n")

    cfg = ConfigManager()

    assert cfg.getboolean("OPENROUTER", "auto_cycle_free") is False
    assert cfg.get("OPENROUTER", "free_models") == ""
    cfg2 = ConfigManager()
    assert cfg2.getboolean("OPENROUTER", "auto_cycle_free") is False


def test_set_and_get_round_trip(isolated_settings):
    cfg = ConfigManager()
    cfg.set("GENERAL", "mc_dir", r"D:\games\Instances\Foo")
    # A fresh ConfigManager instance re-reading the same file must see it.
    cfg2 = ConfigManager()
    assert cfg2.get("GENERAL", "mc_dir") == r"D:\games\Instances\Foo"


def test_getboolean_returns_fallback_for_empty_value(isolated_settings):
    cfg = ConfigManager()
    cfg._config.set("SESSION", "translate_mods", "")
    assert cfg.getboolean("SESSION", "translate_mods", fallback=True) is True
    assert cfg.getboolean("SESSION", "translate_mods", fallback=False) is False


def test_getboolean_returns_fallback_for_missing_section(isolated_settings):
    cfg = ConfigManager()
    assert cfg.getboolean("NOPE", "nope", fallback=True) is True


def test_getint_returns_fallback_for_missing_option(isolated_settings):
    cfg = ConfigManager()
    assert cfg.getint("GENERAL", "does_not_exist", 42) == 42


def test_getint_returns_fallback_for_non_numeric_value(isolated_settings):
    cfg = ConfigManager()
    cfg.set("AI", "gpu_layers", "not-a-number")
    assert cfg.getint("AI", "gpu_layers", 99) == 99


def test_getint_parses_negative_values(isolated_settings):
    """Unlike the old str.isdigit()-based check (which silently coerced any
    negative number to the fallback), a real negative value must parse."""
    cfg = ConfigManager()
    cfg.set("AI", "gpu_layers", "-1")
    assert cfg.getint("AI", "gpu_layers", 99) == -1


def test_corrupted_settings_ini_recovers_with_backup(isolated_settings):
    with open(isolated_settings, "w", encoding="utf-8") as f:
        f.write("this is not valid ini [[[content")

    cfg = ConfigManager()
    # Recovered to defaults instead of crashing.
    assert cfg.get("GENERAL", "theme") == "Dark"

    backups = glob.glob(isolated_settings + ".bak-corrupt-*")
    assert len(backups) == 1
    with open(backups[0], encoding="utf-8") as f:
        assert "this is not valid ini" in f.read()
