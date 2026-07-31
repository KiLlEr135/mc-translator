"""Tests for mc_translator.gui.bridge.Api -- specifically clear_cache, the
GUI's cache-wipe operation. Cache files are redirected to tmp_path so tests
never touch the real project's cache.json/ai_cache.json."""
import os

import pytest

from mc_translator import cache as cache_module
from mc_translator.gui import bridge as bridge_module
from mc_translator.gui import settings_api as settings_api_module


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "CACHE_FILE_STD", str(tmp_path / "cache.json"))
    monkeypatch.setattr(cache_module, "CACHE_FILE_AI", str(tmp_path / "ai_cache.json"))
    return bridge_module.Api()


def test_clear_cache_all_mode_wipes_both_caches_on_successful_backup(api, tmp_path):
    api.cache_std.set("ru", "hello", "привет")
    api.cache_ai.set("ru", "world", "мир")
    api.cache_std.save()
    api.cache_ai.save()

    result = api.clear_cache({"mode": "all"})

    assert result["ok"] is True
    assert result["errors"] == []
    assert len(api.cache_std) == 0
    assert len(api.cache_ai) == 0
    # Original content preserved in a .bak-* file, not lost.
    backups = list(tmp_path.glob("cache.json.bak-*"))
    assert len(backups) == 1


def test_clear_cache_all_mode_does_not_wipe_a_cache_whose_backup_failed(api, tmp_path, monkeypatch):
    """Regression test: clear_cache(mode="all") used to call clear() on
    BOTH caches unconditionally, even if os.replace() (the backup step)
    raised for one of them. That silently discarded the failed-backup
    cache's in-memory data with no backup AND no error surfaced loudly
    enough to stop it -- the next save() would then persist an empty file,
    permanently losing every previously paid-for translation in that
    cache. Only a cache whose backup actually succeeded (or that had no
    file to begin with) may be cleared."""
    api.cache_std.set("ru", "hello", "привет")
    api.cache_ai.set("ru", "world", "мир")
    api.cache_std.save()
    api.cache_ai.save()

    std_path = api.cache_std.filepath
    real_replace = os.replace

    def failing_replace(src, dst):
        if src == std_path:
            raise OSError("simulated backup failure")
        return real_replace(src, dst)

    monkeypatch.setattr(bridge_module.os, "replace", failing_replace)

    result = api.clear_cache({"mode": "all"})

    assert result["ok"] is False
    assert result["errors"]
    # std cache's backup failed -- its in-memory data (and on-disk file)
    # must survive untouched.
    assert len(api.cache_std) == 1
    assert api.cache_std.get("ru", "hello") == "привет"
    assert os.path.exists(std_path)
    # ai cache's backup succeeded -- it's cleared as normal.
    assert len(api.cache_ai) == 0


def test_clear_cache_all_mode_refused_while_job_running(api):
    api.job_state.is_running = True
    result = api.clear_cache({"mode": "all"})
    assert result["ok"] is False
    assert len(api.cache_std) == 0  # untouched (nothing was ever set, but no crash)


def test_scoped_clear_cache_does_not_persist_when_backup_fails(api, tmp_path, monkeypatch):
    """Regression test: the mode="mods"/"kinds" scoped clear used to call
    cache.save() unconditionally even when its own backup (shutil.copy2)
    failed -- unlike the mode="all" branch, which skips the destructive
    write on a failed backup. That silently persisted the already-discarded
    (in-memory) entries over the on-disk cache with no recovery copy."""
    api.cache_std.set("ru", "hello", "привет")
    api.cache_std.save()

    from mc_translator.usage_log import save_usage_log

    save_usage_log(str(tmp_path), "ru_ru", {
        "hello": {"mods": ["TestMod"], "kind": "Интерфейс", "engine": "google"}
    })
    monkeypatch.setattr(bridge_module.settings, "get", lambda section, key: str(tmp_path) if key == "mc_dir" else "")

    std_path = api.cache_std.filepath

    import mc_translator.gui.cache_ops as cache_ops_module
    real_copy2 = cache_ops_module.shutil.copy2

    def failing_copy2(src, dst):
        if src == std_path:
            raise OSError("simulated backup failure")
        return real_copy2(src, dst)

    monkeypatch.setattr(cache_ops_module.shutil, "copy2", failing_copy2)

    with open(std_path, encoding="utf-8") as f:
        on_disk_before = f.read()

    result = api.clear_cache({"mode": "mods", "values": ["TestMod"], "languages": "all"})

    assert result["ok"] is False
    assert result["errors"]
    # The on-disk cache must be untouched -- no save() after the failed backup.
    with open(std_path, encoding="utf-8") as f:
        assert f.read() == on_disk_before


def test_save_settings_refused_while_job_running(api, monkeypatch):
    class _SpySettings:
        def set(self, *a, **kw):
            raise AssertionError("settings.set must not be called while a job is running")

    monkeypatch.setattr(settings_api_module, "settings", _SpySettings())
    api.job_state.is_running = True

    result = api.save_settings({"aiExePath": "koboldcpp.exe"})

    assert result["ok"] is False


# ---------------------------------------------------------------------
# Update check (GitHub releases)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "current,latest,expected",
    [
        ("9.1.0", "9.2.0", True),
        ("9.1.0", "10.0.0", True),
        ("9.9.0", "9.10.0", True),  # numeric, not lexicographic, comparison
        ("9.1.0", "9.1.0", False),
        ("9.1.0", "9.0.5", False),
        ("v9.1.0", "v9.1.1", True),  # a leading "v" on either side is fine
    ],
)
def test_is_newer_version(current, latest, expected):
    assert bridge_module._is_newer_version(current, latest) is expected


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_check_latest_release_returns_version_when_newer(monkeypatch):
    monkeypatch.setattr(
        bridge_module.requests, "get", lambda url, timeout=5: _FakeResponse({"tag_name": "v9.5.0"})
    )
    assert bridge_module._check_latest_release("owner/repo", "9.1.0") == "9.5.0"


def test_check_latest_release_returns_none_when_not_newer(monkeypatch):
    monkeypatch.setattr(
        bridge_module.requests, "get", lambda url, timeout=5: _FakeResponse({"tag_name": "v9.1.0"})
    )
    assert bridge_module._check_latest_release("owner/repo", "9.1.0") is None


def test_check_latest_release_silently_returns_none_on_network_failure(monkeypatch):
    def raise_error(url, timeout=5):
        raise ConnectionError("offline")

    monkeypatch.setattr(bridge_module.requests, "get", raise_error)
    assert bridge_module._check_latest_release("owner/repo", "9.1.0") is None


def test_check_latest_release_returns_none_on_malformed_response(monkeypatch):
    monkeypatch.setattr(bridge_module.requests, "get", lambda url, timeout=5: _FakeResponse({}))
    assert bridge_module._check_latest_release("owner/repo", "9.1.0") is None


def test_check_updates_logs_notice_when_newer_version_available(api, monkeypatch):
    class _ImmediateThread:
        def __init__(self, target, daemon=None):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(bridge_module.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(bridge_module, "_check_latest_release", lambda repo, current: "9.9.0")
    logged = []
    monkeypatch.setattr(api, "on_log", lambda message, tag="white": logged.append(message))

    api.check_updates()

    assert any("9.9.0" in message for message in logged)
    assert any(bridge_module.UPDATE_REPO in message for message in logged)


def test_check_updates_logs_nothing_when_up_to_date(api, monkeypatch):
    class _ImmediateThread:
        def __init__(self, target, daemon=None):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(bridge_module.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(bridge_module, "_check_latest_release", lambda repo, current: None)
    logged = []
    monkeypatch.setattr(api, "on_log", lambda message, tag="white": logged.append(message))

    api.check_updates()

    assert logged == []
