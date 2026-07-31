"""Tests for mc_translator.gui.ai_setup_api.AiSetupMixin (mixed into
Api -- see gui/bridge.py) -- the "auto-setup local AI" JS-callable methods:
detect_ai_hardware() (probe + build a DownloadPlan) and download_ai_bundle()
(execute it). Cache files are redirected to tmp_path like test_bridge.py;
`settings` is replaced with an in-memory fake so no test ever touches the
real project's settings.ini, and threading.Thread is replaced with a
synchronous stand-in so download_ai_bundle()'s background work completes
before the JS-callable method's assertions run."""
import pytest

from mc_translator import cache as cache_module
from mc_translator.gui import ai_setup_api as ai_setup_api_module
from mc_translator.gui import bridge as bridge_module
from mc_translator.runtime import ai_catalog, ai_downloader
from mc_translator.utils.hardware_detect import GpuInfo


class _FakeSettings:
    def __init__(self):
        self.values = {}

    def set(self, section, key, value):
        self.values[(section, key)] = str(value)

    def get(self, section, key):
        return self.values.get((section, key), "")


class _SyncThread:
    """threading.Thread stand-in that runs its target synchronously on
    .start() -- so download_ai_bundle()'s background daemon thread completes
    before the test's assertions run, without any real thread/sleep."""

    def __init__(self, target=None, daemon=None, **kwargs):
        self._target = target

    def start(self):
        if self._target:
            self._target()


class _FakeStreamResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self._payload

    def close(self):
        pass


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "CACHE_FILE_STD", str(tmp_path / "cache.json"))
    monkeypatch.setattr(cache_module, "CACHE_FILE_AI", str(tmp_path / "ai_cache.json"))
    fake_settings = _FakeSettings()
    monkeypatch.setattr(ai_setup_api_module, "settings", fake_settings)
    monkeypatch.setattr(ai_setup_api_module, "app_path", lambda *parts: str(tmp_path.joinpath(*parts)))
    instance = bridge_module.Api()
    instance.settings = fake_settings  # convenience handle for assertions
    return instance


def _fake_release(asset_name: str, size: int) -> dict:
    return {
        "tag_name": "v1.117.1",
        "assets": [{"name": asset_name, "browser_download_url": f"http://example/{asset_name}", "size": size}],
    }


def test_detect_ai_hardware_refused_while_job_running(api):
    api.job_state.is_running = True
    result = api.detect_ai_hardware()
    assert result["ok"] is False


def test_detect_ai_hardware_refused_while_download_active(api):
    api._ai_download_active = True
    result = api.detect_ai_hardware()
    assert result["ok"] is False


def test_detect_ai_hardware_happy_path_matches_this_projects_proven_4gb_setup(api, monkeypatch):
    """Anchor case: a 4GB NVIDIA card (this project's own real, live-verified
    hardware) must resolve to the 3B tier and the CUDA koboldcpp build."""
    monkeypatch.setattr(ai_setup_api_module, "detect_gpu", lambda: GpuInfo("nvidia", "GTX 1650", 4096, "test"))
    monkeypatch.setattr(ai_downloader, "fetch_latest_release", lambda repo: _fake_release("koboldcpp.exe", 600_000_000))

    result = api.detect_ai_hardware()

    assert result["ok"] is True
    assert result["gpu"]["vendor"] == "nvidia"
    assert result["gpu"]["vramGb"] == 4.0
    assert result["model"]["label"] == "Qwen2.5-3B-Instruct"
    assert result["engine"]["name"] == "koboldcpp.exe"
    assert result["enoughDisk"] is True
    assert result["overwriteFiles"] == []
    assert api._pending_ai_plan is not None
    assert api._pending_ai_plan.model.key == "3b"
    assert api._pending_ai_plan.gpu_backend == "cuda"
    assert api._pending_ai_plan.gpu_layers == 99


def test_detect_ai_hardware_no_gpu_selects_smallest_model_and_nocuda_engine(api, monkeypatch):
    monkeypatch.setattr(ai_setup_api_module, "detect_gpu", lambda: GpuInfo("none", "", 0, "none"))
    monkeypatch.setattr(ai_downloader, "fetch_latest_release", lambda repo: _fake_release("koboldcpp-nocuda.exe", 100_000_000))

    result = api.detect_ai_hardware()

    assert result["ok"] is True
    assert result["model"]["label"] == "Qwen2.5-0.5B-Instruct"
    assert result["engine"]["name"] == "koboldcpp-nocuda.exe"
    assert api._pending_ai_plan.gpu_backend == "cpu"
    assert api._pending_ai_plan.gpu_layers == 0


def test_detect_ai_hardware_release_fetch_failure_returns_clean_error(api, monkeypatch):
    monkeypatch.setattr(ai_setup_api_module, "detect_gpu", lambda: GpuInfo("nvidia", "GTX 1650", 4096, "test"))

    def fail_fetch(repo):
        raise ai_downloader.DownloadError("network is down")

    monkeypatch.setattr(ai_downloader, "fetch_latest_release", fail_fetch)

    result = api.detect_ai_hardware()

    assert result["ok"] is False
    assert "network is down" in result["error"]
    assert api._pending_ai_plan is None


def test_download_ai_bundle_requires_a_pending_plan(api):
    result = api.download_ai_bundle()
    assert result["ok"] is False


def test_download_ai_bundle_refused_while_job_running(api, monkeypatch):
    monkeypatch.setattr(ai_setup_api_module, "detect_gpu", lambda: GpuInfo("nvidia", "GTX 1650", 4096, "test"))
    monkeypatch.setattr(ai_downloader, "fetch_latest_release", lambda repo: _fake_release("koboldcpp.exe", 10))
    api.detect_ai_hardware()
    api.job_state.is_running = True

    result = api.download_ai_bundle()

    assert result["ok"] is False


def test_download_ai_bundle_happy_path_downloads_both_and_saves_settings(api, monkeypatch, tmp_path):
    monkeypatch.setattr(ai_setup_api_module, "detect_gpu", lambda: GpuInfo("nvidia", "GTX 1650", 4096, "test"))
    monkeypatch.setattr(ai_downloader, "fetch_latest_release", lambda repo: _fake_release("koboldcpp.exe", 5))

    # Swap in a tiny ModelSpec so the fake download doesn't need to move
    # gigabytes of real bytes through the test.
    tiny_model = ai_catalog.ModelSpec(
        key="tiny", label="TinyTestModel", repo="test/repo", filename="tiny-model.gguf", size_bytes=7, min_total_vram_gb=0
    )
    monkeypatch.setattr(ai_catalog, "select_model", lambda gpu: tiny_model)
    monkeypatch.setattr(ai_setup_api_module.threading, "Thread", _SyncThread)

    def fake_get(url, stream, timeout):
        if url.endswith("koboldcpp.exe"):
            return _FakeStreamResponse(b"E" * 5)
        return _FakeStreamResponse(b"M" * 7)

    monkeypatch.setattr(ai_downloader.requests, "get", fake_get)

    detect_result = api.detect_ai_hardware()
    assert detect_result["ok"] is True

    result = api.download_ai_bundle()
    assert result["ok"] is True

    ai_dir = tmp_path / "AI"
    assert (ai_dir / "koboldcpp.exe").read_bytes() == b"E" * 5
    assert (ai_dir / "tiny-model.gguf").read_bytes() == b"M" * 7

    assert api.settings.get("AI", "exe_path") == str(ai_dir / "koboldcpp.exe")
    assert api.settings.get("AI", "model_path") == str(ai_dir / "tiny-model.gguf")
    assert api.settings.get("AI", "gpu_backend") == "cuda"
    assert api.settings.get("AI", "gpu_layers") == "99"
    assert api.settings.get("AI", "ai_provider") == "local"

    # State cleaned up after the run.
    assert api._ai_download_active is False
    assert api._pending_ai_plan is None


def test_download_ai_bundle_cancellation_leaves_no_partial_files(api, monkeypatch, tmp_path):
    monkeypatch.setattr(ai_setup_api_module, "detect_gpu", lambda: GpuInfo("nvidia", "GTX 1650", 4096, "test"))
    monkeypatch.setattr(ai_downloader, "fetch_latest_release", lambda repo: _fake_release("koboldcpp.exe", 5))
    tiny_model = ai_catalog.ModelSpec(
        key="tiny", label="TinyTestModel", repo="test/repo", filename="tiny-model.gguf", size_bytes=7, min_total_vram_gb=0
    )
    monkeypatch.setattr(ai_catalog, "select_model", lambda gpu: tiny_model)
    monkeypatch.setattr(ai_setup_api_module.threading, "Thread", _SyncThread)

    # Simulates the user clicking "cancel" right as the download starts --
    # cancel_ai_download() fires as a side effect of the very first network
    # call, so should_continue()'s first check (before any chunk is written)
    # already sees it cancelled. download_ai_bundle() itself resets
    # _ai_download_cancelled to False when it starts, so setting the flag
    # any earlier than this would be silently undone.
    def fake_get(url, stream, timeout):
        api.cancel_ai_download()
        return _FakeStreamResponse(b"E" * 5)

    monkeypatch.setattr(ai_downloader.requests, "get", fake_get)

    api.detect_ai_hardware()

    result = api.download_ai_bundle()
    assert result["ok"] is True  # the call itself succeeds -- cancellation is reported via on_log, not the return value

    ai_dir = tmp_path / "AI"
    assert not (ai_dir / "koboldcpp.exe").exists()
    assert not (ai_dir / "koboldcpp.exe.part").exists()
    assert api.settings.get("AI", "ai_provider") == ""  # never got as far as saving settings
