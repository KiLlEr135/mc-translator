"""Tests for mc_translator.runtime.ai_launcher.AiLauncher -- terminate()
(regression coverage for a real bug where a stopped run left koboldcpp.exe
running and holding VRAM because the Popen-level terminate() silently
no-opped with no visibility into the failure) and ensure_running()'s
gpu_backend-gated launch flags (added for the "auto-setup local AI" feature,
which can download a koboldcpp-nocuda.exe build with no CUDA support -- it
must never be launched with --usecublas)."""
from mc_translator.runtime.ai_launcher import AiLauncher


class FakeConfig:
    def __init__(self, values=None):
        self._values = values or {}

    def get(self, section, key):
        return self._values.get((section, key), "")


class FakeProcess:
    def __init__(self, survives: bool) -> None:
        self.survives = survives
        self.terminate_called = False

    def terminate(self):
        self.terminate_called = True

    def wait(self, timeout=None):
        pass


def test_terminate_is_quiet_when_process_actually_exits(monkeypatch):
    launcher = AiLauncher(FakeConfig())
    launcher.process = FakeProcess(survives=False)
    monkeypatch.setattr(AiLauncher, "is_alive", lambda self: False)

    logs = []
    launcher.terminate(on_log=lambda msg, tag="white": logs.append(msg))

    assert launcher.process is None
    assert logs == []


def test_terminate_warns_when_server_survives_termination(monkeypatch):
    """If the tracked process handle didn't actually kill the real server
    (observed on this codebase's real hardware), the user must be told --
    otherwise the next run silently reuses a stale, VRAM-holding instance."""
    launcher = AiLauncher(FakeConfig())
    launcher.process = FakeProcess(survives=True)
    monkeypatch.setattr(AiLauncher, "is_alive", lambda self: True)

    logs = []
    launcher.terminate(on_log=lambda msg, tag="white": logs.append(msg))

    assert launcher.process is None
    assert any("не отвечает" not in msg and "видеопамять" in msg for msg in logs)


def test_terminate_without_on_log_does_not_raise(monkeypatch):
    launcher = AiLauncher(FakeConfig())
    launcher.process = FakeProcess(survives=True)
    monkeypatch.setattr(AiLauncher, "is_alive", lambda self: True)

    launcher.terminate()  # must not raise even though the server "survived"

    assert launcher.process is None


# ---------------------------------------------------------------------
# ensure_running() -- gpu_backend-gated launch flags
# ---------------------------------------------------------------------


def _run_ensure_running(monkeypatch, gpu_backend_value, vulkan_device_index="unset"):
    values = {
        ("AI", "exe_path"): "koboldcpp.exe",
        ("AI", "model_path"): "model.gguf",
        ("AI", "gpu_layers"): "99",
    }
    if gpu_backend_value is not None:
        values[("AI", "gpu_backend")] = gpu_backend_value
    launcher = AiLauncher(FakeConfig(values))

    # Real vulkaninfo may genuinely be installed on the machine running
    # these tests (it is, on this project's own dev machine) -- must not
    # let ensure_running() shell out to it for real. "unset" (the default)
    # means "this test doesn't care", so it's still mocked to a fixed value
    # rather than left to hit the real tool.
    if vulkan_device_index != "unset":
        monkeypatch.setattr(
            "mc_translator.runtime.ai_launcher.detect_vulkan_discrete_device_index", lambda: vulkan_device_index
        )
    else:
        monkeypatch.setattr("mc_translator.runtime.ai_launcher.detect_vulkan_discrete_device_index", lambda: None)

    # First is_alive() call (before launch) says "not running yet"; every
    # call after the process is spawned says "warmed up" so the polling loop
    # exits on its first iteration instead of sleeping.
    alive_calls = {"n": 0}

    def fake_is_alive(self):
        alive_calls["n"] += 1
        return alive_calls["n"] > 1

    monkeypatch.setattr(AiLauncher, "is_alive", fake_is_alive)

    captured = {}

    class FakePopen:
        def __init__(self, argv, **kwargs):
            captured["argv"] = argv

        def poll(self):
            return None

    monkeypatch.setattr("mc_translator.runtime.ai_launcher.subprocess.Popen", FakePopen)

    ok = launcher.ensure_running(should_continue=lambda: True, on_status=lambda text: None, on_log=lambda msg, tag="white": None)
    assert ok is True
    return captured["argv"]


def test_ensure_running_cuda_backend_uses_usecublas_and_quantkv(monkeypatch):
    argv = _run_ensure_running(monkeypatch, "cuda")
    assert "--usecublas" in argv
    assert "--quantkv" in argv
    assert "--usevulkan" not in argv
    assert "--noflashattention" not in argv


def test_ensure_running_vulkan_backend_flags_without_a_resolved_device_index(monkeypatch):
    """When detect_vulkan_discrete_device_index() can't resolve a device
    (vulkaninfo missing, integrated-only system, etc.), fall back to a bare
    --usevulkan -- today's behavior, never worse than before this feature."""
    argv = _run_ensure_running(monkeypatch, "vulkan", vulkan_device_index=None)
    assert "--usevulkan" in argv
    # No index argument was appended after the flag -- the very next argv
    # entry is the next flag, not a digit string.
    idx = argv.index("--usevulkan")
    assert argv[idx + 1] == "--noflashattention"
    assert "--usecublas" not in argv
    assert "--quantkv" not in argv
    # See ai_launcher.py's comment: flash attention silently falls back to
    # CPU compute on most non-NVIDIA Vulkan GPUs, causing severe slowdowns.
    assert "--noflashattention" in argv


def test_ensure_running_vulkan_backend_pins_resolved_discrete_device_index(monkeypatch):
    """Regression guard for the hybrid-GPU case (integrated GPU enumerates
    before a discrete one) -- when a device index IS resolved, it must be
    passed explicitly right after --usevulkan, not left blank."""
    argv = _run_ensure_running(monkeypatch, "vulkan", vulkan_device_index=1)
    idx = argv.index("--usevulkan")
    assert argv[idx + 1] == "1"


def test_ensure_running_cpu_backend_uses_no_gpu_flag(monkeypatch):
    argv = _run_ensure_running(monkeypatch, "cpu")
    assert "--usecublas" not in argv
    assert "--usevulkan" not in argv
    assert "--quantkv" not in argv
    assert "--noflashattention" not in argv


def test_ensure_running_missing_gpu_backend_key_defaults_to_cuda(monkeypatch):
    """Backward compatibility: an existing settings.ini predating this
    feature (or a FakeConfig that just returns "") must launch exactly like
    before -- --usecublas, not silently drop GPU acceleration."""
    argv = _run_ensure_running(monkeypatch, None)
    assert "--usecublas" in argv
