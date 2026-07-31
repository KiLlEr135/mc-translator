"""Tests for mc_translator.utils.hardware_detect.detect_gpu() -- the GPU
probe behind the "auto-setup local AI" feature. All subprocess calls are
monkeypatched; no real nvidia-smi/PowerShell is invoked."""
from mc_translator.utils import hardware_detect as hd


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


def test_detect_gpu_uses_nvidia_smi_when_available(monkeypatch):
    def fake_run(cmd, **kwargs):
        assert cmd[0] == "nvidia-smi"
        return _FakeCompletedProcess(0, "NVIDIA GeForce GTX 1650, 4096\n")

    monkeypatch.setattr(hd.subprocess, "run", fake_run)

    gpu = hd.detect_gpu()

    assert gpu.vendor == "nvidia"
    assert gpu.name == "NVIDIA GeForce GTX 1650"
    assert gpu.vram_mb == 4096
    assert gpu.source == "nvidia-smi"


def test_detect_gpu_picks_largest_of_multiple_nvidia_gpus(monkeypatch):
    def fake_run(cmd, **kwargs):
        return _FakeCompletedProcess(0, "NVIDIA T400, 2048\nNVIDIA RTX 4090, 24564\n")

    monkeypatch.setattr(hd.subprocess, "run", fake_run)

    gpu = hd.detect_gpu()

    assert gpu.name == "NVIDIA RTX 4090"
    assert gpu.vram_mb == 24564


def test_detect_gpu_falls_back_to_registry_when_nvidia_smi_missing(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[0])
        if cmd[0] == "nvidia-smi":
            raise FileNotFoundError("no such file")
        # AMD card: VEN_1002, 8 GB reported in bytes (qwMemorySize is bytes).
        return _FakeCompletedProcess(0, "AMD Radeon RX 6600|PCI\\VEN_1002&DEV_73FF|8589934592\n")

    monkeypatch.setattr(hd.subprocess, "run", fake_run)

    gpu = hd.detect_gpu()

    assert calls[0] == "nvidia-smi"
    assert gpu.vendor == "amd"
    assert gpu.name == "AMD Radeon RX 6600"
    assert gpu.vram_mb == 8192
    assert gpu.source == "registry"


def test_detect_gpu_returns_none_when_both_sources_fail(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(hd.subprocess, "run", fake_run)

    gpu = hd.detect_gpu()

    assert gpu.vendor == "none"
    assert gpu.vram_mb == 0
    assert gpu.source == "none"


def test_registry_fallback_never_reads_adapter_ram_wmi_field(monkeypatch):
    """Regression guard: Win32_VideoController.AdapterRAM is a 32-bit WMI
    field that wraps/caps around ~4GB on any card with more VRAM -- the
    PowerShell query must only ever reference the 64-bit
    HardwareInformation.qwMemorySize registry value."""
    captured = {}

    def fake_run(cmd, **kwargs):
        if cmd[0] == "nvidia-smi":
            raise FileNotFoundError
        captured["script"] = cmd[-1]
        return _FakeCompletedProcess(0, "")

    monkeypatch.setattr(hd.subprocess, "run", fake_run)
    hd.detect_gpu()

    assert "qwMemorySize" in captured["script"]
    assert "AdapterRAM" not in captured["script"]


def test_vendor_from_pnp_parses_known_vendor_tokens():
    assert hd._vendor_from_pnp(r"PCI\VEN_10DE&DEV_2504&SUBSYS_...") == "nvidia"
    assert hd._vendor_from_pnp(r"PCI\VEN_1002&DEV_73FF") == "amd"
    assert hd._vendor_from_pnp(r"PCI\VEN_8086&DEV_9A49") == "intel"


def test_vendor_from_pnp_returns_none_for_unrecognized_or_missing_token():
    assert hd._vendor_from_pnp(r"PCI\VEN_FFFF&DEV_0000") == "none"
    assert hd._vendor_from_pnp("") == "none"
    assert hd._vendor_from_pnp(None) == "none"


def test_detect_gpu_never_raises_on_unexpected_subprocess_error(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(hd.subprocess, "run", fake_run)

    gpu = hd.detect_gpu()

    assert gpu.vendor == "none"


# ---------------------------------------------------------------------
# detect_vulkan_discrete_device_index() -- picks which Vulkan device index
# koboldcpp should be pinned to via --usevulkan <N>, since a bare
# --usevulkan hands device selection to raw (non-VRAM-aware) Vulkan
# enumeration order, which real reports show can put an integrated GPU
# before a discrete one on hybrid systems.
# ---------------------------------------------------------------------

# Real `vulkaninfo --summary` stdout, captured verbatim from this project's
# own dev machine -- which happens to be exactly the hybrid case this
# function exists to handle: GPU0 is an AMD integrated GPU, GPU1 is a
# discrete NVIDIA GPU. Trimmed to the "Devices:" section (the parser never
# looks above it); the real output also has ~40 lines of instance
# extensions/layers above this that are irrelevant to parsing.
REAL_VULKANINFO_HYBRID_SAMPLE = """\
Devices:
========
GPU0:
\tapiVersion         = 1.3.260
\tdriverVersion      = 2.0.279
\tvendorID           = 0x1002
\tdeviceID           = 0x1636
\tdeviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
\tdeviceName         = AMD Radeon(TM) Graphics
\tdriverID           = DRIVER_ID_AMD_PROPRIETARY
\tdriverName         = AMD proprietary driver
\tdriverInfo         = 25.8.1 (AMD proprietary shader compiler)
\tconformanceVersion = 1.3.3.1
\tdeviceUUID         = 00000000-0500-0000-0000-000000000000
\tdriverUUID         = 414d442d-5749-4e2d-4452-560000000000
GPU1:
\tapiVersion         = 1.4.341
\tdriverVersion      = 610.47.0.0
\tvendorID           = 0x10de
\tdeviceID           = 0x1f99
\tdeviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
\tdeviceName         = NVIDIA GeForce GTX 1650
\tdriverID           = DRIVER_ID_NVIDIA_PROPRIETARY
\tdriverName         = NVIDIA
\tdriverInfo         = 610.47
\tconformanceVersion = 1.4.3.3
\tdeviceUUID         = 2316d445-994c-e945-4f89-0e2177c4f803
\tdriverUUID         = 85ab6016-790d-5634-a35e-f8d3c54b9858
"""


def test_parse_vulkan_discrete_index_on_real_hybrid_machine_output():
    """The exact scenario the whole function exists for: an integrated GPU
    enumerates FIRST (index 0), a discrete GPU SECOND (index 1) -- must
    resolve to 1, not 0."""
    assert hd._parse_vulkan_discrete_index(REAL_VULKANINFO_HYBRID_SAMPLE) == 1


def test_parse_vulkan_discrete_index_when_discrete_gpu_is_first():
    sample = (
        "Devices:\n========\nGPU0:\n\tdeviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\n"
        "\tdeviceName         = AMD Radeon RX 6600\nGPU1:\n\tdeviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU\n"
    )
    assert hd._parse_vulkan_discrete_index(sample) == 0


def test_parse_vulkan_discrete_index_returns_none_when_no_discrete_gpu_present():
    """A laptop with only an integrated GPU (no discrete card at all) --
    nothing to pin, ai_launcher.py falls back to a bare --usevulkan."""
    sample = "Devices:\n========\nGPU0:\n\tdeviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU\n"
    assert hd._parse_vulkan_discrete_index(sample) is None


def test_parse_vulkan_discrete_index_returns_none_for_unparseable_output():
    assert hd._parse_vulkan_discrete_index("not vulkaninfo output at all") is None
    assert hd._parse_vulkan_discrete_index("") is None


def test_detect_vulkan_discrete_device_index_returns_none_when_vulkaninfo_missing(monkeypatch):
    def fake_run(cmd, **kwargs):
        assert cmd[0] == "vulkaninfo"
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(hd.subprocess, "run", fake_run)

    assert hd.detect_vulkan_discrete_device_index() is None


def test_detect_vulkan_discrete_device_index_end_to_end(monkeypatch):
    monkeypatch.setattr(hd.subprocess, "run", lambda cmd, **kwargs: _FakeCompletedProcess(0, REAL_VULKANINFO_HYBRID_SAMPLE))

    assert hd.detect_vulkan_discrete_device_index() == 1


def test_detect_vulkan_discrete_device_index_never_raises_on_unexpected_error(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(hd.subprocess, "run", fake_run)

    assert hd.detect_vulkan_discrete_device_index() is None
