"""Best-effort GPU detection for the "auto-setup local AI" feature (see
runtime/ai_catalog.py, runtime/ai_downloader.py, gui/bridge.py's
detect_ai_hardware/download_ai_bundle).

This whole feature exists because of a real incident on this project: a
14B model was manually configured with only 6 of ~48 layers on GPU (a 4GB
card), which was launch-stable but silently ran most inference on CPU --
a 2+ hour hang under real load with nothing in the code to catch it. The
fix that got the app working again was picking a model sized to fit the
GPU's actual VRAM. detect_gpu() automates the "how much VRAM do I
actually have" step that was previously done by hand.

Never raises -- any detection failure just means "no GPU found", which
ai_catalog.select_model() maps to the smallest (CPU-friendly) model tier
rather than crashing the whole auto-setup flow.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

# Avoids flashing a console window when this app is run as a windowed
# (non-console) frozen .exe -- subprocess would otherwise briefly pop a
# black terminal on every detect_gpu() call. 0 on non-Windows platforms
# (getattr fallback), where it's a no-op flag anyway.
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# PnP class GUID for "Display adapters" in the Windows registry -- every
# GPU driver (NVIDIA/AMD/Intel, discrete or integrated) registers itself
# as a numbered subkey under this class.
_DISPLAY_CLASS_GUID = "{4d36e968-e325-11ce-bfc1-08002be10318}"

_VENDOR_TOKENS = {
    "VEN_10DE": "nvidia",
    "VEN_1002": "amd",
    "VEN_8086": "intel",
}


@dataclass(frozen=True)
class GpuInfo:
    vendor: str  # "nvidia" | "amd" | "intel" | "none"
    name: str
    vram_mb: int
    source: str  # diagnostic: "nvidia-smi" | "registry" | "none"


def detect_gpu() -> GpuInfo:
    gpu = _detect_nvidia()
    if gpu is not None:
        return gpu
    gpu = _detect_via_registry()
    if gpu is not None:
        return gpu
    return GpuInfo(vendor="none", name="", vram_mb=0, source="none")


def _detect_nvidia() -> GpuInfo | None:
    """nvidia-smi ships with every consumer NVIDIA driver and is normally on
    PATH -- when present, it's the single most authoritative source (exact
    vendor, exact VRAM, no registry-field ambiguity). Picks the largest GPU
    if multiple are reported (multi-GPU rigs), since that's the one koboldcpp
    would be pointed at for a single-model setup."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None

    best: GpuInfo | None = None
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name, mem_raw = parts[0], parts[1]
        try:
            vram_mb = int(float(mem_raw))
        except ValueError:
            continue
        if best is None or vram_mb > best.vram_mb:
            best = GpuInfo(vendor="nvidia", name=name, vram_mb=vram_mb, source="nvidia-smi")
    return best


def _detect_via_registry() -> GpuInfo | None:
    """Fallback for non-NVIDIA (or NVIDIA-without-nvidia-smi-on-PATH) rigs:
    one PowerShell call enumerating the Display-adapters registry class for
    DriverDesc/MatchingDeviceId/HardwareInformation.qwMemorySize.

    Deliberately NOT Win32_VideoController.AdapterRAM (WMI): that field is a
    32-bit DWORD that wraps/caps at ~4GB on any card with more VRAM -- a
    long-standing, confirmed WMI bug. qwMemorySize is the 64-bit value the
    driver itself writes and is what tools like GPU-Z read for "real" VRAM.
    """
    ps_script = (
        f"Get-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{_DISPLAY_CLASS_GUID}\\*' "
        "-ErrorAction SilentlyContinue | "
        "Where-Object { $_.'HardwareInformation.qwMemorySize' -gt 0 } | "
        "ForEach-Object { \"$($_.DriverDesc)|$($_.MatchingDeviceId)|$($_.'HardwareInformation.qwMemorySize')\" }"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None

    best: GpuInfo | None = None
    for line in result.stdout.strip().splitlines():
        fields = line.strip().split("|")
        if len(fields) < 3:
            continue
        name, device_id, mem_raw = fields[0].strip(), fields[1].strip(), fields[2].strip()
        try:
            vram_mb = int(mem_raw) // (1024 * 1024)
        except ValueError:
            continue
        if vram_mb <= 0:
            continue
        vendor = _vendor_from_pnp(device_id)
        if best is None or vram_mb > best.vram_mb:
            best = GpuInfo(vendor=vendor, name=name, vram_mb=vram_mb, source="registry")
    return best


def _vendor_from_pnp(device_id: str) -> str:
    """Parses the VEN_XXXX token out of a PNP MatchingDeviceId (e.g.
    "PCI\\VEN_10DE&DEV_2504&..."). Unknown/unmatched vendor IDs (rare
    discrete GPUs, virtual adapters) default to "none" rather than
    guessing -- ai_catalog.gpu_backend()/kobold_asset_name() route unknown
    vendors to the safe Vulkan/CPU koboldcpp build either way."""
    m = re.search(r"VEN_[0-9A-Fa-f]{4}", device_id or "")
    if not m:
        return "none"
    return _VENDOR_TOKENS.get(m.group(0).upper(), "none")


_VULKAN_GPU_HEADER_RE = re.compile(r"^GPU(\d+):\s*$")


def detect_vulkan_discrete_device_index() -> int | None:
    """Resolves which Vulkan device index koboldcpp should be pinned to via
    an explicit `--usevulkan <N>` (see runtime/ai_launcher.py), instead of
    leaving the flag blank ("--usevulkan" alone, meaning "autodetect").

    This matters because "autodetect" is NOT "pick the strongest GPU" --
    confirmed directly from koboldcpp/llama.cpp source
    (ggml_vk_instance_init enumerates every Vulkan-capable device,
    integrated or discrete, in raw loader-defined order) and from real
    reports of it landing on the wrong device on hybrid systems
    (LostRuins/koboldcpp#2056: an integrated GPU enumerated before a
    discrete one). A model sized (by ai_catalog.select_model) for a
    discrete AMD/Intel GPU's VRAM would then silently try to load onto a
    much smaller integrated GPU instead -- precisely the kind of silent
    performance/capacity cliff this whole "auto-setup local AI" feature
    exists to prevent (see the module docstring's 14B/6-layers incident).

    koboldcpp's own GUI resolves this identically: it shells out to
    `vulkaninfo --summary` and picks the first
    PHYSICAL_DEVICE_TYPE_DISCRETE_GPU entry's index. This mirrors that
    exact approach. Verified against real `vulkaninfo --summary` output on
    this project's own dev machine, which -- coincidentally -- IS a hybrid
    system (an AMD integrated GPU at GPU0, a discrete NVIDIA GPU at GPU1;
    see tests/test_hardware_detect.py for the captured sample), so the
    parser below is checked against real tool output, not a guessed format.

    Only used for backend=="vulkan" launches (AMD/Intel/no-NVIDIA-detected
    -- see ai_catalog.gpu_backend). Returns None (never raises) if
    vulkaninfo isn't on PATH, times out, or its output doesn't parse
    cleanly -- ai_launcher.py then falls back to a bare "--usevulkan",
    today's behavior, so a parsing miss is never worse than not having
    this fix at all."""
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if not result.stdout:
        return None
    return _parse_vulkan_discrete_index(result.stdout)


def _parse_vulkan_discrete_index(output: str) -> int | None:
    current_index = None
    for line in output.splitlines():
        header = _VULKAN_GPU_HEADER_RE.match(line.strip())
        if header:
            current_index = int(header.group(1))
            continue
        if current_index is not None and "deviceType" in line and "DISCRETE_GPU" in line:
            return current_index
    return None
