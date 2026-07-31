"""Model/engine selection logic for the "auto-setup local AI" feature.

Pure data + pure functions (no I/O) -- gui/bridge.py's detect_ai_hardware()
combines these with utils/hardware_detect.detect_gpu() and a live
koboldcpp-releases lookup (runtime/ai_downloader.py) to build a DownloadPlan,
which download_ai_bundle() then executes.

Model tiers are Qwen2.5-Instruct at q5_k_m -- the same family/quant already
proven on this project's own real hardware (see the "local_ai_koboldcpp_setup"
project memory: a 3B model fully GPU-resident on a 4GB card, ~2.44GB weights
+ ~1.8GB headroom). Tier floors are chosen per-tier, anchored to that real
data point for the 3B row, rather than derived from one blanket "reserve
X%" formula -- a formula generic enough to cover every tier would have
excluded this project's own proven-working configuration from its own
"safe" tier, which defeats the point of anchoring to real data at all.
"""
from __future__ import annotations

from dataclasses import dataclass

from mc_translator.utils.hardware_detect import GpuInfo

KOBOLDCPP_REPO = "LostRuins/koboldcpp"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    repo: str
    filename: str
    size_bytes: int
    min_total_vram_gb: float


# Ordered ascending by min_total_vram_gb -- select_model() picks the LAST
# tier whose floor the detected VRAM clears.
#
# bartowski's repos are used for the 7B+ rows specifically because the
# *official* Qwen GGUF repos split those into multiple sibling files
# (model-00001-of-0000N.gguf); bartowski re-packages the identical quant as
# a single file, so ai_downloader.py never needs multi-file download logic.
#
# The 32B row's floor (26GB) deliberately sits just above the common 24GB
# consumer-card ceiling (RTX 3090/4090) -- a 24GB card falls back to the 14B
# tier instead of attempting a tight, marginal 32B fit, matching this
# project's own hard-learned lesson that partial/marginal GPU fit (not an
# outright OOM) was what caused the original silent-hang incident.
MODEL_TIERS: list[ModelSpec] = [
    ModelSpec(
        key="0.5b",
        label="Qwen2.5-0.5B-Instruct",
        repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="qwen2.5-0.5b-instruct-q5_k_m.gguf",
        size_bytes=522_186_592,  # exact, via HF API siblings[].size -- ~0.49 GB
        min_total_vram_gb=0.0,
    ),
    ModelSpec(
        key="1.5b",
        label="Qwen2.5-1.5B-Instruct",
        repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename="qwen2.5-1.5b-instruct-q5_k_m.gguf",
        size_bytes=1_285_494_304,  # exact -- ~1.20 GB
        min_total_vram_gb=2.0,
    ),
    ModelSpec(
        key="3b",
        label="Qwen2.5-3B-Instruct",
        repo="Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename="qwen2.5-3b-instruct-q5_k_m.gguf",
        size_bytes=2_438_740_384,  # exact -- ~2.27 GB; this project's own proven config
        min_total_vram_gb=3.5,
    ),
    ModelSpec(
        key="7b",
        label="Qwen2.5-7B-Instruct",
        repo="bartowski/Qwen2.5-7B-Instruct-GGUF",
        filename="Qwen2.5-7B-Instruct-Q5_K_M.gguf",
        size_bytes=5_444_831_936,  # exact -- ~5.07 GB
        min_total_vram_gb=7.0,
    ),
    ModelSpec(
        key="14b",
        label="Qwen2.5-14B-Instruct",
        repo="bartowski/Qwen2.5-14B-Instruct-GGUF",
        filename="Qwen2.5-14B-Instruct-Q5_K_M.gguf",
        size_bytes=10_508_873_856,  # exact -- ~9.79 GB
        min_total_vram_gb=12.5,
    ),
    ModelSpec(
        key="32b",
        label="Qwen2.5-32B-Instruct",
        repo="bartowski/Qwen2.5-32B-Instruct-GGUF",
        filename="Qwen2.5-32B-Instruct-Q5_K_M.gguf",
        size_bytes=23_262_157_696,  # exact -- ~21.67 GB
        min_total_vram_gb=26.0,
    ),
]


@dataclass(frozen=True)
class DownloadPlan:
    gpu: GpuInfo
    model: ModelSpec
    engine_asset_name: str
    engine_url: str
    engine_size_bytes: int
    gpu_backend: str  # "cuda" | "vulkan" | "cpu"
    gpu_layers: int
    model_dest: str
    engine_dest: str

    @property
    def total_size_bytes(self) -> int:
        return self.model.size_bytes + self.engine_size_bytes


def select_model(gpu: GpuInfo) -> ModelSpec:
    vram_gb = gpu.vram_mb / 1024
    chosen = MODEL_TIERS[0]
    for tier in MODEL_TIERS:
        if vram_gb >= tier.min_total_vram_gb:
            chosen = tier
    return chosen


def gpu_backend(vendor: str) -> str:
    if vendor == "nvidia":
        return "cuda"
    if vendor in ("amd", "intel"):
        return "vulkan"
    return "cpu"


def kobold_asset_name(vendor: str) -> str:
    """koboldcpp.exe (full CUDA build) for NVIDIA; koboldcpp-nocuda.exe
    (Vulkan+CPU, the project's own README-recommended choice for non-NVIDIA)
    for everything else, including "no GPU detected"."""
    return "koboldcpp.exe" if vendor == "nvidia" else "koboldcpp-nocuda.exe"


def recommended_gpu_layers(gpu: GpuInfo) -> int:
    """0 (CPU-only) only for the true no-GPU case; every real tier's floor
    is sized so the chosen model fits fully in VRAM, so 99 (koboldcpp's
    "as many as exist" sentinel) is safe for every other case."""
    return 0 if gpu.vendor == "none" else 99


def download_url(spec: ModelSpec) -> str:
    return f"https://huggingface.co/{spec.repo}/resolve/main/{spec.filename}"
