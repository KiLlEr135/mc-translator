"""Tests for mc_translator.runtime.ai_catalog -- the model/engine tier
selection logic behind the "auto-setup local AI" feature."""
from mc_translator.runtime import ai_catalog as catalog
from mc_translator.utils.hardware_detect import GpuInfo


def _gpu(vendor, vram_mb):
    return GpuInfo(vendor=vendor, name="test", vram_mb=vram_mb, source="test")


def test_select_model_no_gpu_gets_smallest_tier():
    spec = catalog.select_model(_gpu("none", 0))
    assert spec.key == "0.5b"


def test_select_model_just_below_1_5b_floor_stays_on_0_5b():
    spec = catalog.select_model(_gpu("nvidia", int(1.9 * 1024)))
    assert spec.key == "0.5b"


def test_select_model_at_1_5b_floor():
    spec = catalog.select_model(_gpu("nvidia", 2 * 1024))
    assert spec.key == "1.5b"


def test_select_model_real_proven_4gb_card_selects_3b():
    """Anchor case: this project's own real, live-verified hardware (a 4GB
    GTX 1650) must resolve to the 3B tier -- the exact config already
    confirmed working end-to-end on a real ATM10 translation run."""
    spec = catalog.select_model(_gpu("nvidia", 4 * 1024))
    assert spec.key == "3b"


def test_select_model_just_below_3b_floor_stays_on_1_5b():
    spec = catalog.select_model(_gpu("nvidia", int(3.4 * 1024)))
    assert spec.key == "1.5b"


def test_select_model_7b_tier():
    spec = catalog.select_model(_gpu("nvidia", 8 * 1024))
    assert spec.key == "7b"


def test_select_model_14b_tier():
    spec = catalog.select_model(_gpu("nvidia", 16 * 1024))
    assert spec.key == "14b"


def test_select_model_24gb_card_falls_back_to_14b_not_32b():
    """A common top-end consumer card (RTX 3090/4090, 24GB) deliberately
    does NOT qualify for the 32B tier -- see ai_catalog.py's comment on why
    the 32B floor sits above the 24GB ceiling."""
    spec = catalog.select_model(_gpu("nvidia", 24 * 1024))
    assert spec.key == "14b"


def test_select_model_32b_tier_needs_26gb():
    spec = catalog.select_model(_gpu("nvidia", 26 * 1024))
    assert spec.key == "32b"


def test_gpu_backend_maps_vendors_correctly():
    assert catalog.gpu_backend("nvidia") == "cuda"
    assert catalog.gpu_backend("amd") == "vulkan"
    assert catalog.gpu_backend("intel") == "vulkan"
    assert catalog.gpu_backend("none") == "cpu"


def test_kobold_asset_name_nvidia_gets_cuda_build():
    assert catalog.kobold_asset_name("nvidia") == "koboldcpp.exe"


def test_kobold_asset_name_non_nvidia_gets_nocuda_build():
    assert catalog.kobold_asset_name("amd") == "koboldcpp-nocuda.exe"
    assert catalog.kobold_asset_name("intel") == "koboldcpp-nocuda.exe"
    assert catalog.kobold_asset_name("none") == "koboldcpp-nocuda.exe"


def test_recommended_gpu_layers_zero_only_when_no_gpu():
    assert catalog.recommended_gpu_layers(_gpu("none", 0)) == 0
    assert catalog.recommended_gpu_layers(_gpu("nvidia", 4 * 1024)) == 99
    assert catalog.recommended_gpu_layers(_gpu("amd", 8 * 1024)) == 99


def test_download_url_format():
    spec = catalog.select_model(_gpu("nvidia", 4 * 1024))
    url = catalog.download_url(spec)
    assert url == f"https://huggingface.co/{spec.repo}/resolve/main/{spec.filename}"
    assert url.startswith("https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/")


def test_model_tiers_are_sorted_ascending_by_vram_floor():
    floors = [tier.min_total_vram_gb for tier in catalog.MODEL_TIERS]
    assert floors == sorted(floors)
