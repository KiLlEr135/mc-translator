"""Auto-setup local AI" JS-callable methods -- split out of gui/bridge.py
purely to keep that file under the project's 500-line guideline.
AiSetupMixin is mixed into Api (bridge.py), which owns the
self.job_state/self._ai_download_active/self._ai_download_cancelled/
self._pending_ai_plan/self.on_log/self.on_status/self._batcher state this
mixin reads and writes.

detect_ai_hardware() probes the GPU and resolves a concrete DownloadPlan
(model tier + matching koboldcpp build, via runtime/ai_catalog.py) without
downloading anything -- its return value drives the GUI's confirmation
dialog. download_ai_bundle() then executes that plan in a background
thread, following the same daemon-thread + on_log/on_status pattern as
start_translation/start_analysis."""
from __future__ import annotations

import os
import threading

from mc_translator.config import settings
from mc_translator.runtime import ai_catalog, ai_downloader
from mc_translator.utils.app_paths import app_path
from mc_translator.utils.hardware_detect import detect_gpu


def _gb(n_bytes: int) -> float:
    return round(n_bytes / (1024**3), 2)


class AiSetupMixin:
    def detect_ai_hardware(self) -> dict:
        if self.job_state.is_running or self._ai_download_active:
            return {"ok": False, "error": "Дождитесь завершения текущей операции перед автонастройкой ИИ."}

        gpu = detect_gpu()
        model = ai_catalog.select_model(gpu)
        backend = ai_catalog.gpu_backend(gpu.vendor)
        gpu_layers = ai_catalog.recommended_gpu_layers(gpu)
        asset_name = ai_catalog.kobold_asset_name(gpu.vendor)

        try:
            release = ai_downloader.fetch_latest_release(ai_catalog.KOBOLDCPP_REPO)
            engine_url, engine_size = ai_downloader.pick_asset(release, asset_name)
        except (ai_downloader.DownloadError, LookupError) as exc:
            return {"ok": False, "error": f"Не удалось получить информацию о KoboldCPP: {exc}"}

        ai_dir = app_path("AI")
        # Canonical filename regardless of which variant (koboldcpp.exe vs
        # koboldcpp-nocuda.exe) was actually fetched -- exe_path stays
        # predictable across re-runs, and the real backend is tracked
        # separately via the gpu_backend setting (see ai_launcher.py).
        engine_dest = os.path.join(ai_dir, "koboldcpp.exe")
        model_dest = os.path.join(ai_dir, model.filename)

        plan = ai_catalog.DownloadPlan(
            gpu=gpu,
            model=model,
            engine_asset_name=asset_name,
            engine_url=engine_url,
            engine_size_bytes=engine_size,
            gpu_backend=backend,
            gpu_layers=gpu_layers,
            model_dest=model_dest,
            engine_dest=engine_dest,
        )
        self._pending_ai_plan = plan

        overwrite_files = []
        if os.path.exists(engine_dest) and os.path.getsize(engine_dest) != engine_size:
            overwrite_files.append(os.path.basename(engine_dest))
        if os.path.exists(model_dest) and os.path.getsize(model_dest) != model.size_bytes:
            overwrite_files.append(os.path.basename(model_dest))

        return {
            "ok": True,
            "gpu": {
                "vendor": gpu.vendor,
                "name": gpu.name or None,
                "vramGb": round(gpu.vram_mb / 1024, 1) if gpu.vram_mb else 0,
            },
            "model": {"label": model.label, "sizeGb": _gb(model.size_bytes)},
            "engine": {"name": asset_name, "sizeGb": _gb(engine_size)},
            "totalSizeGb": _gb(plan.total_size_bytes),
            "enoughDisk": ai_downloader.has_enough_disk(ai_dir, plan.total_size_bytes),
            "overwriteFiles": overwrite_files,
        }

    def download_ai_bundle(self) -> dict:
        if self.job_state.is_running or self._ai_download_active:
            return {"ok": False, "error": "already running"}
        plan = self._pending_ai_plan
        if plan is None:
            return {"ok": False, "error": "Сначала выполните определение оборудования."}

        self._ai_download_active = True
        self._ai_download_cancelled = False
        self._batcher.push("downloadState", {"active": True})

        def should_continue() -> bool:
            return not self._ai_download_cancelled

        def run() -> None:
            try:
                self._run_ai_download(plan, should_continue)
            finally:
                self._ai_download_active = False
                self._ai_download_cancelled = False
                self._pending_ai_plan = None
                self._batcher.push("downloadState", {"active": False})

        threading.Thread(target=run, daemon=True).start()
        return {"ok": True}

    def cancel_ai_download(self) -> None:
        self._ai_download_cancelled = True

    def _run_ai_download(self, plan, should_continue) -> None:
        ai_dir = os.path.dirname(plan.engine_dest)
        try:
            if not ai_downloader.has_enough_disk(ai_dir, plan.total_size_bytes):
                self.on_log("❌ Недостаточно места на диске для загрузки.", "red")
                return

            self.on_log(f"🔧 Движок KoboldCPP ({plan.engine_asset_name})...", "cyan")
            if not ai_downloader.ensure_downloaded(
                plan.engine_url,
                plan.engine_dest,
                plan.engine_size_bytes,
                self._ai_progress(0, plan.total_size_bytes),
                should_continue,
            ):
                self.on_log("✅ Движок уже загружен, пропуск.", "dim")

            self.on_log(f"📦 Модель {plan.model.label}...", "cyan")
            if not ai_downloader.ensure_downloaded(
                ai_catalog.download_url(plan.model),
                plan.model_dest,
                plan.model.size_bytes,
                self._ai_progress(plan.engine_size_bytes, plan.total_size_bytes),
                should_continue,
            ):
                self.on_log("✅ Модель уже загружена, пропуск.", "dim")

            settings.set("AI", "exe_path", plan.engine_dest)
            settings.set("AI", "model_path", plan.model_dest)
            settings.set("AI", "gpu_layers", plan.gpu_layers)
            settings.set("AI", "gpu_backend", plan.gpu_backend)
            settings.set("AI", "ai_provider", "local")
            self.on_log("✅ Локальный ИИ настроен автоматически!", "green")
            self.on_status("Готово", 1.0)
        except ai_downloader.DownloadCancelled:
            self.on_log("⏹ Загрузка отменена.", "yellow")
        except ai_downloader.DownloadError as exc:
            self.on_log(f"❌ Ошибка загрузки: {exc}", "red")
        except Exception as exc:  # noqa: BLE001 -- must never crash the daemon thread silently
            self.on_log(f"❌ Непредвиденная ошибка при автонастройке ИИ: {exc}", "red")

    def _ai_progress(self, offset_bytes: int, total_bytes: int):
        def on_progress(got: int, _total_for_file: int) -> None:
            overall = offset_bytes + got
            frac = (overall / total_bytes) if total_bytes else None
            mb_done = overall / (1024 * 1024)
            mb_total = total_bytes / (1024 * 1024)
            self.on_status(f"Загрузка ИИ... {mb_done:.0f} / {mb_total:.0f} МБ", frac)

        return on_progress
