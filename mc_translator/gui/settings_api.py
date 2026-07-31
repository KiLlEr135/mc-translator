"""Settings-tab JS-callable methods -- split out of gui/bridge.py purely to
keep that file under the project's 500-line guideline. SettingsMixin is
mixed into Api (bridge.py); these two methods don't touch any Api instance
state (they're pure functions of the global `settings` object), so this is
pure code motion, not a behavior change."""
from __future__ import annotations

from mc_translator.config import settings
from mc_translator.constants import DEFAULT_CUSTOM_API_URL, DEFAULT_OPENROUTER_MODEL, MOD_CONTENT_FILE_TYPES


class SettingsMixin:
    def get_settings(self) -> dict:
        return {
            "aiExePath": settings.get("AI", "exe_path"),
            "aiModelPath": settings.get("AI", "model_path"),
            "aiGlobalContext": settings.get("AI", "global_context"),
            "aiGpuLayers": settings.getint("AI", "gpu_layers", 99),
            "openrouterApiKey": settings.get("OPENROUTER", "api_key"),
            "openrouterModel": settings.get("OPENROUTER", "model") or DEFAULT_OPENROUTER_MODEL,
            "openrouterSiteUrl": settings.get("OPENROUTER", "site_url"),
            "openrouterAppName": settings.get("OPENROUTER", "app_name"),
            "openrouterFreeModels": settings.get("OPENROUTER", "free_models"),
            "customBaseUrl": settings.get("CUSTOM_AI", "base_url") or DEFAULT_CUSTOM_API_URL,
            "customApiKey": settings.get("CUSTOM_AI", "api_key"),
            "customModel": settings.get("CUSTOM_AI", "model"),
            "anthropicApiKey": settings.get("ANTHROPIC", "api_key"),
            "anthropicModel": settings.get("ANTHROPIC", "model"),
            "smartGlue": settings.getboolean("GENERAL", "smart_glue"),
            "googleWorkers": settings.getint("GENERAL", "google_workers", 5),
            "deeplKey": settings.get("API", "deepl_key"),
            "fileTypes": [
                {"key": key, "label": label, "enabled": settings.getboolean("FILETYPES", key)}
                for key, label in MOD_CONTENT_FILE_TYPES
            ],
        }

    def save_settings(self, data: dict) -> dict:
        """Refused while a job is running: TranslationService rebuilds its
        engine (and re-reads OPENROUTER/AI settings live) roughly once per
        file for the whole duration of a run, so a settings change landing
        mid-run would switch the model/key/gpu_layers out from under an
        in-progress translation with no isolation between the live job and
        whatever the user is currently editing -- same rule as clear_cache."""
        if self.job_state.is_running:
            return {"ok": False, "error": "Дождитесь завершения текущей операции перед сохранением настроек."}
        settings.set("AI", "exe_path", (data.get("aiExePath") or "").strip())
        settings.set("AI", "model_path", (data.get("aiModelPath") or "").strip())
        settings.set("AI", "global_context", (data.get("aiGlobalContext") or "").strip())
        try:
            gpu_layers = int(data.get("aiGpuLayers"))
        except (TypeError, ValueError):
            gpu_layers = 99
        settings.set("AI", "gpu_layers", gpu_layers)
        settings.set("OPENROUTER", "api_key", (data.get("openrouterApiKey") or "").strip())
        settings.set("OPENROUTER", "model", (data.get("openrouterModel") or "").strip())
        settings.set("OPENROUTER", "site_url", (data.get("openrouterSiteUrl") or "").strip())
        settings.set("OPENROUTER", "app_name", (data.get("openrouterAppName") or "").strip())
        settings.set("OPENROUTER", "free_models", (data.get("openrouterFreeModels") or "").strip())
        settings.set("CUSTOM_AI", "base_url", (data.get("customBaseUrl") or "").strip() or DEFAULT_CUSTOM_API_URL)
        settings.set("CUSTOM_AI", "api_key", (data.get("customApiKey") or "").strip())
        settings.set("CUSTOM_AI", "model", (data.get("customModel") or "").strip())
        settings.set("ANTHROPIC", "api_key", (data.get("anthropicApiKey") or "").strip())
        settings.set("ANTHROPIC", "model", (data.get("anthropicModel") or "").strip())
        settings.set("GENERAL", "smart_glue", bool(data.get("smartGlue")))
        try:
            workers = int(data.get("googleWorkers"))
        except (TypeError, ValueError):
            workers = 5
        settings.set("GENERAL", "google_workers", workers)
        settings.set("API", "deepl_key", (data.get("deeplKey") or "").strip())
        file_types = {ft["key"]: ft.get("enabled") for ft in (data.get("fileTypes") or [])}
        for key, _label in MOD_CONTENT_FILE_TYPES:
            if key in file_types:
                settings.set("FILETYPES", key, bool(file_types[key]))
        return {"ok": True}
