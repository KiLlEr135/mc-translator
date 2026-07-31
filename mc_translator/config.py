import configparser
import io
import os
import time

from mc_translator.constants import DEFAULT_CUSTOM_API_URL, MOD_CONTENT_FILE_TYPES, SETTINGS_FILE
from mc_translator.utils.atomic_write import atomic_write


class ConfigManager:
    """Persistent application settings (settings.ini)."""

    _DEFAULTS = {
        "GENERAL": {
            "mc_dir": os.getcwd(),
            "theme": "Dark",
            "color": "blue",
            "smart_glue": "True",
            "google_workers": "5",
        },
        "AI": {
            "exe_path": "koboldcpp.exe",
            "model_path": "",
            "gpu_layers": "99",
            "ai_provider": "local",
            "global_context": "",
            # "cuda" | "vulkan" | "cpu" -- which launch flags ai_launcher.py
            # uses to start exe_path. Defaults to "cuda" so every existing
            # install (which was always launched with --usecublas before
            # this key existed) keeps behaving identically. Only the
            # "auto-setup local AI" flow (gui/bridge.py's
            # detect_ai_hardware/download_ai_bundle) sets this to something
            # other than the default, matching whichever koboldcpp build it
            # actually downloaded.
            "gpu_backend": "cuda",
        },
        "API": {
            "deepl_key": "",
        },
        "OPENROUTER": {
            "api_key": "",
            "model": "google/gemma-2-9b-it:free",
            "site_url": "",
            "app_name": "MC Translator",
            # Auto-cycle feature: when a free model's rate/quota limit is hit,
            # switch to the next free model instead of stopping (see
            # engines/openrouter.py's ModelRotator). "model" above stays the
            # single-model fallback used when this is off.
            "auto_cycle_free": "False",
            # User-supplied list of free model IDs to cycle through (one per
            # line or comma-separated). Empty = auto-fetch every ":free"
            # model from OpenRouter's public /models endpoint at run start.
            "free_models": "",
        },
        "CUSTOM_AI": {
            "base_url": DEFAULT_CUSTOM_API_URL,
            "api_key": "",
            "model": "gpt-4o-mini",
        },
        "ANTHROPIC": {
            "api_key": "",
            "model": "",
        },
        "SESSION": {
            "language": "Русский",
            "mc_version": "1.20.1",
            "output_mode": "resourcepack",
            "pack_name": "MCTranslator_Pack",
            "translate_mods": "True",
            "translate_books": "True",
            "translate_quests": "True",
            "engine": "google",
            "google_mode": "single",
            "ai_mode": "safe",
            "process_mode": "append",
        },
        # One bool per bucket in MOD_CONTENT_FILE_TYPES -- all on by default,
        # so a fresh install translates exactly like before this feature
        # existed. Individually toggleable in the GUI's "Типы файлов" tab.
        "FILETYPES": {key: "True" for key, _ in MOD_CONTENT_FILE_TYPES},
    }

    def __init__(self) -> None:
        self._config = configparser.ConfigParser()
        self.load()

    def load(self) -> None:
        try:
            self._config.read(SETTINGS_FILE, encoding="utf-8")
        except configparser.Error:
            # A truncated/corrupt settings.ini (e.g. from an interrupted
            # non-atomic write, or a bad manual edit) used to crash the
            # module-level `settings` singleton at import time, taking down
            # both the GUI and the CLI with no recovery path. Back up the
            # broken file and start fresh from defaults instead.
            if os.path.exists(SETTINGS_FILE):
                backup_path = f"{SETTINGS_FILE}.bak-corrupt-{int(time.time())}"
                try:
                    os.replace(SETTINGS_FILE, backup_path)
                except OSError:
                    pass
            self._config = configparser.ConfigParser()
        changed = False
        for section, keys in self._DEFAULTS.items():
            if not self._config.has_section(section):
                self._config.add_section(section)
                changed = True
            for key, value in keys.items():
                if not self._config.has_option(section, key):
                    self._config.set(section, key, str(value))
                    changed = True
        if changed:
            self.save()

    def save(self) -> None:
        buf = io.StringIO()
        self._config.write(buf)
        atomic_write(SETTINGS_FILE, buf.getvalue().encode("utf-8"))

    def get(self, section: str, key: str) -> str:
        return self._config.get(section, key)

    def set(self, section: str, key: str, value) -> None:
        self._config.set(section, key, str(value))
        self.save()

    def getboolean(self, section: str, key: str, fallback: bool = False) -> bool:
        try:
            return self._config.getboolean(section, key)
        except (configparser.Error, ValueError):
            return fallback

    def getint(self, section: str, key: str, fallback: int = 0) -> int:
        try:
            raw = self.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
        try:
            return int(raw.strip())
        except (ValueError, AttributeError):
            return fallback


# Shared singleton for GUI and jobs
settings = ConfigManager()
