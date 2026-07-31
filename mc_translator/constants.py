from mc_translator.utils.app_paths import app_path

SETTINGS_FILE = app_path("settings.ini")
CACHE_FILE_STD = app_path("cache.json")
CACHE_FILE_AI = app_path("ai_cache.json")
DICT_FILE = app_path("dictionary.json")
KOBOLD_API = "http://localhost:5001/v1/chat/completions"
KOBOLD_MODELS_URL = "http://localhost:5001/v1/models"
OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_OPENROUTER_MODEL = "google/gemma-2-9b-it:free"

# Claude's native API -- NOT the OpenAI-compatible chat-completions shape
# CustomApiEngine speaks (different auth header, endpoint path, and response
# body), so it's the one provider that needs its own engine/constants
# instead of just a base_url pointed at CustomApiEngine (see engines/
# anthropic.py). ANTHROPIC_VERSION is a required header Anthropic versions
# independently of any particular model.
ANTHROPIC_API = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# Built-in safety net for the "auto-cycle free OpenRouter models" feature --
# used only when the user hasn't supplied their own model list AND a live
# fetch from OPENROUTER_MODELS_URL fails (offline, API down, etc.), so the
# feature still has *something* to rotate through instead of silently
# degrading to a single hardcoded model.
FALLBACK_FREE_MODELS = [
    "google/gemma-2-9b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
]

AI_PROVIDERS = {
    "local": "Локально (KoboldCPP)",
    "openrouter": "OpenRouter (облако)",
    "custom": "Другой ИИ (свой API)",
    "anthropic": "Claude (Anthropic)",
}

# Placeholder default for CUSTOM_AI/base_url -- OpenAI's own endpoint, since
# it's the most recognizable example of the OpenAI-compatible chat-completions
# shape CustomApiEngine speaks (see engines/custom_api.py). Any other
# provider/local server using the same shape just needs its URL pasted in.
DEFAULT_CUSTOM_API_URL = "https://api.openai.com/v1/chat/completions"

KEYS_TO_TRANSLATE = frozenset({
    "name", "title", "text", "description", "subtitle", "label", "hover_text", "link_text", "pages",
})

BOOK_PATH_MARKERS = ("patchouli", "lexicon", "guide")
MD_PATH_MARKERS = ("/en_us/", "/ae2guide/", "/guide/", "/manual/", "/lexicon/")
RESEARCH_PATH_MARKERS = ("/research/", "/researches/", "/quests/")

LOOSE_JSON_SEARCH_DIRS = ("kubejs/assets", "defaultconfigs", "config/ftbquests/lang")

IGNORE_TERMS = [
    "RF", "FE", "EU", "J", "mB", "mB/t", "RF/t", "FE/t", "AE", "kW", "kRF", "mB/tick",
    "ticks", "GUI", "UI", "HUD", "JEI", "REI", "EMI", "API", "JSON", "NBT", "FPS", "TPS",
    "HP", "XP", "MP", "XP/t", "XYZ", "RGB", "ID", "II", "III", "IV", "VI", "VII", "VIII",
    "IX", "XI", "XII",
]
IGNORE_TERMS.sort(key=len, reverse=True)

LANGUAGES = {
    "Русский": {"file": "ru_ru", "api": "ru", "deepl": "RU", "name": "Russian", "regex": r"[А-Яа-яЁё]"},
    "Українська": {"file": "uk_ua", "api": "uk", "deepl": "UK", "name": "Ukrainian", "regex": r"[А-Яа-яІіЇїЄєҐґ]"},
    "English (UK)": {"file": "en_gb", "api": "en", "deepl": "EN-GB", "name": "English", "regex": r"[a-zA-Z]"},
    "Español": {"file": "es_es", "api": "es", "deepl": "ES", "name": "Spanish", "regex": r"[áéíóúüñÁÉÍÓÚÜÑ]"},
    "Deutsch": {"file": "de_de", "api": "de", "deepl": "DE", "name": "German", "regex": r"[äöüßÄÖÜẞ]"},
    "Français": {"file": "fr_fr", "api": "fr", "deepl": "FR", "name": "French", "regex": r"[àâæçéèêëîïôœùûüÿÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ]"},
    "中文 (Упрощ.)": {"file": "zh_cn", "api": "zh-CN", "deepl": "ZH", "name": "Simplified Chinese", "regex": r"[\u4e00-\u9fff]"},
    "日本語": {"file": "ja_jp", "api": "ja", "deepl": "JA", "name": "Japanese", "regex": r"[\u3040-\u30ff\u4e00-\u9fff]"},
    "한국어": {"file": "ko_kr", "api": "ko", "deepl": "KO", "name": "Korean", "regex": r"[\uac00-\ud7af]"},
    "Português": {"file": "pt_br", "api": "pt", "deepl": "PT-BR", "name": "Portuguese", "regex": r"[ãõáéíóúâêôÃÕÁÉÍÓÚÂÊÔ]"},
    "Italiano": {"file": "it_it", "api": "it", "deepl": "IT", "name": "Italian", "regex": r"[àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ]"},
    "Polski": {"file": "pl_pl", "api": "pl", "deepl": "PL", "name": "Polish", "regex": r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]"},
}

PACK_FORMATS = {
    "1.7.10": {"rp": 1, "dp": 1},
    "1.8.9": {"rp": 1, "dp": 1},
    "1.10.2": {"rp": 2, "dp": 1},
    "1.12.2": {"rp": 3, "dp": 1},
    "1.14.4": {"rp": 4, "dp": 4},
    "1.15.2": {"rp": 5, "dp": 5},
    "1.16.5": {"rp": 6, "dp": 6},
    "1.18.2": {"rp": 8, "dp": 9},
    "1.19.2": {"rp": 9, "dp": 10},
    # 1.19.3 has its own distinct pack_format (verified against the
    # Minecraft Wiki's pack format history table) -- it does NOT share
    # 1.19.2's or 1.19.4's format, unlike most other patch releases.
    "1.19.3": {"rp": 12, "dp": 10},
    "1.19.4": {"rp": 13, "dp": 12},
    "1.20.1": {"rp": 15, "dp": 15},
    # 1.20.2 also has its own distinct format, between 1.20.1's and
    # 1.20.4's (which 1.20.3 shares).
    "1.20.2": {"rp": 18, "dp": 18},
    "1.20.4": {"rp": 22, "dp": 26},  # 1.20.3 shares this exact format
    "1.20.6": {"rp": 32, "dp": 41},  # 1.20.5 shares this exact format
    "1.21.1": {"rp": 34, "dp": 48},
    "1.21.3": {"rp": 42, "dp": 57},
    "1.21.4": {"rp": 46, "dp": 61},
}

MC_VERSIONS = list(PACK_FORMATS.keys())

# The file-format "buckets" produced by discover_mod_content_files() (plus
# mcfunction, discovered separately) -- individually toggleable in the GUI's
# "Типы файлов" settings tab, on top of the three coarse translate_mods/
# translate_books/translate_quests switches. (key, display label).
MOD_CONTENT_FILE_TYPES = [
    ("generic_json", "JSON-файлы"),
    ("lang", ".lang файлы"),
    ("nbt", "NBT (.nbt/.dat)"),
    ("yaml_toml", "YAML/TOML"),
    ("text", "Текст (.txt/.md)"),
    ("cfg", "CFG"),
    ("xml", "XML"),
    ("bat", "BAT"),
    ("mcfunction", ".mcfunction"),
    ("archive", "Архивы (.zip/.mcpack/...)"),
]
