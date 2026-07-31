"""
Per-modpack, per-language log of which mods use which translated string.

The translation cache (mc_translator/cache.py) already dedupes translations by
source text across every mod, but it stores no metadata -- just
source -> translated. This log adds just enough context (which mods a string
appears in, what kind of content it is, which engine translated it) to make
the GUI's translation review screen useful, without touching the cache
format itself. Mirrors mc_translator/runtime/manifest.py's style: per-mc_dir,
per-language, plain functions rather than a class. Lives at the top level
(next to cache.py), not under runtime/, so mc_translator.engines.service can
import it without pulling in the runtime package's __init__ (which imports
TranslationJob, which imports engines.service -- a real circular import).
"""
from __future__ import annotations

import json
import os

from mc_translator.utils.atomic_write import atomic_write


def _usage_log_path(mc_dir: str, lang_file_code: str) -> str:
    return os.path.join(mc_dir, f".mc_translator_usage_{lang_file_code}.json")


def load_usage_log(mc_dir: str, lang_file_code: str) -> dict:
    path = _usage_log_path(mc_dir, lang_file_code)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_usage_log(mc_dir: str, lang_file_code: str, usage_log: dict) -> None:
    path = _usage_log_path(mc_dir, lang_file_code)
    try:
        atomic_write(path, json.dumps(usage_log, ensure_ascii=False, indent=2).encode("utf-8"))
    except OSError:
        pass  # best-effort -- losing the usage log only affects the review screen


def record_usage(
    usage_log: dict,
    source_text: str,
    mod_name: str,
    kind: str,
    engine: str,
) -> None:
    """Record that `source_text` appears in `mod_name`, updating `usage_log`
    in place. Safe to call repeatedly for the same string across many
    files/mods -- mod names accumulate without duplicates."""
    if not source_text:
        return
    entry = usage_log.setdefault(source_text, {"mods": [], "kind": kind, "engine": engine})
    if mod_name and mod_name not in entry["mods"]:
        entry["mods"].append(mod_name)
    entry["kind"] = kind
    entry["engine"] = engine
