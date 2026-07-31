"""
Per-modpack, per-language manifest of already-processed mod jars.

The translation cache (mc_translator/cache.py) already avoids re-translating an
individual string twice, but every run still re-opens and re-scans every jar
to find out which strings are new. This manifest lets a re-run skip that
file-level work entirely for jars that are byte-identical to last time AND
were processed with the same options (translate_mods/translate_books/
output_mode) — if any of those differ, the jar is treated as changed so
nothing is silently left untranslated.
"""
from __future__ import annotations

import json
import os

from mc_translator.utils.atomic_write import atomic_write


def _manifest_path(mc_dir: str, lang_file_code: str) -> str:
    return os.path.join(mc_dir, f".mc_translator_manifest_{lang_file_code}.json")


def load_jar_manifest(mc_dir: str, lang_file_code: str) -> dict:
    path = _manifest_path(mc_dir, lang_file_code)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_jar_manifest(mc_dir: str, lang_file_code: str, manifest: dict) -> None:
    path = _manifest_path(mc_dir, lang_file_code)
    try:
        atomic_write(path, json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    except OSError:
        pass  # best-effort — losing the manifest only costs a slower next run


def jar_fingerprint(
    path: str,
    *,
    translate_mods: bool,
    translate_books: bool,
    output_mode: str,
) -> dict | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return {
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "mods": translate_mods,
        "books": translate_books,
        "output": output_mode,
    }


def _estimate_cache_path(mc_dir: str, lang_file_code: str) -> str:
    return os.path.join(mc_dir, f".mc_translator_estimates_{lang_file_code}.json")


def load_estimate_cache(mc_dir: str, lang_file_code: str) -> dict:
    """Per-modpack, per-language cache of StringEstimator's per-jar
    translatable-string count, keyed by jar path.

    A jar's count is a pure function of its own bytes plus
    (translate_mods, translate_books, process_mode) -- StringEstimator._count_lang
    compares the English source against the SAME jar's own bundled
    translation, never against pack/sidecar state -- so a cache hit returns
    exactly the count a fresh re-parse would produce, not an approximation.

    Separate from load_jar_manifest's skip-manifest (which additionally
    gates whether a jar's *translation* is skipped outright, and only
    applies in "inplace" output mode): this cache only ever feeds
    StringEstimator's total_strings estimate (the ETA denominator and the
    pre-flight cost preview) -- it never affects what gets translated or
    written, so a stale/missing entry only costs a slower estimate pass,
    never wrong output.
    """
    path = _estimate_cache_path(mc_dir, lang_file_code)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_estimate_cache(mc_dir: str, lang_file_code: str, cache: dict) -> None:
    path = _estimate_cache_path(mc_dir, lang_file_code)
    try:
        atomic_write(path, json.dumps(cache, ensure_ascii=False, indent=2).encode("utf-8"))
    except OSError:
        pass  # best-effort -- losing the cache only costs a slower next estimate pass
