"""
Generic JSON processor for translating any JSON file.
Extracts all string values that look like source language and are not technical terms,
translates them, and writes the updated JSON back.
"""
from __future__ import annotations

import json
import os
from typing import Any

from mc_translator.engines.base import EngineCallbacks
from mc_translator.engines.service import TranslationService
from mc_translator.json_utils import iter_all_json_strings, load_lenient_json
from mc_translator.processors.resourcepack_path import compute_target_paths
from mc_translator.text_processing import (
    already_translated,
    is_technical_term,
    looks_like_source_language,
)


class GenericJsonProcessor:
    def __init__(
        self,
        service: TranslationService,
        state: "JobState",
        callbacks: EngineCallbacks,
    ) -> None:
        self.service = service
        self.state = state
        self.callbacks = callbacks

    def process(
        self,
        file_path: str,
        *,
        target_lang: dict,
        mode: str,
        output_mode: str,
        pack_writer: object | None = None,
        mc_dir: str | None = None,
    ) -> None:
        """
        Process a single JSON file.

        Args:
            file_path: path to the source JSON file.
            target_lang: language dict with keys 'file' (lang code) and 'regex' (regex for already translated).
            mode: translation mode ('append', 'skip', 'force').
            output_mode: output mode ('resourcepack' or 'inplace').
            pack_writer: PackWriter instance if output_mode is 'resourcepack'.
            mc_dir: modpack root, used to compute a collision-safe resourcepack
                internal path (see resourcepack_path.py). Optional -- callers
                that don't pass it (e.g. ArchiveProcessor, always inplace) get
                the old basename-only behavior, which is harmless there since
                it never reaches the resourcepack write path.
        """
        try:
            with open(file_path, encoding="utf-8-sig") as f:
                raw = f.read()
            # load_lenient_json (not bare json.loads) strips "//"/"/* */"
            # line/block comments and trailing commas -- several mods (e.g.
            # lithostitched.json, polylib.json) ship JSON5/JSONC-style
            # config with real "//" line comments, which bare json.loads
            # rejects outright, silently skipping the whole file.
            data = load_lenient_json(raw)
        except (json.JSONDecodeError, OSError) as exc:
            self.callbacks.on_log(f"❌ Ошибка чтения JSON {file_path}: {exc}", "red")
            return

        # Determine target path for output
        # If file contains language code like en_us, replace it; else append language code before extension.
        base_name = os.path.basename(file_path)
        target_name, target_path, internal_path = compute_target_paths(file_path, target_lang["file"], mc_dir)

        # Load an existing translation (if any) so append/skip modes can tell
        # what's already done, AND so already-translated strings get
        # re-applied to `data` below -- without this, every append-mode run
        # silently reverted any string not freshly re-translated this run
        # back to English, since only this run's `translations` ever got
        # written into `data` (the English source was never merged with a
        # prior translation the way _process_lang_entry/loose_json.py do).
        # Keyed by the REAL path tuple, not a "/"-joined string: a joined
        # string is ambiguous whenever a key itself contains "/", which is
        # common in Forge/NeoForge config JSON that uses a "//" key as an
        # inline comment (e.g. attributefix's per-attribute files --
        # {"modify_range": {"//": "...", "value": false}, "min": {...}, ...}
        # -- path ("modify_range", "//") joined to "modify_range///" and
        # split back into FOUR segments instead of two). The old join/split
        # round-trip silently mis-reconstructed such paths on write-back, so
        # a translated comment string was computed (and cached/paid for) but
        # never actually applied -- logged every time as an obscure "Ошибка
        # применения перевода" with an empty exception message.
        existing_map: dict[tuple, str] = {}
        if os.path.exists(target_path):
            try:
                with open(target_path, encoding="utf-8") as f:
                    existing_data = load_lenient_json(f.read())
                existing_map = dict(iter_all_json_strings(existing_data))
            except (json.JSONDecodeError, OSError):
                existing_map = {}

        # Collect translatable strings
        pending: dict[tuple, str] = {}
        for path_tuple, string in iter_all_json_strings(data):
            if is_technical_term(string):
                continue
            if not looks_like_source_language(string):
                continue
            existing = existing_map.get(path_tuple, "")
            if mode == "append" and existing.strip() and existing != string:
                continue  # already has a real translation -- keep it
            if mode == "append" and existing.strip() == string:
                continue  # "translation" happens to equal source -- treat as done
            if mode == "skip" and existing.strip() and already_translated(existing, target_lang["regex"]):
                continue
            pending[path_tuple] = string

        # Seed `data` with already-translated values for paths that aren't
        # being freshly translated this run, so they survive into the output
        # instead of reverting to the English source.
        for path_tuple, value in existing_map.items():
            if path_tuple in pending or not value.strip():
                continue
            try:
                set_at_path(data, path_tuple, value)
            except Exception:
                pass  # existing_map's structure has diverged from data's; skip

        if not pending:
            # Nothing to translate; just copy if needed. Prefer an existing
            # translated file over the English source, so a prior 'inplace'
            # run isn't silently overwritten with untranslated text.
            if output_mode == "resourcepack" and pack_writer:
                source_path = target_path if os.path.exists(target_path) else file_path
                with open(source_path, "rb") as f:
                    pack_writer.write(internal_path, f.read())
            elif output_mode == "inplace":
                # No change needed; just ensure target exists? skip.
                pass
            return

        # translate_dict only needs plain string keys to correlate its
        # input/output -- assign collision-free sequential indices and keep
        # the real path tuples here for write-back, instead of round-
        # tripping through a joined string.
        index_to_path = dict(enumerate(pending))
        to_translate = {str(i): s for i, s in enumerate(pending.values())}

        # Translate
        self.callbacks.on_log(
            f"⚡ Перевод JSON {os.path.basename(file_path)} — {len(to_translate)} строк",
            "cyan",
        )
        translated = self.service.translate_dict(
            to_translate,
            target_lang,
            self.callbacks,
            context="JSON",
            usage_label=(base_name, "JSON"),
        )
        # Apply translations
        for idx_str, value in translated.items():
            path_tuple = index_to_path[int(idx_str)]
            try:
                set_at_path(data, path_tuple, value)
            except Exception as exc:
                self.callbacks.on_log(
                    f"❌ Ошибка применения перевода к {path_tuple} in {file_path}: {exc}",
                    "red",
                )
                continue

        # Write output
        if output_mode == "resourcepack" and pack_writer:
            payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            pack_writer.write(internal_path, payload)
        elif output_mode == "inplace":
            try:
                with open(target_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except OSError as exc:
                self.callbacks.on_log(
                    f"❌ Ошибка записи JSON {target_path}: {exc}", "red"
                )
        else:
            # Should not happen
            pass


def set_at_path(data: Any, path: tuple[Any, ...], value: Any) -> None:
    """Set a nested value in a dict/list structure given a path tuple."""
    cur = data
    for part in path[:-1]:
        if isinstance(cur, dict):
            cur = cur[part]
        elif isinstance(cur, list):
            # Convert index to int
            idx = int(str(part))
            if idx < 0 or idx >= len(cur):
                raise IndexError(f"List index {idx} out of range")
            cur = cur[idx]
        else:
            raise TypeError(f"Unexpected type {type(cur)} while traversing path")
    last = path[-1]
    if isinstance(cur, dict):
        cur[last] = value
    elif isinstance(cur, list):
        idx = int(str(last))
        if idx < 0 or idx >= len(cur):
            raise IndexError(f"List index {idx} out of range")
        cur[idx] = value
    else:
        raise TypeError(f"Cannot set attribute on {type(cur)}")