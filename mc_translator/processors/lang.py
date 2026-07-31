"""
Processor for .lang files (key=value language files).
"""
from __future__ import annotations

import os

from mc_translator.engines.base import EngineCallbacks
from mc_translator.engines.service import TranslationService
from mc_translator.processors.resourcepack_path import compute_target_paths
from mc_translator.text_processing import (
    already_translated,
)
from mc_translator.utils.legacy_lang import is_legacy_version, legacy_lang_code


class LangProcessor:
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
        mc_version: str = "1.12.2",
        mc_dir: str | None = None,
    ) -> None:
        """
        Process a single .lang file.

        Args:
            file_path: path to source .lang file (e.g., en_us.lang).
            target_lang: language dict with 'file' (lang code) and 'regex' for detection.
            mode: translation mode ('append', 'skip', 'force').
            output_mode: output mode ('resourcepack' or 'inplace').
            pack_writer: PackWriter if output_mode is 'resourcepack'.
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError as exc:
            self.callbacks.on_log(f"❌ Ошибка чтения .lang {file_path}: {exc}", "red")
            return

        # Determine target path for output
        base_name = os.path.basename(file_path)

        # Determine case and format based on version
        # 1.13+ is lower case ru_ru. 1.12- is often ru_RU
        target_file_code = target_lang["file"]
        if is_legacy_version(mc_version):
            target_file_code = legacy_lang_code(target_file_code)

        # compute_target_paths does a case-preserving "en_us"->target_file_code
        # replace (case-insensitive match, original-case substitution), which
        # already generalizes this file's old dedicated "match original case
        # if it was en_US" branch -- no separate handling needed here anymore.
        target_name, target_path, internal_path = compute_target_paths(file_path, target_file_code, mc_dir)

        # Parse lines into segments: either comment/blank or key=value
        parsed: list[tuple[str, str]] = []  # (type, content) where type: 'comment', 'blank', 'kv'
        kv_values: list[str] = []  # original values for keys that need translation
        kv_indices: list[int] = []  # indices in parsed where value is stored
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                parsed.append(("blank", line))
                continue
            if stripped.startswith("#"):
                parsed.append(("comment", line))
                continue
            # Try to split by first '=' (allow empty value)
            if "=" in stripped:
                key, value = stripped.split("=", 1)
                key = key.rstrip()
                value = value.lstrip()
                # Keep original line for reconstruction
                parsed.append(("kv", line))
                kv_values.append(value)
                kv_indices.append(len(parsed) - 1)
            else:
                # malformed line, treat as comment
                parsed.append(("comment", line))

        # Load existing translation if any (for append/skip modes)
        existing_values: dict[str, str] = {}
        if os.path.exists(target_path):
            try:
                with open(target_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.rstrip("\n")
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            existing_values[k.strip()] = v.lstrip()
            except OSError:
                pass

        # Build list of indices to translate
        translate_indices: list[int] = []
        for idx, (orig_val, _) in enumerate(zip(kv_values, kv_indices)):
            key_line = parsed[kv_indices[idx]][1]  # original line
            # Extract key from line (before first '=')
            if "=" not in key_line:
                continue
            key = key_line.split("=", 1)[0].rstrip()
            existing = existing_values.get(key)
            if mode == "append" and existing and existing.strip() and existing != kv_values[idx]:
                # Already has a different (presumably translated) value --
                # keep it instead of the English kv_values[idx]. Rebuild
                # below runs whenever ANY key in this file needs
                # translating, and without this it would silently revert
                # this already-translated key back to English.
                kv_values[idx] = existing
                continue
            if mode == "append" and existing and existing.strip() == kv_values[idx]:
                # already same translation, skip
                continue
            if mode == "skip":
                if existing and already_translated(existing, target_lang["regex"]):
                    # Same reasoning as the append branch above: preserve
                    # the existing translated value across a rebuild
                    # triggered by other new/untranslated keys.
                    kv_values[idx] = existing
                    continue
            # Determine if value needs translation (not already translated)
            # We'll rely on the fact that we haven't replaced it yet.
            translate_indices.append(idx)

        if not translate_indices:
            # Nothing to translate; just copy if needed. Prefer an existing
            # translated file over the English source, so a prior 'inplace'
            # run isn't silently overwritten with untranslated text.
            if output_mode == "resourcepack" and pack_writer:
                source_path = target_path if os.path.exists(target_path) else file_path
                with open(source_path, "rb") as f:
                    pack_writer.write(internal_path, f.read())
            elif output_mode == "inplace":
                # No change needed
                pass
            return

        # Extract values to translate
        to_translate_dict: dict[str, str] = {}
        for idx in translate_indices:
            to_translate_dict[str(idx)] = kv_values[idx]

        self.callbacks.on_log(
            f"⚡ Перевод .lang {os.path.basename(file_path)} — {len(to_translate_dict)} строк",
            "cyan",
        )
        translated = self.service.translate_dict(
            to_translate_dict,
            target_lang,
            self.callbacks,
            context=".lang файл",
            usage_label=(base_name, "Словарь"),
        )

        # Update kv_values with translations
        for idx_str, new_val in translated.items():
            idx = int(idx_str)
            if 0 <= idx < len(kv_values):
                kv_values[idx] = new_val

        # Rebuild lines
        out_lines: list[str] = []
        kv_idx = 0
        for typ, content in parsed:
            if typ == "blank":
                out_lines.append(content)  # keep original newline
            elif typ == "comment":
                out_lines.append(content)
            else:  # kv
                # Replace value while preserving original whitespace around '='?
                # Simpler: reconstruct as f"{key}={new_val}\n"
                # But we want to keep original spacing; we'll just replace the value part.
                # Find first '=' in original line
                eq_pos = content.find("=")
                if eq_pos == -1:
                    out_lines.append(content)
                    continue
                key_part = content[:eq_pos]  # includes possible trailing spaces
                # Determine new value from kv_values
                new_val = kv_values[kv_idx]
                kv_idx += 1
                # Keep whitespace after '=' as in original? We'll just use one space.
                out_lines.append(f"{key_part.rstrip()}={new_val}\n")

        # Write output
        if output_mode == "resourcepack" and pack_writer:
            try:
                pack_writer.write(internal_path, "".join(out_lines).encode("utf-8"))
            except OSError as exc:
                self.callbacks.on_log(
                    f"❌ Ошибка записи .lang в ресурспак {internal_path}: {exc}", "red"
                )
        elif output_mode == "inplace":
            try:
                with open(target_path, "w", encoding="utf-8") as f:
                    f.writelines(out_lines)
            except OSError as exc:
                self.callbacks.on_log(
                    f"❌ Ошибка записи .lang файла {target_path}: {exc}", "red"
                )
        else:
            pass