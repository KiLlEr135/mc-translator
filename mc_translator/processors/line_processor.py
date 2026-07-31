"""
Shared line-based translator for config/batch/mcfunction/plain-text formats.
Each concrete format (see cfg.py, bat.py, mcfunction.py, text.py) only differs
in which lines are candidates for translation and the user-facing label/context
used for logging — everything else (target-path resolution, existing-translation
lookup, mode handling, output writing) is identical and lives here once.
"""
from __future__ import annotations

import os
from collections.abc import Callable

from mc_translator.engines.base import EngineCallbacks
from mc_translator.engines.service import TranslationService
from mc_translator.processors.resourcepack_path import compute_target_paths
from mc_translator.text_processing import (
    already_translated,
    is_technical_term,
    looks_like_source_language,
)

CandidateFn = Callable[[list[str]], set[int]]


class LineProcessor:
    def __init__(
        self,
        service: TranslationService,
        state: "JobState",
        callbacks: EngineCallbacks,
        *,
        candidate_lines: CandidateFn,
        log_label: str,
        context_label: str,
    ) -> None:
        self.service = service
        self.state = state
        self.callbacks = callbacks
        self._candidate_lines = candidate_lines
        self.log_label = log_label
        self.context_label = context_label

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
        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError as exc:
            self.callbacks.on_log(f"❌ Ошибка чтения {file_path}: {exc}", "red")
            return

        # Determine target path for output
        base_name = os.path.basename(file_path)
        target_name, target_path, internal_path = compute_target_paths(file_path, target_lang["file"], mc_dir)

        candidates = self._candidate_lines(lines)

        out_lines: list[str] = []
        to_translate: list[tuple[int, str]] = []
        for idx, line in enumerate(lines):
            content = line.rstrip("\n\r")
            stripped = content.strip()
            if not stripped or idx not in candidates:
                out_lines.append(line)
                continue
            if not looks_like_source_language(content):
                out_lines.append(line)
                continue
            if is_technical_term(content):
                out_lines.append(line)
                continue
            out_lines.append(line)  # placeholder
            to_translate.append((idx, content))

        # Load existing translation if target file exists (for append/skip modes)
        existing_lines: list[str] = []
        if os.path.exists(target_path):
            try:
                with open(target_path, encoding="utf-8") as f:
                    existing_lines = f.readlines()
            except OSError:
                existing_lines = []
        # The target file is structurally identical to the source (same
        # candidate-line positions), so "already translated?" must be a
        # POSITIONAL lookup -- existing_lines[idx] is the translation of
        # lines[idx]. The previous code compared the English `original`
        # against the *set* of already-TRANSLATED lines, which by
        # definition never matched -- append/skip silently degraded to
        # "translate everything, every run" (safe for output correctness
        # since the file is always rebuilt from source, but wasted API
        # calls/time and discarded any manual edits to the sidecar between
        # runs). Only trust the positional lookup if the existing file has
        # the same line count as the source; a shape change (mod update
        # added/removed lines) falls back to translating every candidate,
        # same as before this fix.
        existing_aligned = len(existing_lines) == len(lines)

        translate_indices: list[int] = []
        for line_idx, original in to_translate:
            existing_line = existing_lines[line_idx].rstrip("\n\r") if existing_aligned else None
            if mode == "append":
                if existing_line is not None and existing_line != original:
                    # Already translated (or manually edited) -- keep it as
                    # the output for this line instead of the English
                    # placeholder, and don't re-queue it.
                    out_lines[line_idx] = existing_lines[line_idx]
                    continue
                translate_indices.append(line_idx)
            elif mode == "skip":
                if existing_line is not None and already_translated(existing_line, target_lang["regex"]):
                    out_lines[line_idx] = existing_lines[line_idx]
                    continue
                translate_indices.append(line_idx)
            else:  # force
                translate_indices.append(line_idx)

        if not translate_indices:
            # Nothing to translate; copy if needed. Prefer an existing
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

        self.callbacks.on_log(
            f"⚡ Перевод {self.log_label} {os.path.basename(file_path)} — {len(translate_indices)} строк",
            "cyan",
        )
        # Short badge for the review screen's "kind" column -- matches the
        # vocabulary mc_translator/processors/analyzer.py already uses
        # (icons = {"cfg": "CFG", "mcfunction": "MCF", "text": "TXT", ...}).
        kind_badge = os.path.splitext(base_name)[1].lstrip(".").upper() or "TXT"

        to_translate_dict = {f"line::{idx}": lines[idx].rstrip("\n\r") for idx in translate_indices}
        translated = self.service.translate_dict(
            to_translate_dict,
            target_lang,
            self.callbacks,
            context=f"{self.context_label} {os.path.basename(file_path)}",
            usage_label=(os.path.basename(file_path), kind_badge),
        )
        for fake_key, translated_text in translated.items():
            idx = int(fake_key.split("::")[1])
            original_line = lines[idx]
            if original_line.endswith("\r\n"):
                ending = "\r\n"
            elif original_line.endswith("\n"):
                ending = "\n"
            else:
                ending = ""
            out_lines[idx] = translated_text + ending

        if not self.state.should_run():
            # Stopped mid-translation: translate_dict returned only a
            # partial result, so out_lines is now a mix of fresh
            # translations and English placeholders for the untranslated
            # tail (in force mode especially, since -- unlike append/skip --
            # it never pre-seeds already-translated lines). Writing that
            # would clobber a previously fully-translated output file with
            # a half-English one. Leave the existing output untouched; a
            # future run picks up the freshly-cached lines instantly.
            return

        if output_mode == "resourcepack" and pack_writer:
            try:
                payload = "".join(out_lines).encode("utf-8")
                pack_writer.write(internal_path, payload)
            except OSError as exc:
                self.callbacks.on_log(
                    f"❌ Ошибка записи в ресурспак {target_name}: {exc}", "red"
                )
        elif output_mode == "inplace":
            try:
                with open(target_path, "w", encoding="utf-8") as f:
                    f.writelines(out_lines)
            except OSError as exc:
                self.callbacks.on_log(f"❌ Ошибка записи {target_path}: {exc}", "red")


def cfg_candidates(lines: list[str]) -> set[int]:
    result: set[int] = set()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            result.add(idx)
    return result


def bat_candidates(lines: list[str]) -> set[int]:
    result: set[int] = set()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower().startswith("rem") or stripped.startswith("::"):
            result.add(idx)
    return result


def mcfunction_candidates(lines: list[str]) -> set[int]:
    return {idx for idx, line in enumerate(lines) if line.strip().startswith("#")}


def text_candidates(lines: list[str]) -> set[int]:
    """Every line outside a fenced (```) code block is a candidate."""
    result: set[int] = set()
    in_code_block = False
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        result.add(idx)
    return result
