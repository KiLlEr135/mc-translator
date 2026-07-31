"""
Processor for SNBT files (.snbt).
Handles raw SNBT (Named Binary Tag in text format) used by FTB Quests.
"""
from __future__ import annotations

import os
import shutil

from mc_translator.engines.base import EngineCallbacks
from mc_translator.engines.service import TranslationService
from mc_translator.processors.snbt_extract import apply_snbt_translations, extract_snbt_strings


class SnbtProcessor:
    def __init__(
        self,
        service: TranslationService,
        state: "JobState",
        callbacks: EngineCallbacks,
    ) -> None:
        self.service = service
        self.state = state
        self.callbacks = callbacks

    def process(self, file_path: str, *, target_lang: dict, mode: str) -> None:
        backup = file_path + ".bak"
        if not os.path.exists(backup):
            shutil.copy2(file_path, backup)
        elif mode == "force":
            self._refresh_stale_backup(file_path, backup)

        source_path = file_path if mode != "force" else backup
        target_regex = target_lang["regex"]

        try:
            with open(source_path, encoding="utf-8") as f:
                content = f.read()
        except OSError as exc:
            self.callbacks.on_log(f"❌ Ошибка чтения {file_path}: {exc}", "red")
            return

        skip_regex = None if mode == "force" else target_regex
        strings = extract_snbt_strings(content, skip_translated_regex=skip_regex)

        if mode == "skip":
            total = len(extract_snbt_strings(content, require_source_language=False))
            if total and (total - len(strings)) >= total * 0.9:
                return

        if not strings:
            return

        name = os.path.basename(file_path)
        self.callbacks.on_log(f"⚡ Перевод {name} [Квесты] — {len(strings)} строк", "yellow")
        chunk = {str(i): s for i, s in enumerate(strings)}
        translated = self.service.translate_dict(
            chunk, target_lang, self.callbacks, context=name, usage_label=(name, "Квесты")
        )
        mapping = {strings[i]: translated.get(str(i), strings[i]) for i in range(len(strings))}

        new_content = apply_snbt_translations(content, mapping)
        temp_path = file_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        os.replace(temp_path, file_path)

    def _refresh_stale_backup(self, file_path: str, backup: str) -> None:
        """"force" mode rebuilds the whole file from the one-time .bak
        snapshot so it always retranslates from pristine English rather than
        whatever partially-translated state the live file is in. If the live
        file has since gained new fields (e.g. a modpack update added a new
        quest) that the backup never saw, rebuilding from the stale backup
        would silently drop them from the live file on write-back.

        Field-occurrence count survives in-place translation of existing
        fields unchanged (same field, new value), so a live/backup count
        mismatch is a reliable signal of real structural growth rather than
        translation having merely happened -- refresh only fires on that
        signal, so force mode still redoes from true original English in the
        common (no structural change) case."""
        try:
            with open(backup, encoding="utf-8") as f:
                backup_count = len(extract_snbt_strings(f.read()))
            with open(file_path, encoding="utf-8") as f:
                live_count = len(extract_snbt_strings(f.read()))
        except OSError:
            return
        if live_count > backup_count:
            shutil.copy2(file_path, backup)