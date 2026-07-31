from __future__ import annotations

import json
import os

from mc_translator.engines.base import EngineCallbacks
from mc_translator.engines.service import TranslationService
from mc_translator.json_utils import load_lenient_json
from mc_translator.output.pack_writer import PackWriter
from mc_translator.processors.locale_keys import collect_lang_keys_to_translate, count_translatable_lang_entries
from mc_translator.processors.resourcepack_path import compute_target_paths


class LooseJsonProcessor:
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
        mc_dir: str,
        *,
        target_lang: dict,
        mode: str,
        output_mode: str,
        pack_writer: PackWriter | None,
    ) -> None:
        _, tr_disk, tr_internal = compute_target_paths(file_path, target_lang["file"], mc_dir)

        try:
            with open(file_path, encoding="utf-8") as f:
                en_data = load_lenient_json(f.read().encode("utf-8"))
            tr_data = {}
            if os.path.exists(tr_disk):
                with open(tr_disk, encoding="utf-8") as f:
                    tr_data = load_lenient_json(f.read().encode("utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            self.callbacks.on_log(f"❌ Ошибка словаря {file_path}: {exc}", "red")
            return

        pending = collect_lang_keys_to_translate(en_data, tr_data, mode, target_lang["regex"])
        total = count_translatable_lang_entries(en_data)
        label = "Словарь: " + os.path.basename(os.path.dirname(os.path.dirname(file_path)))

        if mode == "skip" and total and (total - len(pending)) >= total * 0.9:
            if output_mode == "resourcepack" and pack_writer and os.path.exists(tr_disk):
                with open(tr_disk, "rb") as f:
                    pack_writer.write(tr_internal, f.read())
            return

        merged = en_data.copy()
        for k, v in tr_data.items():
            if k in merged:
                merged[k] = v

        if pending:
            self.callbacks.on_log(f"⚡ Перевод {label} — {len(pending)} строк", "cyan")
            translated = self.service.translate_dict(
                pending,
                target_lang,
                self.callbacks,
                context="Локализация Квестов/Скриптов",
                usage_label=(label, "Квесты/Скрипты"),
            )
            merged.update(translated)

        payload = json.dumps(merged, ensure_ascii=False, indent=2).encode("utf-8")
        if output_mode == "resourcepack" and pack_writer:
            pack_writer.write(tr_internal, payload)
        elif output_mode == "inplace":
            with open(tr_disk, "wb") as f:
                f.write(payload)