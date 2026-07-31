"""
Processor for config files (.cfg).
Treats each line as plain text and translates comment lines (# or //) that
look like source language. See line_processor.py for the shared implementation.
"""
from __future__ import annotations

from mc_translator.engines.base import EngineCallbacks
from mc_translator.engines.service import TranslationService
from mc_translator.processors.line_processor import LineProcessor, cfg_candidates


class CfgProcessor(LineProcessor):
    def __init__(
        self,
        service: TranslationService,
        state: "JobState",
        callbacks: EngineCallbacks,
    ) -> None:
        super().__init__(
            service,
            state,
            callbacks,
            candidate_lines=cfg_candidates,
            log_label="конфигурационного файла",
            context_label="конфиг",
        )
