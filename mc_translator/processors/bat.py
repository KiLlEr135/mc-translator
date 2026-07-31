"""
Processor for Windows batch files (.bat).
Treats each line as plain text and translates comment lines (rem/::) that
look like source language. See line_processor.py for the shared implementation.
Note: Translating batch syntax may break functionality; use with caution.
"""
from __future__ import annotations

from mc_translator.engines.base import EngineCallbacks
from mc_translator.engines.service import TranslationService
from mc_translator.processors.line_processor import LineProcessor, bat_candidates


class BatProcessor(LineProcessor):
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
            candidate_lines=bat_candidates,
            log_label="батч-файла",
            context_label="батч файл",
        )
