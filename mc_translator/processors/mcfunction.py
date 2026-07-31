"""
Processor for Minecraft function files (.mcfunction).
Treats each line as plain text and translates comment lines (#) that look
like source language. See line_processor.py for the shared implementation.
Note: Translating command syntax may break functionality; use with caution.
"""
from __future__ import annotations

from mc_translator.engines.base import EngineCallbacks
from mc_translator.engines.service import TranslationService
from mc_translator.processors.line_processor import LineProcessor, mcfunction_candidates


class McfunctionProcessor(LineProcessor):
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
            candidate_lines=mcfunction_candidates,
            log_label="функции",
            context_label="minecraft функция",
        )
