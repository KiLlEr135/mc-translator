"""
Processor for plain text files (.txt, .md).
Translates lines that are not inside fenced (```) code blocks and appear to
be natural language. See line_processor.py for the shared implementation.
"""
from __future__ import annotations

from mc_translator.engines.base import EngineCallbacks
from mc_translator.engines.service import TranslationService
from mc_translator.processors.line_processor import LineProcessor, text_candidates


class TextProcessor(LineProcessor):
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
            candidate_lines=text_candidates,
            log_label="текста",
            context_label="текстовый файл",
        )
