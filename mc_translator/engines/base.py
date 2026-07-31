from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class EngineItem:
    key: str
    original: str
    masked: str
    mapping: dict[str, str] = field(default_factory=dict)


@dataclass
class EngineCallbacks:
    should_run: Callable[[], bool]
    wait_if_paused: Callable[[], None]
    on_log: Callable[[str, str], None]  # message, color_tag
    on_status: Callable[[str], None]
    # source, translated, engine, language_label -- fired once per freshly
    # translated string (not cache hits) so the GUI can render a live,
    # editable line in the log while a run is still in progress. Optional/
    # defaulted so the one place that builds EngineCallbacks is the only
    # thing that needs to change.
    on_translation: Callable[[str, str, str, str], None] | None = None


class TranslationEngine(ABC):
    @abstractmethod
    def translate_batch(
        self,
        items: dict[str, EngineItem],
        target_lang: dict,
        callbacks: EngineCallbacks,
    ) -> dict[str, str]:
        """Return key -> translated text (may omit keys on failure)."""
