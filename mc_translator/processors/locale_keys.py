
from mc_translator.text_processing import already_translated, is_technical_term, looks_like_source_language


def collect_lang_keys_to_translate(
    en_data: dict,
    tr_data: dict,
    mode: str,
    target_regex: str,
) -> dict[str, str]:
    """Return lang file keys that still need translation."""
    pending: dict[str, str] = {}
    for key, value in en_data.items():
        if not isinstance(value, str) or not value.strip():
            continue
        if is_technical_term(value):
            continue
        if not looks_like_source_language(value):
            continue

        existing = tr_data.get(key) if isinstance(tr_data.get(key), str) else ""
        if mode == "append":
            if existing.strip() and existing != value:
                continue
            pending[key] = value
        elif mode == "skip":
            if existing.strip() and already_translated(existing, target_regex):
                continue
            pending[key] = value
        else:  # force
            pending[key] = value
    return pending


def count_translatable_lang_entries(en_data: dict) -> int:
    return sum(
        1
        for v in en_data.values()
        if isinstance(v, str) and looks_like_source_language(v) and not is_technical_term(v)
    )
