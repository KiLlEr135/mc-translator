"""Shared Minecraft 1.12-and-earlier lang-file conventions.

Pre-1.13 mods commonly ship assets/<modid>/lang/en_US.lang (uppercase region
code, .lang extension) instead of the modern assets/<modid>/lang/en_us.json.
This is the single place that knows how to detect a legacy version and convert
a modern lang code/filename to the legacy convention, so
mc_translator/processors/lang.py and mc_translator/output/pack_writer.py don't each carry
their own (and potentially drifting) copy of this logic.
"""
from __future__ import annotations


def is_legacy_version(mc_version: str) -> bool:
    """True for Minecraft 1.12.x and earlier (1.7-1.12), where lang files use
    the underscore+uppercase-region convention (ru_RU) and a .lang extension."""
    try:
        major, minor = (int(x) for x in mc_version.split(".")[:2])
        return major == 1 and minor <= 12
    except (ValueError, IndexError):
        return False


def legacy_lang_code(code: str) -> str:
    """'ru_ru' -> 'ru_RU'. Leaves anything not shaped like lang_region unchanged."""
    parts = code.split("_")
    if len(parts) == 2:
        return f"{parts[0].lower()}_{parts[1].upper()}"
    return code


def legacy_lang_filename(filename: str) -> str:
    """Apply the legacy region-case convention to a bare filename, e.g.
    'ru_ru.json' -> 'ru_RU.json'. Only the stem before the extension is
    affected; a stem without a two-part underscore shape is left unchanged."""
    if "." in filename:
        stem, ext = filename.rsplit(".", 1)
        return f"{legacy_lang_code(stem)}.{ext}"
    return legacy_lang_code(filename)
