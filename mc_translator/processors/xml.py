"""
Processor for XML files (.xml).
Translates text content while leaving tags and attribute values unchanged.
Simple implementation: translates lines that are not XML tags.
"""
from __future__ import annotations

import os

try:
    from lxml import etree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from io import BytesIO

from mc_translator.engines.base import EngineCallbacks
from mc_translator.engines.service import TranslationService
from mc_translator.processors.resourcepack_path import compute_target_paths
from mc_translator.text_processing import (
    already_translated,
    is_technical_term,
    looks_like_source_language,
)


def _needs_translation(key: str, value: str, existing_map: dict, mode: str, target_lang: dict) -> bool:
    if not (value.strip() and looks_like_source_language(value) and not is_technical_term(value)):
        return False
    if mode == "force":
        return True
    existing = existing_map.get(key, "")
    if mode == "append":
        return not (existing and existing.strip())
    if mode == "skip":
        return not (existing and existing.strip() and already_translated(existing, target_lang["regex"]))
    return True


def collect_translatable(
    root, translatable_attrs: set[str], mode: str, target_lang: dict, existing_map: dict | None = None
) -> dict:
    """Walk an XML tree, collecting translatable text nodes, tails and attributes.

    Keys are stable, purely positional child-index paths from the root (e.g.
    "0/1/0@title"), NOT id(element). id() used to be used here, which happens
    to be stable for the root (a Python variable keeps it alive across both
    this function and the later apply_translated call), but is NOT stable for
    any child: `for child in element` (lxml in particular) can return a fresh
    proxy object -- with a different id() -- on each separate iteration, since
    nothing holds a persistent reference to the child in between. That made
    apply_translated's id()-keyed lookup silently miss every non-root
    text/attribute/tail: the string still got translated and cached (wasting
    the API call), but the translation never made it into the output XML.

    `existing_map` (from collect_all_text() on a previously-translated sidecar
    file, if any) drives append/skip mode -- see _needs_translation. Without
    it (the default), every mode behaves like "force", matching this
    function's original behavior for callers that don't track prior output
    (e.g. jar.py's _process_book_xml).
    """
    existing_map = existing_map or {}
    to_translate: dict = {}

    def _walk(element, path: str) -> None:
        if element.text and element.text.strip():
            txt = element.text.strip()
            if _needs_translation(path, txt, existing_map, mode, target_lang):
                to_translate[path] = txt

        for attr in element.attrib:
            if attr.lower() in translatable_attrs:
                val = element.attrib[attr]
                key = f"{path}@{attr}"
                if _needs_translation(key, val, existing_map, mode, target_lang):
                    to_translate[key] = val

        for idx, child in enumerate(element):
            child_path = f"{path}/{idx}"
            _walk(child, child_path)
            if child.tail and child.tail.strip():
                txt = child.tail.strip()
                key = f"{child_path}#tail"
                if _needs_translation(key, txt, existing_map, mode, target_lang):
                    to_translate[key] = txt

    _walk(root, "0")
    return to_translate


def collect_all_text(root) -> dict:
    """Flatten an already-translated XML tree's text/attribute/tail values
    using the same positional-path keying as collect_translatable, for
    existing-translation lookups (append/skip mode). Positional, so this
    only aligns correctly when the two trees have matching structure -- true
    for the common case of re-running against the same source file."""
    out: dict = {}

    def _walk(element, path: str) -> None:
        if element.text and element.text.strip():
            out[path] = element.text.strip()
        for attr, val in element.attrib.items():
            if val.strip():
                out[f"{path}@{attr}"] = val
        for idx, child in enumerate(element):
            child_path = f"{path}/{idx}"
            _walk(child, child_path)
            if child.tail and child.tail.strip():
                out[f"{child_path}#tail"] = child.tail.strip()

    _walk(root, "0")
    return out


def apply_translated(root, translated: dict) -> None:
    """Write translated text nodes, tails and attributes back into an XML tree.
    Must use the exact same positional-path keying as collect_translatable.
    Leading/trailing whitespace around the ORIGINAL text/tail is preserved
    around the translated core -- collect_translatable strips before sending
    text to translate_dict (which would collapse/strip it anyway via the
    format-protection shield), so without this the whitespace between an
    inline tag and surrounding prose was lost on every translated run."""

    def _ws(original: str) -> tuple[str, str]:
        if not original or not original.strip():
            return "", ""
        return original[: len(original) - len(original.lstrip())], original[len(original.rstrip()):]

    def _walk(element, path: str) -> None:
        if path in translated:
            lead, trail = _ws(element.text or "")
            element.text = lead + translated[path] + trail

        for attr in element.attrib:
            attr_key = f"{path}@{attr}"
            if attr_key in translated:
                element.attrib[attr] = translated[attr_key]

        for idx, child in enumerate(element):
            child_path = f"{path}/{idx}"
            _walk(child, child_path)
            tail_key = f"{child_path}#tail"
            if tail_key in translated:
                lead, trail = _ws(child.tail or "")
                child.tail = lead + translated[tail_key] + trail

    _walk(root, "0")


class XmlProcessor:
    def __init__(
        self,
        service: TranslationService,
        state: "JobState",
        callbacks: EngineCallbacks,
    ) -> None:
        self.service = service
        self.state = state
        self.callbacks = callbacks
        self.translatable_attrs = {"title", "name", "header", "label", "text"}

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
            tree = ET.parse(file_path)
            root = tree.getroot()
        except Exception as exc:
            self.callbacks.on_log(f"❌ Ошибка XML {file_path}: {exc}", "red")
            return

        # Determine target path for output -- a language-suffixed sidecar,
        # matching every other loose-file processor (nbt/yaml_toml/
        # line_processor). Writing straight back over file_path (the old
        # behavior) destroyed the English original with no backup and made
        # append/skip meaningless on re-runs (there was no separate,
        # already-translated file to diff against).
        target_name, target_path, internal_path = compute_target_paths(file_path, target_lang["file"], mc_dir)

        existing_map: dict = {}
        if os.path.exists(target_path):
            try:
                existing_root = ET.parse(target_path).getroot()
                existing_map = collect_all_text(existing_root)
            except Exception:
                existing_map = {}

        # Must run BEFORE seeding root with existing_map below: collect_
        # translatable's looks_like_source_language check needs to see the
        # true English values from `file_path`. Seeding first would make
        # every already-translated node look like target-language text and
        # get excluded regardless of mode, silently defeating "force" (and
        # "skip"/"append" re-checks) on any node that already has a prior
        # translation.
        to_translate = collect_translatable(root, self.translatable_attrs, mode, target_lang, existing_map)

        if existing_map:
            apply_translated(root, existing_map)

        if not to_translate:
            # Nothing to translate; copy if needed. Prefer an existing
            # translated file over the English source, so a prior 'inplace'
            # run isn't silently overwritten with untranslated text.
            if output_mode == "resourcepack" and pack_writer:
                source_path = target_path if os.path.exists(target_path) else file_path
                with open(source_path, "rb") as f:
                    pack_writer.write(internal_path, f.read())
            elif output_mode == "inplace":
                pass
            return

        self.callbacks.on_log(f"⚡ Перевод XML {os.path.basename(file_path)} — {len(to_translate)} строк", "cyan")
        translated = self.service.translate_dict(
            to_translate,
            target_lang,
            self.callbacks,
            context="XML",
            usage_label=(os.path.basename(file_path), "XML"),
        )

        apply_translated(root, translated)

        # Output
        out = BytesIO()
        if hasattr(tree, 'write'):
            tree.write(out, encoding="utf-8", xml_declaration=True)
        else: # lxml etree or root
            out.write(ET.tostring(root, encoding="utf-8", xml_declaration=True))
        payload = out.getvalue()

        if output_mode == "resourcepack" and pack_writer:
            pack_writer.write(internal_path, payload)
        else:
            try:
                with open(target_path, "wb") as f:
                    f.write(payload)
            except OSError as exc:
                self.callbacks.on_log(f"❌ Ошибка записи {target_path}: {exc}", "red")