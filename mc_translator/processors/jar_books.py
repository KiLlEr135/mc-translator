"""Patchouli/GuideME in-jar book translation (JSON/MD/XML books) --
split out of jar.py purely to keep that file under the project's 500-line
guideline. BookProcessorMixin is mixed into JarProcessor (jar.py), which
still owns self.service/self.state/self.callbacks and the shared
_copy_existing helper this mixin calls; this is pure code motion, not a
behavior change."""
from __future__ import annotations

import json
import re

from mc_translator.json_utils import (
    apply_translations_by_path,
    iter_translatable_strings,
    key_to_path,
    load_lenient_json,
    path_to_key,
    set_at_path,
)
from mc_translator.processors.guide_md import guide_target_path, translate_guide_markdown
from mc_translator.processors.xml import apply_translated, collect_all_text, collect_translatable
from mc_translator.text_processing import is_technical_term, looks_like_source_language


class BookProcessorMixin:
    def _process_book_json(
        self, zin, zout, item, locale_files, target_lang, mode,
        output_mode, pack_writer, mod_name, written_inplace,
    ) -> bool:
        tr_path = re.sub(r"/en_us/", f"/{target_lang['file']}/", item.filename, flags=re.IGNORECASE)
        tr_key = tr_path.lower()

        # Own-path entries (RESEARCH_PATH_MARKERS books with no "/en_us/"
        # segment -- tr_path == item.filename here, see writes_own_path in
        # jar.py's outer loop) get overwritten in place on every inplace
        # run, so the true English source would otherwise be gone after the
        # first run: "force" mode could never redo it, and a bad/corrupted
        # cached translation (see project memory on AI translation quality
        # issues) could never be fixed. Snapshot the original English into
        # an inert "<name>.bak" zip entry, once -- the same one-time-
        # snapshot pattern snbt.py already uses for its sibling .bak file on
        # disk, adapted to a jar-internal entry no mod loader ever looks
        # for. Only in inplace mode: in resourcepack mode the source jar is
        # never modified at all, so zin.read(item) is always genuinely
        # English already and no backup is needed.
        is_own_path = "/en_us/" not in item.filename.lower()
        backup_active = is_own_path and output_mode == "inplace"
        bak_name = item.filename + ".bak"
        bak_info = None
        if backup_active:
            try:
                bak_info = zin.getinfo(bak_name)
            except KeyError:
                bak_info = None

        try:
            en_raw = zin.read(bak_info) if bak_info is not None else zin.read(item)
            en_data = load_lenient_json(en_raw)
        except (json.JSONDecodeError, OSError):
            return False

        tr_data = {}
        if tr_key in locale_files:
            try:
                tr_data = load_lenient_json(zin.read(locale_files[tr_key]))
            except (json.JSONDecodeError, OSError):
                pass
        elif backup_active and bak_info is not None:
            # A backup already exists, so the LIVE entry (item.filename) now
            # holds this book's existing translation from a prior run --
            # read it as tr_data instead of re-translating everything, and
            # so the seed-back loop below can preserve it.
            try:
                tr_data = load_lenient_json(zin.read(item))
            except (json.JSONDecodeError, OSError):
                tr_data = {}

        en_map = {path_to_key(p): s for p, s in iter_translatable_strings(en_data) if s.strip()}
        tr_map = {path_to_key(p): s for p, s in iter_translatable_strings(tr_data)} if tr_data else {}

        pending: dict[str, str] = {}
        for path_key, source in en_map.items():
            if is_technical_term(source):
                continue
            if not looks_like_source_language(source):
                continue
            existing = tr_map.get(path_key, "")
            if mode == "append" and existing.strip() and existing != source:
                continue
            if mode == "append" and existing.strip() == source:
                continue
            pending[path_key] = source

        if not en_map:
            return False

        if mode == "skip" and len(pending) <= len(en_map) * 0.1:
            return self._copy_existing(zin, locale_files, tr_key, tr_path, output_mode, pack_writer, en_data, tr_data, mode)

        # Seed en_data with the already-translated values from tr_map before
        # writing. Without this, any path_key excluded from `pending` above
        # (because it already has a real, different-from-English translation)
        # was never applied to en_data -- the final payload was built purely
        # from the English source plus this run's freshly-translated
        # `pending` keys, silently reverting every previously-completed
        # translation back to English on each subsequent append-mode run.
        # tr_map's paths come from tr_data (a different document than
        # en_data), so a per-key structural mismatch is tolerated rather
        # than aborting the whole file.
        for path_key, value in tr_map.items():
            if path_key in pending or not value.strip():
                continue
            try:
                set_at_path(en_data, key_to_path(path_key), value)
            except (KeyError, IndexError, TypeError):
                pass

        if pending:
            self.callbacks.on_log(f"⚡ Перевод {mod_name} [Книга JSON] — {len(pending)} строк", "magenta")
            translated = self.service.translate_dict(
                pending, target_lang, self.callbacks, context=mod_name, usage_label=(mod_name, "Книга(JSON)")
            )
            apply_translations_by_path(en_data, translated)

        if backup_active and bak_info is None and zout is not None and bak_name not in written_inplace:
            # One-time snapshot -- only when no backup exists yet, so we
            # never back up already-translated content as if it were
            # pristine English.
            zout.writestr(bak_name, en_raw)
            written_inplace.add(bak_name)

        payload = json.dumps(en_data, ensure_ascii=False, indent=2).encode("utf-8")
        if output_mode == "resourcepack" and pack_writer:
            pack_writer.write(tr_path, payload)
            return True
        if zout:
            zout.writestr(tr_path, payload)
            written_inplace.add(tr_path)
            return True
        return False

    def _process_book_md(
        self, zin, zout, item, locale_files, target_lang, mode,
        output_mode, pack_writer, mod_name, written_inplace,
    ) -> bool:
        fl = item.filename.lower()
        tr_path = guide_target_path(fl, target_lang["file"])
        if tr_path is None:
            # Already a localized variant (this run's own prior output, or
            # another mod-shipped locale) -- never a translation source; the
            # generic copy-through in process() preserves it untouched.
            return False
        tr_key = tr_path

        try:
            en_text = zin.read(item).decode("utf-8-sig", errors="replace")
        except OSError:
            return False

        tr_text = ""
        if tr_key in locale_files:
            try:
                tr_text = zin.read(locale_files[tr_key]).decode("utf-8-sig", errors="replace")
            except OSError:
                tr_text = ""

        def _translate(pending: dict[str, str]) -> dict[str, str]:
            self.callbacks.on_log(
                f"⚡ Перевод {mod_name} [Гайд MD] — {len(pending)} строк", "magenta"
            )
            translated = self.service.translate_dict(
                pending, target_lang, self.callbacks, context=mod_name, usage_label=(mod_name, "Гайд(MD)")
            )
            return translated

        new_text = translate_guide_markdown(en_text, tr_text, mode, target_lang, _translate)
        payload = new_text.encode("utf-8")

        if output_mode == "resourcepack" and pack_writer:
            pack_writer.write(tr_path, payload)
            return True
        if zout:
            zout.writestr(tr_path, payload)
            written_inplace.add(tr_path)
            return True
        return False

    def _process_book_xml(
        self, zin, zout, item, locale_files, target_lang, mode,
        output_mode, pack_writer, mod_name, written_inplace,
    ) -> bool:
        fl = item.filename.lower()
        tr_path = guide_target_path(fl, target_lang["file"])
        if tr_path is None:
            # Already a localized variant (this run's own prior output, or
            # another mod-shipped locale) -- never a translation source; the
            # generic copy-through in process() preserves it untouched.
            return False
        tr_key = tr_path

        try:
            # Raw bytes, not decoded to str: lxml rejects a str whose
            # declaration names an encoding, and ET.tostring(...,
            # xml_declaration=True) below always emits one -- re-parsing this
            # method's own prior output (existing_map below) would otherwise
            # always fail and silently defeat the seed-back this fix depends
            # on. Matches xml.py's XmlProcessor, which avoids the same trap
            # via ET.parse(path).
            xml_raw = zin.read(item)
        except OSError:
            return False

        try:
            from lxml import etree as ET
        except ImportError:
            import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(xml_raw)
        except Exception as exc:
            self.callbacks.on_log(f"❌ Ошибка XML {fl}: {exc}", "red")
            return False

        # Parsed into existing_map so append/skip know what's already
        # translated, AND so those values get seeded back into root before
        # any fresh translation (same regression class fixed for
        # nbt/yaml_toml/xml.py -- without this, re-runs reverted translated
        # text/attributes back to English).
        existing_map: dict = {}
        if tr_key in locale_files:
            try:
                existing_root = ET.fromstring(zin.read(locale_files[tr_key]))
                existing_map = collect_all_text(existing_root)
            except Exception:
                existing_map = {}

        # Must run BEFORE seeding root with existing_map below -- same
        # ordering fix as xml.py's XmlProcessor: collect_translatable's
        # looks_like_source_language check needs to see the true English
        # values from `xml_raw`, not whatever a prior run already seeded in,
        # or "force" (and skip/append re-checks) on any already-translated
        # node is silently defeated regardless of mode.
        translatable_attrs = {"title", "name", "header", "label", "text", "desc", "description"}
        to_translate = collect_translatable(root, translatable_attrs, mode, target_lang, existing_map)

        if existing_map:
            apply_translated(root, existing_map)

        if not to_translate:
            # Nothing NEW to translate. If there was a prior translation,
            # root was already seeded above -- serialize that. Otherwise
            # (first run, nothing translatable at all) write the original
            # bytes through unchanged.
            if existing_map:
                payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            else:
                payload = xml_raw
            if output_mode == "resourcepack" and pack_writer:
                pack_writer.write(tr_path, payload)
            elif zout:
                zout.writestr(tr_path, payload)
                written_inplace.add(tr_path)
            return False

        self.callbacks.on_log(f"⚡ Перевод {mod_name} [Книга XML] — {len(to_translate)} строк", "magenta")
        translated = self.service.translate_dict(
            to_translate, target_lang, self.callbacks, context=mod_name, usage_label=(mod_name, "Книга(XML)")
        )

        apply_translated(root, translated)

        try:
            xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True, pretty_print=True)
        except Exception:
            # pretty_print is lxml-only -- stdlib ElementTree's tostring
            # raises on the unknown kwarg, so fall back to plain output.
            xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)

        # Full internal path, not just the basename -- avoids zip-root
        # placement (unloaded by the game) and same-named-book collisions.
        if output_mode == "resourcepack" and pack_writer:
            pack_writer.write(tr_path, xml_bytes)
            return True
        elif zout:
            zout.writestr(tr_path, xml_bytes)
            written_inplace.add(tr_path)
            return True
        return False
