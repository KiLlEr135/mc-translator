import copy
import json
import os
import re
import shutil
import zipfile

from mc_translator.constants import BOOK_PATH_MARKERS, MD_PATH_MARKERS, RESEARCH_PATH_MARKERS
from mc_translator.engines.base import EngineCallbacks
from mc_translator.engines.service import TranslationService
from mc_translator.json_utils import load_lenient_json
from mc_translator.mod_names import get_mod_name
from mc_translator.processors.guide_md import is_localized_guide_path, is_target_locale_path
from mc_translator.processors.jar_books import BookProcessorMixin
from mc_translator.text_processing import (
    already_translated,
    is_technical_term,
    looks_like_source_language,
)
from mc_translator.utils.legacy_lang import legacy_lang_filename


def _parse_legacy_lang(data: bytes) -> dict[str, str]:
    """Parse pre-1.13 .lang (key=value line) bytes into a flat dict.
    Comments ('#'-prefixed) and blank lines are dropped -- unlike
    LangProcessor (loose .lang files), which preserves exact file structure
    for a single mod's own primary lang file, a jar's internal source is
    always regenerated wholesale on each run, so byte-for-byte comment
    preservation isn't needed here."""
    result: dict[str, str] = {}
    for line in data.decode("utf-8-sig", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _serialize_legacy_lang(data: dict[str, str]) -> bytes:
    lines = [f"{key}={value}" for key, value in data.items() if isinstance(value, str)]
    return ("\n".join(lines) + "\n").encode("utf-8")


class JarProcessor(BookProcessorMixin):
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
        jar_path: str,
        *,
        target_lang: dict,
        mode: str,
        output_mode: str,
        translate_mods: bool,
        translate_books: bool,
        pack_writer: object | None = None,
    ) -> None:
        if not translate_mods and not translate_books:
            return

        mod_name = get_mod_name(jar_path)
        target_file = f"{target_lang['file']}.json"
        # Pre-1.13 mods ship assets/<modid>/lang/en_US.lang (plain key=value,
        # uppercase-region convention) instead of the modern en_us.json --
        # see _process_legacy_lang_entry below. Properly cased (e.g.
        # "ru_RU.lang") since it's used as the actual output filename, not
        # just for comparisons -- callers that need a lowercase comparison
        # key lowercase it at the point of use, same as target_file/fl.
        target_legacy_lang_file = legacy_lang_filename(f"{target_lang['file']}.lang")
        temp_path = jar_path + ".temp"
        modified = False

        try:
            with zipfile.ZipFile(jar_path, "r") as zin:
                zout = (
                    zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED)
                    if output_mode == "inplace"
                    else None
                )
                written_inplace: set[str] = set()
                target_legacy_lang_file_lower = target_legacy_lang_file.lower()
                locale_files = {
                    item.filename.lower(): item
                    for item in zin.infolist()
                    if target_file in item.filename.lower()
                    or target_legacy_lang_file_lower in item.filename.lower()
                    or is_target_locale_path(item.filename.lower(), target_lang["file"])
                }

                try:
                    for item in zin.infolist():
                        if not self.state.should_run():
                            break
                        self.state.wait_if_paused()
                        fl = item.filename.lower()

                        is_book_json = fl.endswith(".json") and (
                            ("/en_us/" in fl and any(m in fl for m in BOOK_PATH_MARKERS))
                            or any(m in fl for m in RESEARCH_PATH_MARKERS)
                        ) and not is_localized_guide_path(fl)
                        # Pages already under a language subfolder (this
                        # run's own prior output, or another mod-shipped
                        # locale) are never a source -- see guide_md.py.
                        is_book_md = (
                            (fl.endswith(".md") or fl.endswith(".txt"))
                            and any(m in fl for m in MD_PATH_MARKERS)
                            and not is_localized_guide_path(fl)
                        )
                        is_book_xml = (
                            fl.endswith(".xml")
                            and any(m in fl for m in BOOK_PATH_MARKERS)
                            and not is_localized_guide_path(fl)
                        )
                        is_lang = fl.endswith("en_us.json") and not is_book_json
                        is_legacy_lang = fl.endswith("en_us.lang")

                        # A RESEARCH_PATH_MARKERS JSON book has no "/en_us/"
                        # segment to substitute, so its tr_path is this exact
                        # path (own_path); defer the raw copy-through until
                        # after the processor ran, as a fallback only, to
                        # avoid a duplicate zip entry. is_book_md/is_book_xml
                        # are NOT part of this -- guide_target_path always
                        # resolves to a distinct "_<lang>/"-or-"/<lang>/"
                        # path (or None), covered by is_target_locale_path.
                        writes_own_path = translate_books and (is_book_json and "/en_us/" not in fl)
                        own_path = item.filename if writes_own_path else None

                        if output_mode == "inplace" and zout and not writes_own_path:
                            if (
                                target_file not in fl
                                and target_legacy_lang_file_lower not in fl
                                and not is_target_locale_path(fl, target_lang["file"])
                            ):
                                # writestr(ZipInfo, ...) mutates that exact
                                # ZipInfo in place (header_offset, CRC, etc.
                                # get rewritten to this entry's position in
                                # zout). Since `item` is the SAME object zin
                                # uses internally, writing it unmodified would
                                # corrupt zin's bookkeeping for this entry --
                                # any later zin.read(item) for the same item
                                # (e.g. is_lang's own _process_lang_entry call
                                # right below) would then read from the wrong
                                # offset and raise BadZipFile. Pass a copy so
                                # only the copy gets mutated.
                                zout.writestr(copy.copy(item), zin.read(item))

                        if translate_mods and is_lang:
                            modified |= self._process_lang_entry(
                                zin, zout, item, locale_files, target_file, target_lang, mode,
                                output_mode, pack_writer, mod_name, written_inplace,
                            )
                        elif translate_mods and is_legacy_lang:
                            modified |= self._process_legacy_lang_entry(
                                zin, zout, item, locale_files, target_legacy_lang_file, target_lang, mode,
                                output_mode, pack_writer, mod_name, written_inplace,
                            )
                        elif translate_books and is_book_json:
                            modified |= self._process_book_json(
                                zin, zout, item, locale_files, target_lang, mode,
                                output_mode, pack_writer, mod_name, written_inplace,
                            )
                        elif translate_books and is_book_md:
                            modified |= self._process_book_md(
                                zin, zout, item, locale_files, target_lang, mode,
                                output_mode, pack_writer, mod_name, written_inplace,
                            )
                        elif translate_books and is_book_xml:
                            modified |= self._process_book_xml(
                                zin, zout, item, locale_files, target_lang, mode,
                                output_mode, pack_writer, mod_name, written_inplace,
                            )

                        if (
                            writes_own_path
                            and output_mode == "inplace"
                            and zout
                            and own_path not in written_inplace
                        ):
                            zout.writestr(copy.copy(item), zin.read(item))

                    if output_mode == "inplace" and zout:
                        for item in zin.infolist():
                            fl = item.filename.lower()
                            if (
                                target_file in fl
                                or target_legacy_lang_file_lower in fl
                                or is_target_locale_path(fl, target_lang["file"])
                            ) and item.filename not in written_inplace:
                                zout.writestr(copy.copy(item), zin.read(item))
                finally:
                    if zout:
                        zout.close()

            if output_mode == "inplace":
                if modified and self.state.should_run():
                    shutil.move(temp_path, jar_path)
                elif os.path.exists(temp_path):
                    os.remove(temp_path)
            elif os.path.exists(temp_path):
                os.remove(temp_path)

        except (OSError, zipfile.BadZipFile) as exc:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            self.callbacks.on_log(f"❌ Ошибка в {mod_name}: {exc}", "red")

    def _process_lang_entry(
        self, zin, zout, item, locale_files, target_file, target_lang, mode,
        output_mode, pack_writer, mod_name, written_inplace,
    ) -> bool:
        tr_path = re.sub(r"en_us\.json$", target_file, item.filename, flags=re.IGNORECASE)
        tr_key = tr_path.lower()
        try:
            en_data = load_lenient_json(zin.read(item))
        except (json.JSONDecodeError, OSError):
            return False

        tr_data = {}
        if tr_key in locale_files:
            try:
                tr_data = load_lenient_json(zin.read(locale_files[tr_key]))
            except (json.JSONDecodeError, OSError):
                tr_data = {}

        pending = self._collect_pending_translations(en_data, tr_data, mode, target_lang)
        total = self._count_translatable_entries(en_data)
        if total == 0:
            return False

        if mode == "skip" and (total - len(pending)) >= total * 0.9:
            return self._copy_existing(zin, locale_files, tr_key, tr_path, output_mode, pack_writer, en_data, tr_data, mode)

        merged = en_data.copy()
        for k, v in tr_data.items():
            if k in merged and isinstance(merged[k], str) and v:
                merged[k] = v

        if not pending:
            return self._write_lang_output(merged, tr_path, output_mode, pack_writer, zout, written_inplace, item, en_data)

        self.callbacks.on_log(f"⚡ Перевод {mod_name} [Интерфейс] — {len(pending)} строк", "cyan")
        translated = self.service.translate_dict(
            pending, target_lang, self.callbacks, context=mod_name, usage_label=(mod_name, "Интерфейс")
        )
        for key, value in translated.items():
            merged[key] = value
        return self._write_lang_output(merged, tr_path, output_mode, pack_writer, zout, written_inplace, item, en_data)

    def _process_legacy_lang_entry(
        self, zin, zout, item, locale_files, target_legacy_lang_file, target_lang, mode,
        output_mode, pack_writer, mod_name, written_inplace,
    ) -> bool:
        """Legacy (pre-1.13) counterpart of _process_lang_entry -- same
        append/skip/force merge and translate flow (via the same
        format-agnostic _collect_pending_translations/_count_translatable_
        entries helpers), but for a plain key=value .lang source (e.g.
        assets/<modid>/lang/en_US.lang) instead of a JSON one. Kept as a
        fully separate method rather than folded into _process_lang_entry:
        the JSON path is the primary, heavily-used, already-tested one, and
        this keeps it completely unaffected by the legacy addition."""
        tr_path = re.sub(r"en_us\.lang$", target_legacy_lang_file, item.filename, flags=re.IGNORECASE)
        tr_key = tr_path.lower()
        try:
            en_data = _parse_legacy_lang(zin.read(item))
        except OSError:
            return False

        tr_data = {}
        if tr_key in locale_files:
            try:
                tr_data = _parse_legacy_lang(zin.read(locale_files[tr_key]))
            except OSError:
                tr_data = {}

        pending = self._collect_pending_translations(en_data, tr_data, mode, target_lang)
        total = self._count_translatable_entries(en_data)
        if total == 0:
            return False

        if mode == "skip" and (total - len(pending)) >= total * 0.9:
            return self._copy_existing(zin, locale_files, tr_key, tr_path, output_mode, pack_writer, en_data, tr_data, mode)

        merged = en_data.copy()
        for k, v in tr_data.items():
            if k in merged and v:
                merged[k] = v

        if not pending:
            return self._write_legacy_lang_output(merged, tr_path, output_mode, pack_writer, zout, written_inplace)

        self.callbacks.on_log(f"⚡ Перевод {mod_name} [Интерфейс, legacy] — {len(pending)} строк", "cyan")
        translated = self.service.translate_dict(
            pending, target_lang, self.callbacks, context=mod_name, usage_label=(mod_name, "Интерфейс")
        )
        for key, value in translated.items():
            merged[key] = value
        return self._write_legacy_lang_output(merged, tr_path, output_mode, pack_writer, zout, written_inplace)

    def _collect_pending_translations(self, en_data, tr_data, mode, target_lang):
        pending: dict[str, str] = {}
        for key, value in en_data.items():
            if not isinstance(value, str) or not value.strip():
                continue
            if not looks_like_source_language(value):
                continue
            if is_technical_term(value):
                continue
            existing = tr_data.get(key)
            if mode == "append" and existing and existing.strip() and existing != value:
                continue
            if mode == "append" and existing and existing.strip() == value:
                continue
            if mode == "skip":
                if existing and already_translated(existing, target_lang["regex"]):
                    continue
            pending[key] = value
        return pending

    def _count_translatable_entries(self, en_data: dict) -> int:
        return sum(
            1
            for v in en_data.values()
            if isinstance(v, str) and looks_like_source_language(v) and not is_technical_term(v)
        )

    def _write_lang_output(self, data, tr_path, output_mode, pack_writer, zout, written_inplace, item, en_data) -> bool:
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        if output_mode == "resourcepack" and pack_writer:
            pack_writer.write(tr_path, payload)
            return True
        if zout:
            zout.writestr(tr_path, payload)
            written_inplace.add(tr_path)
            return True
        return False

    def _write_legacy_lang_output(self, data, tr_path, output_mode, pack_writer, zout, written_inplace) -> bool:
        payload = _serialize_legacy_lang(data)
        if output_mode == "resourcepack" and pack_writer:
            pack_writer.write(tr_path, payload)
            return True
        if zout:
            zout.writestr(tr_path, payload)
            written_inplace.add(tr_path)
            return True
        return False

    def _copy_existing(self, zin, locale_files, tr_key, tr_path, output_mode, pack_writer, en_data, tr_data, mode) -> bool:
        if tr_key not in locale_files:
            return False
        raw = zin.read(locale_files[tr_key])
        if output_mode == "resourcepack" and pack_writer:
            pack_writer.write(tr_path, raw)
            return True
        return False

