import json
import os
import re
import zipfile

from mc_translator.constants import BOOK_PATH_MARKERS, MD_PATH_MARKERS, RESEARCH_PATH_MARKERS
from mc_translator.json_utils import iter_translatable_strings, load_lenient_json
from mc_translator.mod_names import get_mod_name
from mc_translator.output.pack_writer import expected_pack_paths
from mc_translator.processors.discovery import (
    discover_jar_files,
    discover_loose_lang_files,
    discover_mcfunction_files,
    discover_mod_content_files,
    discover_snbt_files,
)
from mc_translator.processors.guide_md import (
    count_guide_blocks,
    guide_target_path,
    is_localized_guide_path,
)
from mc_translator.processors.resourcepack_path import compute_target_paths
from mc_translator.processors.snbt_extract import extract_snbt_strings
from mc_translator.text_processing import (
    already_translated,
    is_technical_term,
    looks_like_source_language,
)


class _PackIndex:
    """Read-only view into the most recent resourcepack/datapack zips for
    a given pack_name (if any exist on disk yet).

    Why this exists: output_mode="resourcepack" (the recommended, default
    mode) never writes a translated string back into the original mod jar
    or source file -- every processor writes exclusively into PackWriter's
    separate zips (see e.g. lang.py/generic_json.py's "if output_mode ==
    'resourcepack' and pack_writer: pack_writer.write(...)" branches, and
    PackWriter itself). Before this class existed, ModpackAnalyzer only
    ever looked at the original on-disk source, so "Готовность" could only
    ever reflect output_mode="inplace" runs -- for a resourcepack-mode
    user (the common case), the Analyze screen was structurally incapable
    of showing real progress for anything except FTB Quests SNBT (the one
    content type snbt.py always writes in-place, regardless of
    output_mode). PackWriter rebuilds its zip from scratch every run but
    always re-emits every previously-cached translation into it (not just
    this run's new strings), so the CURRENT resourcepack/datapack zip is a
    complete, accurate on-disk snapshot of "everything translated so far"
    -- safe to treat as a second source of truth alongside the original
    file/jar.
    """

    def __init__(self, mc_dir: str, pack_name: str | None) -> None:
        rp_path, dp_path = expected_pack_paths(mc_dir, pack_name)
        self._rp = self._open(rp_path)
        self._dp = self._open(dp_path)

    @staticmethod
    def _open(path: str) -> zipfile.ZipFile | None:
        if not path or not os.path.isfile(path):
            return None
        try:
            return zipfile.ZipFile(path, "r")
        except (OSError, zipfile.BadZipFile):
            return None

    def _zip_for(self, internal_path: str) -> zipfile.ZipFile | None:
        # Mirrors PackWriter.handle_for_path: "data/"-prefixed entries
        # (e.g. mcfunction) go to the datapack zip, everything else
        # ("assets/...") to the resourcepack zip.
        return self._dp if internal_path.lower().startswith("data/") else self._rp

    def read(self, internal_path: str) -> bytes | None:
        z = self._zip_for(internal_path)
        if z is None:
            return None
        try:
            return z.read(internal_path)
        except KeyError:
            return None

    def read_json(self, internal_path: str) -> dict:
        raw = self.read(internal_path)
        if raw is None:
            return {}
        try:
            data = load_lenient_json(raw)
        except (json.JSONDecodeError, OSError):
            return {}
        # load_lenient_json only guarantees valid JSON, not a JSON *object*
        # -- a stray/foreign zip entry containing e.g. a JSON array at the
        # expected path would otherwise hand callers a list and crash on
        # the .get(...) they all assume dict gives them. Found during this
        # fix's own verification pass.
        return data if isinstance(data, dict) else {}

    def close(self) -> None:
        if self._rp:
            self._rp.close()
        if self._dp:
            self._dp.close()


class ModpackAnalyzer:
    def __init__(self, state) -> None:
        self.state = state
        self._pack_index: _PackIndex | None = None

    def analyze(
        self,
        mc_dir: str,
        *,
        target_lang: dict,
        translate_mods: bool,
        translate_books: bool,
        translate_quests: bool,
        file_types: dict | None = None,
        pack_name: str | None = None,
        on_row,
        on_log,
        on_status,
    ) -> tuple[int, int]:
        target_file = f"{target_lang['file']}.json"
        target_regex = target_lang["regex"]

        total_en = 0
        total_tr = 0

        # See _PackIndex's docstring: output_mode="resourcepack" (the
        # default) never writes translations back into the original
        # source/jar, so every _analyze_* method below also needs to check
        # the current resourcepack/datapack zip (if any) as a fallback.
        # Built once per analyze() call and closed in `finally` below so an
        # open zip handle never lingers and blocks a subsequent run from
        # deleting/recreating the same-named pack.
        self._pack_index = _PackIndex(mc_dir, pack_name)
        try:
            # Discover all files using unified discovery
            jars = discover_jar_files(mc_dir) if (translate_mods or translate_books) else []
            loose = discover_loose_lang_files(mc_dir) if (translate_mods or translate_quests) else []
            snbt = discover_snbt_files(mc_dir) if translate_quests else []

            # Generic files (single-walk discovery, bucketed by type). pack_name
            # excludes the tool's own prior resourcepack/datapack output from the
            # "archive" bucket -- see discover_mod_content_files' docstring.
            mod_content = discover_mod_content_files(mc_dir, pack_name=pack_name) if translate_mods else {}
            generic_json = mod_content.get("generic_json", [])
            lang_files = mod_content.get("lang", [])
            nbt_files = mod_content.get("nbt", [])
            yaml_toml = mod_content.get("yaml_toml", [])
            text_files = mod_content.get("text", [])
            archives = mod_content.get("archive", [])
            cfg_files = mod_content.get("cfg", [])
            mcfunction = discover_mcfunction_files(mc_dir) if translate_mods else []
            xml_files = mod_content.get("xml", [])
            bat_files = mod_content.get("bat", [])

            # Same per-file-type gating as runtime/job.py's run_translation, so
            # analysis counts match what a translation run would actually touch.
            ft = file_types or {}
            if not ft.get("generic_json", True):
                generic_json = []
            if not ft.get("lang", True):
                lang_files = []
            if not ft.get("nbt", True):
                nbt_files = []
            if not ft.get("yaml_toml", True):
                yaml_toml = []
            if not ft.get("text", True):
                text_files = []
            if not ft.get("archive", True):
                archives = []
            if not ft.get("cfg", True):
                cfg_files = []
            if not ft.get("mcfunction", True):
                mcfunction = []
            if not ft.get("xml", True):
                xml_files = []
            if not ft.get("bat", True):
                bat_files = []

            all_lists = [
                (jars, "jar"), (loose, "loose"), (snbt, "snbt"),
                (generic_json, "json"), (lang_files, "lang"), (nbt_files, "nbt"),
                (yaml_toml, "yaml_toml"), (text_files, "text"), (archives, "archive"),
                (cfg_files, "cfg"), (mcfunction, "mcfunction"), (xml_files, "xml"),
                (bat_files, "bat")
            ]

            total_files = sum(len(lst) for lst, _ in all_lists)
            processed_files = 0

            for file_list, kind in all_lists:
                for path in file_list:
                    if not self.state.should_run():
                        break
                    self.state.wait_if_paused()

                    processed_files += 1
                    on_status(f"Анализ: {os.path.basename(path)}...", processed_files / max(total_files, 1))

                    en, tr = 0, 0
                    if kind == "jar":
                        mod_name = get_mod_name(path)
                        en, tr = self._analyze_jar(path, target_file, target_regex, translate_mods, translate_books, on_row, mod_name)
                    elif kind == "snbt":
                        en, tr = self._analyze_snbt(path, target_regex, on_row)
                    elif kind == "loose":
                        en, tr = self._analyze_lang_simple(path, target_file, target_regex, on_row, mc_dir=mc_dir, target_lang=target_lang)
                    elif kind == "lang":
                        en, tr = self._analyze_legacy_lang_file(path, target_lang, target_regex, on_row, mc_dir=mc_dir)
                    elif kind in ("json", "yaml_toml", "xml", "text", "cfg", "mcfunction", "bat"):
                        # Generic text-based analysis
                        en, tr = self._analyze_generic_file(path, target_regex, kind, on_row, mc_dir=mc_dir, target_lang=target_lang)

                    total_en += en
                    total_tr += tr

            return total_en, total_tr
        finally:
            self._pack_index.close()
            self._pack_index = None

    def _analyze_lang_simple(self, path, target_file, target_regex, on_row, *, mc_dir=None, target_lang=None):
        en_c = tr_c = 0
        try:
            with open(path, encoding="utf-8") as f:
                en_data = load_lenient_json(f.read())
            tr_data = {}
            # compute_target_paths matches "en_us" case-insensitively (see
            # its docstring/implementation), unlike the plain
            # path.replace("en_us.json", target_file) this used to be --
            # discover_loose_lang_files finds a file by
            # name.lower() == "en_us.json", so an on-disk "En_us.json" made
            # the old .replace() a silent no-op (tr_path == path, "exists"
            # trivially true, tr_data got the ENGLISH source re-read as if
            # it were the translation -- always 0%, and the pack-fallback
            # below was never even reached). Found + confirmed via a
            # standalone repro during this fix's own verification pass.
            if target_lang is not None:
                _, tr_path, internal_path = compute_target_paths(path, target_lang["file"], mc_dir)
            else:
                tr_path, internal_path = path.replace("en_us.json", target_file), None
            if os.path.exists(tr_path):
                with open(tr_path, encoding="utf-8") as f:
                    tr_data = load_lenient_json(f.read())
            elif internal_path is not None and self._pack_index is not None:
                # No inplace sidecar -- check the current resourcepack/
                # datapack zip (output_mode="resourcepack" never writes
                # the sidecar file on disk, see _PackIndex's docstring).
                tr_data = self._pack_index.read_json(internal_path)

            for key, value in en_data.items():
                if not isinstance(value, str) or not looks_like_source_language(value) or is_technical_term(value):
                    continue
                en_c += 1
                existing = str(tr_data.get(key, ""))
                if existing.strip() and already_translated(existing, target_regex):
                    tr_c += 1
        except Exception:
            pass
        if en_c:
            on_row("📄", os.path.basename(path), "Словарь", tr_c, en_c, int(tr_c / en_c * 100))
        return en_c, tr_c

    def _analyze_legacy_lang_file(self, path, target_lang, target_regex, on_row, *, mc_dir=None):
        """Real key=value .lang files (pre-1.13 format) -- the "lang"
        discovery bucket only ever contains "en_us.lang" files (see
        discovery.py), which load_lenient_json (a JSON parser) can't parse
        at all. Routing these through _analyze_lang_simple used to always
        raise on the first line and get silently swallowed, so this file's
        content never appeared on the Analyze screen even though the real
        run (LangProcessor) translates it correctly."""
        en_c = tr_c = 0
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()

            target_name = os.path.basename(path).lower().replace("en_us.lang", f"{target_lang['file']}.lang")
            target_path = os.path.join(os.path.dirname(path), target_name)
            existing: dict[str, str] = {}
            if os.path.exists(target_path):
                with open(target_path, encoding="utf-8") as f:
                    for line in f:
                        es = line.strip()
                        if not es or es.startswith("#") or "=" not in es:
                            continue
                        k, v = es.split("=", 1)
                        existing[k.strip()] = v.strip()
            elif self._pack_index is not None:
                # compute_target_paths naturally produces the same lowercase
                # "<lang_code>.lang" name as target_name above (it finds
                # "en_us" in the source filename and substitutes it) -- see
                # its docstring. No legacy-version case adjustment here:
                # analyze() has never received mc_version, so this matches
                # the analyzer's existing (pre-existing-limitation) behavior
                # rather than introducing a new one.
                _, _, internal_path = compute_target_paths(path, target_lang["file"], mc_dir)
                raw = self._pack_index.read(internal_path)
                if raw is not None:
                    try:
                        text = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        text = ""
                    for line in text.splitlines():
                        es = line.strip()
                        if not es or es.startswith("#") or "=" not in es:
                            continue
                        k, v = es.split("=", 1)
                        existing[k.strip()] = v.strip()

            for line in lines:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                key, value = s.split("=", 1)
                value = value.strip()
                if not value or not looks_like_source_language(value) or is_technical_term(value):
                    continue
                en_c += 1
                existing_value = existing.get(key.strip(), "")
                if existing_value.strip() and already_translated(existing_value, target_regex):
                    tr_c += 1
        except OSError:
            pass
        if en_c:
            on_row("📄", os.path.basename(path), "Словарь", tr_c, en_c, int(tr_c / en_c * 100))
        return en_c, tr_c

    def _analyze_generic_file(self, path, target_regex, kind, on_row, *, mc_dir=None, target_lang=None):
        # Very rough estimation for other formats. Previously compared each
        # line of the ENGLISH SOURCE against itself (checking whether an
        # untouched en_us line happened to already match the target-language
        # regex) -- always ~0 regardless of output_mode, since the real
        # translated output never lands back in `path` except for an
        # inplace run that overwrote it outright. Now locates the actual
        # translated counterpart (sidecar file for output_mode="inplace",
        # or the corresponding resourcepack/datapack zip entry for
        # "resourcepack") the same way the real processor would, and
        # compares by line position -- exact for the line-based cfg/
        # mcfunction/bat/text processors, an approximation for json/
        # yaml_toml/xml (which re-serialize and don't preserve original
        # line breaks 1:1), same "very rough" spirit as before.
        en_c = tr_c = 0
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            return 0, 0

        tr_lines: list[str] = []
        if target_lang is not None:
            _, target_path, internal_path = compute_target_paths(path, target_lang["file"], mc_dir)
            if os.path.exists(target_path):
                try:
                    with open(target_path, encoding="utf-8") as f:
                        tr_lines = f.readlines()
                except OSError:
                    tr_lines = []
            elif self._pack_index is not None:
                raw = self._pack_index.read(internal_path)
                if raw is not None:
                    try:
                        tr_lines = raw.decode("utf-8").splitlines(keepends=True)
                    except UnicodeDecodeError:
                        tr_lines = []
        tr_stripped = [ln.strip() for ln in tr_lines]

        try:
            for idx, line in enumerate(lines):
                s = line.strip()
                if not s:
                    continue

                # SAFE MODE Sync: Only count comments for technical files
                if kind in ("cfg", "mcfunction", "bat"):
                    if kind == "cfg" and not (s.startswith("#") or s.startswith("//")):
                        continue
                    if kind == "mcfunction" and not s.startswith("#"):
                        continue
                    if kind == "bat" and not (s.lower().startswith("rem") or s.startswith("::")):
                        continue

                if is_technical_term(s) or not looks_like_source_language(s):
                    continue
                en_c += 1
                existing = tr_stripped[idx] if idx < len(tr_stripped) else ""
                if existing and already_translated(existing, target_regex):
                    tr_c += 1
        except Exception:
            pass
        if en_c:
            icons = {"json": "JSON", "xml": "XML", "yaml_toml": "YML", "text": "TXT", "cfg": "CFG", "mcfunction": "MCF", "bat": "BAT"}
            on_row("📝", os.path.basename(path), icons.get(kind, kind.upper()), tr_c, en_c, int(tr_c / en_c * 100))
        return en_c, tr_c

    def _analyze_jar(self, path, target_file, target_regex, translate_mods, translate_books, on_row, mod_name):
        en_count = 0
        tr_count = 0
        try:
            with zipfile.ZipFile(path, "r") as zin:
                lang_file = target_file.replace(".json", "")
                locale = {
                    i.filename.lower(): i
                    for i in zin.infolist()
                    if target_file in i.filename.lower()
                    or f"/{lang_file}/" in i.filename.lower()
                    or f"_{lang_file}/" in i.filename.lower()
                }
                en_mod = tr_mod = 0
                en_book = tr_book = 0
                if translate_mods:
                    en_mod, tr_mod = self._analyze_mods_ui(zin, locale, target_file, mod_name, on_row)
                if translate_books:
                    en_book, tr_book = self._analyze_books(zin, locale, target_file, target_regex, mod_name, on_row)
                en_count = en_mod + en_book
                tr_count = tr_mod + tr_book
        except (OSError, zipfile.BadZipFile):
            pass
        return en_count, tr_count

    def _analyze_mods_ui(self, zin, locale, target_file, mod_name, on_row):
        en_c = tr_c = 0
        for item in zin.infolist():
            fl = item.filename.lower()
            if not fl.endswith("en_us.json") or any(x in fl for x in BOOK_PATH_MARKERS):
                continue
            try:
                en = load_lenient_json(zin.read(item))
                tr_key = fl.replace("en_us.json", target_file)
                if tr_key in locale:
                    tr = load_lenient_json(zin.read(locale[tr_key]))
                elif self._pack_index is not None:
                    # locale's keys (and tr_key above) are lowercased for a
                    # self-consistent in-jar comparison, but a real mod jar's
                    # internal paths are almost always already lowercase
                    # anyway (Forge/Fabric convention) -- the real writer
                    # (jar.py's _process_lang_entry) computes tr_path from
                    # the ORIGINAL-case item.filename, so recompute that
                    # exact string here rather than reusing the lowercased
                    # tr_key, in case this jar is one of the rare exceptions.
                    tr_path_pack = re.sub(r"en_us\.json$", target_file, item.filename, flags=re.IGNORECASE)
                    tr = self._pack_index.read_json(tr_path_pack)
                else:
                    tr = {}
                for key, value in en.items():
                    if not isinstance(value, str) or not looks_like_source_language(value) or is_technical_term(value):
                        continue
                    en_c += 1
                    existing = str(tr.get(key, ""))
                    if existing.strip() and existing != value:
                        tr_c += 1
            except (json.JSONDecodeError, OSError):
                continue
        if en_c:
            on_row("📦", mod_name, "Интерфейс", tr_c, en_c, int(tr_c / en_c * 100))
        return en_c, tr_c

    def _analyze_books(self, zin, locale, target_file, target_regex, mod_name, on_row):
        b_en = b_tr = m_en = m_tr = xml_en = xml_tr = 0
        for item in zin.infolist():
            fl = item.filename.lower()
            is_jb = fl.endswith(".json") and (
                ("/en_us/" in fl and any(x in fl for x in BOOK_PATH_MARKERS))
                or any(x in fl for x in RESEARCH_PATH_MARKERS)
            ) and not is_localized_guide_path(fl)
            is_mb = (
                (fl.endswith(".md") or fl.endswith(".txt"))
                and any(x in fl for x in MD_PATH_MARKERS)
                and not is_localized_guide_path(fl)
            )
            if is_jb:
                try:
                    en = load_lenient_json(zin.read(item))
                    tr_path = fl.replace("/en_us/", f"/{target_file.replace('.json','')}/")
                    if tr_path in locale:
                        tr = load_lenient_json(zin.read(locale[tr_path]))
                    elif self._pack_index is not None:
                        # Real writer (jar_books.py's _process_book_json)
                        # computes tr_path from item.filename (original
                        # case), not the lowercased fl -- recompute that
                        # exact string for the pack lookup, same reasoning
                        # as _analyze_mods_ui above.
                        tr_path_pack = re.sub(
                            r"/en_us/", f"/{target_file.replace('.json', '')}/", item.filename, flags=re.IGNORECASE
                        )
                        tr = self._pack_index.read_json(tr_path_pack)
                    else:
                        tr = {}
                    en_s = [s for p, s in iter_translatable_strings(en) if s.strip() and looks_like_source_language(s)]
                    tr_s = [s for p, s in iter_translatable_strings(tr)] if tr else []
                    b_en += len(en_s)
                    for idx, s in enumerate(en_s):
                        if idx < len(tr_s) and tr_s[idx] != s and tr_s[idx].strip():
                            b_tr += 1
                except (json.JSONDecodeError, OSError):
                    pass
            elif is_mb:
                try:
                    en_t = zin.read(item).decode("utf-8-sig", errors="replace")
                    # guide_target_path matches JarProcessor's real output
                    # path (GuideME's "_<lang>/" convention, or "/en_us/"
                    # substitution) so this lookup finds a prior run's actual
                    # translation instead of always missing and reporting 0%.
                    tr_path = guide_target_path(fl, target_file.replace(".json", ""))
                    # guide_target_path's real caller (jar_books.py's
                    # _process_book_md) also computes it from a lowercased
                    # `fl`, so -- unlike the JSON/lang branches above -- no
                    # separate case-preserving recompute is needed for the
                    # pack lookup here.
                    if tr_path and tr_path in locale:
                        tr_t = zin.read(locale[tr_path]).decode("utf-8-sig", errors="replace")
                    elif tr_path and self._pack_index is not None:
                        raw = self._pack_index.read(tr_path)
                        tr_t = raw.decode("utf-8-sig", errors="replace") if raw is not None else ""
                    else:
                        tr_t = ""
                    en_n, tr_n = count_guide_blocks(en_t, tr_t, target_regex)
                    m_en += en_n
                    m_tr += tr_n
                except OSError:
                    pass
            elif fl.endswith(".xml") and any(x in fl for x in BOOK_PATH_MARKERS) and not is_localized_guide_path(fl):
                try:
                    # Parse XML and count translatable text content
                    xml_content = zin.read(item).decode("utf-8")
                    # guide_target_path (not a bare "/en_us/" replace-or-self
                    # fallback) -- matches the real writer, jar_books.py's
                    # _process_book_xml, which also handles GuideME's
                    # "_<lang>/"-segment convention for XML books with no
                    # "/en_us/" folder at all. The old bare-fallback
                    # (tr_path = fl when "/en_us/" is absent) could never
                    # find a real translation for that case, in any mode.
                    tr_path = guide_target_path(fl, target_file.replace(".json", ""))
                    tr_content = ""
                    if tr_path and tr_path in locale:
                        tr_content = zin.read(locale[tr_path]).decode("utf-8")
                    elif tr_path and self._pack_index is not None:
                        raw = self._pack_index.read(tr_path)
                        if raw is not None:
                            tr_content = raw.decode("utf-8")

                    # Simple approach: count text nodes that need translation
                    try:
                        from lxml import etree as ET
                    except ImportError:
                        import xml.etree.ElementTree as ET

                    root = ET.fromstring(xml_content)
                    tr_root = ET.fromstring(tr_content) if tr_content else None

                    def count_translatable_text(element):
                        count = 0
                        if element.text and element.text.strip():
                            txt = element.text.strip()
                            if looks_like_source_language(txt) and not is_technical_term(txt):
                                count += 1

                        for attr in element.attrib:
                            if attr.lower() in {"title", "name", "header", "label", "text", "desc", "description"}:
                                val = element.attrib[attr]
                                if val.strip() and looks_like_source_language(val) and not is_technical_term(val):
                                    count += 1

                        for child in element:
                            count += count_translatable_text(child)
                            if child.tail and child.tail.strip():
                                txt = child.tail.strip()
                                if looks_like_source_language(txt) and not is_technical_term(txt):
                                    count += 1
                        return count

                    x_en = 0
                    x_tr = 0
                    en_count = count_translatable_text(root)
                    tr_count = 0
                    if tr_root is not None:
                        # Count how many are already translated by comparing
                        def count_translated_text(orig_elem, trans_elem):
                            count = 0
                            orig_text = orig_elem.text.strip() if orig_elem.text else ""
                            trans_text = trans_elem.text.strip() if trans_elem.text else ""
                            if orig_text and looks_like_source_language(orig_text) and not is_technical_term(orig_text):
                                if trans_text and already_translated(trans_text, target_regex):
                                    count += 1

                            for attr in orig_elem.attrib:
                                if attr.lower() in {"title", "name", "header", "label", "text", "desc", "description"}:
                                    orig_val = orig_elem.attrib[attr]
                                    trans_val = trans_elem.attrib.get(attr, "")
                                    if orig_val.strip() and looks_like_source_language(orig_val) and not is_technical_term(orig_val):
                                        if trans_val.strip() and already_translated(trans_val, target_regex):
                                            count += 1

                            orig_children = list(orig_elem)
                            trans_children = list(trans_elem)
                            for i in range(min(len(orig_children), len(trans_children))):
                                count += count_translated_text(orig_children[i], trans_children[i])

                            # Handle tails
                            for i in range(min(len(orig_children), len(trans_children))):
                                orig_tail = orig_children[i].tail.strip() if orig_children[i].tail else ""
                                trans_tail = trans_children[i].tail.strip() if trans_children[i].tail else ""
                                if orig_tail and looks_like_source_language(orig_tail) and not is_technical_term(orig_tail):
                                    if trans_tail and already_translated(trans_tail, target_regex):
                                        count += 1
                            return count

                        tr_count = count_translated_text(root, tr_root)

                    x_en += en_count
                    x_tr += tr_count
                    # Pre-existing bug, found during resourcepack-fallback
                    # verification (2026-07-16): x_en/x_tr used to be purely
                    # local to this branch and never reached the method's
                    # b_en/b_tr running totals (unlike the JSON/MD branches
                    # above, which fold into b_en/b_tr and m_en/m_tr) -- the
                    # per-file Analyze row was always correct, but the
                    # overall readiness percentage silently excluded 100% of
                    # XML book content, in every output_mode, not just
                    # resourcepack. Fixed by accumulating into xml_en/xml_tr
                    # (initialized alongside b_en/b_tr/m_en/m_tr) and adding
                    # them to the final return below.
                    xml_en += x_en
                    xml_tr += x_tr
                    if en_count:
                        on_row("📄", mod_name, "Книга(XML)", x_tr, x_en, int(x_tr / x_en * 100) if x_en > 0 else 0)
                    continue
                except Exception:
                    pass
        if b_en:
            on_row("📖", mod_name, "Книга(JSON)", b_tr, b_en, int(b_tr / b_en * 100))
        if m_en:
            on_row("📝", mod_name, "Книга(MD)", m_tr, m_en, int(m_tr / m_en * 100))
        return b_en + m_en + xml_en, b_tr + m_tr + xml_tr

    def _analyze_snbt(self, path, target_regex, on_row):
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except OSError:
            return 0, 0
        strings = extract_snbt_strings(content)
        en_c = len(strings)
        tr_c = sum(1 for s in strings if re.search(target_regex, s))
        if en_c:
            on_row("📜", os.path.basename(path), "Квесты", tr_c, en_c, int(tr_c / en_c * 100))
        return en_c, tr_c