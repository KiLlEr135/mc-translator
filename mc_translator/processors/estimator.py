from __future__ import annotations

import json
import os
import zipfile

from mc_translator.constants import BOOK_PATH_MARKERS, MD_PATH_MARKERS, RESEARCH_PATH_MARKERS
from mc_translator.json_utils import iter_all_json_strings, iter_translatable_strings, load_lenient_json
from mc_translator.processors.guide_md import count_book_md_blocks, is_localized_guide_path
from mc_translator.processors.snbt_extract import extract_snbt_strings
from mc_translator.text_processing import already_translated, is_technical_term, looks_like_source_language


class StringEstimator:
    def __init__(self, state: "JobState") -> None:
        self.state = state

    def estimate(
        self,
        jar_files: list[str],
        loose_files: list[str],
        snbt_files: list[str],
        *,
        target_lang: dict,
        mode: str,
        translate_mods: bool,
        translate_books: bool,
        translate_quests: bool,
        smart_glue: bool,
        generic_json_files: list[str] | None = None,
        lang_files: list[str] | None = None,
        nbt_files: list[str] | None = None,
        yaml_toml_files: list[str] | None = None,
        text_files: list[str] | None = None,
        archives: list[str] | None = None,
        cfg_files: list[str] | None = None,
        mcfunction_files: list[str] | None = None,
        xml_files: list[str] | None = None,
        bat_files: list[str] | None = None,
        estimate_cache: dict | None = None,
    ) -> int:
        total = 0
        target_file = f"{target_lang['file']}.json"
        target_regex = target_lang["regex"]

        for path in jar_files:
            if not self.state.should_run():
                return total
            self.state.wait_if_paused()
            total += self._estimate_jar(
                path, target_file, target_lang, mode, translate_mods, translate_books, smart_glue,
                estimate_cache=estimate_cache,
            )

        for path in loose_files:
            if not self.state.should_run():
                return total
            self.state.wait_if_paused()
            total += self._estimate_loose(path, target_file, mode, target_regex)

        if translate_quests:
            for path in snbt_files:
                if not self.state.should_run():
                    return total
                self.state.wait_if_paused()
                total += self._estimate_snbt(path, mode, target_regex)

        if translate_mods:
            # Rough line-based estimate, matching ModpackAnalyzer._analyze_generic_file's
            # approximation so the Analyze screen and the Translate ETA agree.
            # JSON and XML are handled separately below by parsing and
            # counting actual translatable values/nodes -- GenericJsonProcessor
            # and XmlProcessor translate by VALUE, not by line, so a
            # minified/single-line file (hundreds of values, one physical
            # line) was wildly under-counted here, skewing both the ETA and
            # the pre-flight cost estimate shown before a paid run starts.
            line_based = [
                (yaml_toml_files, "yaml_toml"),
                (text_files, "text"),
                (cfg_files, "cfg"),
                (mcfunction_files, "mcfunction"),
                (bat_files, "bat"),
            ]
            for file_list, kind in line_based:
                for path in file_list or []:
                    if not self.state.should_run():
                        return total
                    self.state.wait_if_paused()
                    total += self._count_generic_line_based(path, target_lang, kind, mode)

            for path in generic_json_files or []:
                if not self.state.should_run():
                    return total
                self.state.wait_if_paused()
                total += self._count_generic_json_file(path, target_lang, mode)

            for path in xml_files or []:
                if not self.state.should_run():
                    return total
                self.state.wait_if_paused()
                total += self._count_generic_xml_file(path, target_lang, mode)

            for path in lang_files or []:
                if not self.state.should_run():
                    return total
                self.state.wait_if_paused()
                total += self._count_lang_file(path, target_lang, mode)

            for path in nbt_files or []:
                if not self.state.should_run():
                    return total
                self.state.wait_if_paused()
                total += self._count_nbt_file(path, target_regex, mode)

            for path in archives or []:
                if not self.state.should_run():
                    return total
                self.state.wait_if_paused()
                total += self._count_archive(path, target_regex, mode)

        return total

    def _estimate_jar(
        self, path, target_file, target_lang, mode, translate_mods, translate_books, smart_glue,
        *, estimate_cache: dict | None = None,
    ) -> int:
        fp: dict | None = None
        if estimate_cache is not None:
            try:
                stat = os.stat(path)
            except OSError:
                stat = None
            if stat is not None:
                fp = {
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                    "mode": mode,
                    "mods": translate_mods,
                    "books": translate_books,
                }
                cached = estimate_cache.get(path)
                if cached and all(cached.get(k) == v for k, v in fp.items()):
                    return cached["count"]

        count = 0
        try:
            with zipfile.ZipFile(path, "r") as zin:
                locale = {
                    i.filename.lower(): i
                    for i in zin.infolist()
                    if target_file in i.filename.lower()
                    or f"/{target_lang['file']}/" in i.filename.lower()
                }
                for item in zin.infolist():
                    fl = item.filename.lower()
                    is_book_json = fl.endswith(".json") and (
                        ("/en_us/" in fl and any(m in fl for m in BOOK_PATH_MARKERS))
                        or any(m in fl for m in RESEARCH_PATH_MARKERS)
                    ) and not is_localized_guide_path(fl)
                    is_book_md = (
                        (fl.endswith(".md") or fl.endswith(".txt"))
                        and any(m in fl for m in MD_PATH_MARKERS)
                        and not is_localized_guide_path(fl)
                    )
                    is_lang = fl.endswith("en_us.json") and not is_book_json

                    if translate_mods and is_lang:
                        count += self._count_lang(zin, item, locale, target_file, mode, target_lang["regex"])
                    elif translate_books and is_book_json:
                        count += self._count_book_json(zin, item)
                    elif translate_books and is_book_md:
                        count += self._count_book_md(zin, item)
        except (OSError, zipfile.BadZipFile):
            pass

        if estimate_cache is not None and fp is not None:
            estimate_cache[path] = {**fp, "count": count}

        return count

    def _count_lang(self, zin, item, locale, target_file, mode, target_regex) -> int:
        try:
            en = load_lenient_json(zin.read(item))
        except (json.JSONDecodeError, OSError):
            return 0
        tr_key = item.filename.lower().replace("en_us.json", target_file)
        tr = {}
        if tr_key in locale:
            try:
                tr = load_lenient_json(zin.read(locale[tr_key]))
            except (json.JSONDecodeError, OSError):
                pass
        n = 0
        for key, value in en.items():
            if not isinstance(value, str) or not looks_like_source_language(value) or is_technical_term(value):
                continue
            existing = tr.get(key)
            done = isinstance(existing, str) and existing.strip() and already_translated(existing, target_regex)
            if mode == "force" or not done:
                n += 1
        return n

    def _count_book_json(self, zin, item) -> int:
        try:
            data = load_lenient_json(zin.read(item))
        except (json.JSONDecodeError, OSError):
            return 0
        return sum(
            1
            for _, s in iter_translatable_strings(data)
            if s.strip() and looks_like_source_language(s) and not is_technical_term(s)
        )

    def _count_book_md(self, zin, item) -> int:
        # Delegates to guide_md's block classifier so this estimate matches
        # what a real translation run would actually translate (see
        # JarProcessor._process_book_md / translate_guide_markdown).
        try:
            text = zin.read(item).decode("utf-8-sig", errors="replace")
        except OSError:
            return 0
        return count_book_md_blocks(text)

    def _estimate_loose(self, path, target_file, mode, target_regex) -> int:
        try:
            with open(path, encoding="utf-8") as f:
                en = load_lenient_json(f.read().encode("utf-8"))
            tr_path = path.replace("en_us.json", target_file)
            tr = {}
            if os.path.exists(tr_path):
                with open(tr_path, encoding="utf-8") as f:
                    t = load_lenient_json(f.read().encode("utf-8"))
                    tr = t
        except (json.JSONDecodeError, OSError):
            return 0
        n = 0
        for key, value in en.items():
            if not isinstance(value, str) or not looks_like_source_language(value) or is_technical_term(value):
                continue
            existing = tr.get(key)
            done = isinstance(existing, str) and existing.strip() and already_translated(existing, target_regex)
            if mode == "force" or not done:
                n += 1
        return n

    def _estimate_snbt(self, path, mode, target_regex) -> int:
        """Mirrors SnbtProcessor.process's actual per-mode source selection
        (see processors/snbt.py) so this pre-flight number agrees with what a
        real run will do: force reads from the pristine .bak backup (falling
        back to the live file if no backup exists yet -- first run), while
        append/skip read the live file and skip/append additionally apply the
        skip-mode 90%-done bail threshold."""
        try:
            if mode == "force":
                backup = path + ".bak"
                source_path = backup if os.path.exists(backup) else path
                with open(source_path, encoding="utf-8") as f:
                    content = f.read()
                return len(extract_snbt_strings(content))
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except OSError:
            return 0

        pending = extract_snbt_strings(content, skip_translated_regex=target_regex)
        if mode == "skip":
            total = len(extract_snbt_strings(content, require_source_language=False))
            if total and (total - len(pending)) >= total * 0.9:
                return 0
        return len(pending)

    def _count_lines_in_text(
        self, text: str, existing_text: str | None, target_regex: str, kind: str, mode: str
    ) -> int:
        lines = text.split("\n")
        existing_lines = existing_text.split("\n") if existing_text is not None else None
        existing_aligned = existing_lines is not None and len(existing_lines) == len(lines)
        n = 0
        for idx, line in enumerate(lines):
            s = line.strip()
            if not s:
                continue
            if kind in ("cfg", "mcfunction", "bat"):
                if kind == "cfg" and not (s.startswith("#") or s.startswith("//")):
                    continue
                if kind == "mcfunction" and not s.startswith("#"):
                    continue
                if kind == "bat" and not (s.lower().startswith("rem") or s.startswith("::")):
                    continue
            if is_technical_term(s) or not looks_like_source_language(s):
                continue
            if mode != "force" and existing_aligned:
                existing_line = existing_lines[idx].strip()
                if existing_line and already_translated(existing_line, target_regex):
                    continue
            n += 1
        return n

    def _count_generic_line_based(self, path, target_lang, kind, mode) -> int:
        """Rough line-based estimate, matching ModpackAnalyzer._analyze_generic_file's
        approximation. Checks the target-language SIDECAR (positionally, same
        convention as LineProcessor -- see line_processor.py) for append/skip
        instead of testing the English source line against the target-
        language regex, which can never match and used to make every second
        pass over these formats report 'translate everything' regardless of
        how much was already done."""
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            return 0
        existing_text = None
        if mode != "force":
            target_path = os.path.join(os.path.dirname(path), self._target_name_for(path, target_lang))
            if os.path.exists(target_path):
                try:
                    with open(target_path, encoding="utf-8") as f:
                        existing_text = f.read()
                except OSError:
                    existing_text = None
        return self._count_lines_in_text(text, existing_text, target_lang["regex"], kind, mode)

    @staticmethod
    def _target_name_for(path: str, target_lang: dict) -> str:
        base_name = os.path.basename(path)
        lower_name = base_name.lower()
        if "en_us" in lower_name:
            return lower_name.replace("en_us", target_lang["file"])
        name, ext = os.path.splitext(base_name)
        return f"{name}_{target_lang['file']}{ext}"

    def _count_generic_json_file(self, path, target_lang, mode) -> int:
        """Value-based count matching GenericJsonProcessor's actual
        behavior (iterates every JSON string value via iter_all_json_strings,
        not physical lines). Must read the same way GenericJsonProcessor
        does (utf-8-sig + load_lenient_json) -- a bare utf-8/json.loads used
        to raise (and silently undercount as 0) on a BOM'd or JSONC
        (//comments, trailing commas -- common in mod config JSON) file the
        actual processor parses just fine."""
        try:
            with open(path, encoding="utf-8-sig") as f:
                data = load_lenient_json(f.read())
        except (json.JSONDecodeError, OSError):
            return 0

        target_path = os.path.join(os.path.dirname(path), self._target_name_for(path, target_lang))
        existing_map: dict = {}
        if mode != "force" and os.path.exists(target_path):
            try:
                with open(target_path, encoding="utf-8-sig") as f:
                    existing_data = load_lenient_json(f.read())
                existing_map = {
                    "/".join(map(str, p)): s for p, s in iter_all_json_strings(existing_data)
                }
            except (json.JSONDecodeError, OSError):
                existing_map = {}

        count = 0
        for path_tuple, string in iter_all_json_strings(data):
            if is_technical_term(string) or not looks_like_source_language(string):
                continue
            key = "/".join(map(str, path_tuple))
            existing = existing_map.get(key, "")
            if mode == "append" and existing.strip() and existing != string:
                continue
            if mode == "append" and existing.strip() == string:
                continue
            if mode == "skip" and existing.strip() and already_translated(existing, target_lang["regex"]):
                continue
            count += 1
        return count

    def _count_generic_xml_file(self, path, target_lang, mode) -> int:
        """Node-based count matching XmlProcessor's actual behavior
        (collect_translatable walks text/attribute/tail nodes, not lines)."""
        try:
            from mc_translator.processors.xml import collect_all_text, collect_translatable
            try:
                from lxml import etree as ET
            except ImportError:
                import xml.etree.ElementTree as ET
            root = ET.parse(path).getroot()
        except Exception:
            return 0

        target_path = os.path.join(os.path.dirname(path), self._target_name_for(path, target_lang))
        existing_map: dict = {}
        if mode != "force" and os.path.exists(target_path):
            try:
                existing_map = collect_all_text(ET.parse(target_path).getroot())
            except Exception:
                existing_map = {}

        translatable_attrs = {"title", "name", "header", "label", "text"}
        return len(collect_translatable(root, translatable_attrs, mode, target_lang, existing_map))

    def _count_lang_file(self, path, target_lang, mode) -> int:
        """Checks the target-language sidecar (ru_ru.lang, keyed lookup) for
        append/skip, matching LangProcessor's real keyed comparison -- unlike
        checking the English source value against the target-language regex,
        which can never match and used to report every key as still needing
        translation regardless of how much the sidecar already had done."""
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return 0

        target_regex = target_lang["regex"]
        existing_values: dict[str, str] = {}
        if mode != "force":
            target_path = os.path.join(os.path.dirname(path), self._target_name_for(path, target_lang))
            if os.path.exists(target_path):
                try:
                    with open(target_path, encoding="utf-8") as f:
                        for line in f:
                            es = line.strip()
                            if not es or es.startswith("#") or "=" not in es:
                                continue
                            k, v = es.split("=", 1)
                            existing_values[k.strip()] = v.strip()
                except OSError:
                    pass

        n = 0
        for line in lines:
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            key, value = s.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not value or is_technical_term(value) or not looks_like_source_language(value):
                continue
            existing = existing_values.get(key, "")
            if mode != "force" and existing and already_translated(existing, target_regex):
                continue
            n += 1
        return n

    def _count_nbt_file(self, path, target_regex, mode) -> int:
        try:
            import nbtlib
            from nbtlib import tag
        except Exception:
            return 0
        try:
            # nbtlib.load(path) (not File.load(fileobj), which requires an
            # explicit `gzipped` argument and errors out otherwise) opens the
            # path itself and auto-detects gzip compression -- matches the
            # fix already applied in nbt.py's NbtProcessor.
            data = nbtlib.load(path)
        except Exception:
            return 0

        count = 0

        def _walk(obj) -> None:
            nonlocal count
            # nbtlib's tag classes are named Base/Compound/List/String (no
            # "Tag" suffix) on the installed nbtlib>=2.0.4 -- tag.CompoundTag/
            # ListTag/StringTag never existed, so every isinstance() check
            # here used to raise AttributeError uncaught, crashing the whole
            # Analyze/cost-estimate pass for any modpack with an NBT file.
            if isinstance(obj, tag.Compound):
                for v in obj.values():
                    _walk(v)
            elif isinstance(obj, tag.List):
                for v in obj:
                    _walk(v)
            elif isinstance(obj, tag.String):
                s = str(obj)
                if not s.strip() or is_technical_term(s) or not looks_like_source_language(s):
                    return
                if mode == "force" or not already_translated(s, target_regex):
                    count += 1

        try:
            _walk(data)
        except Exception:
            return 0
        return count

    def _count_archive(self, path, target_regex, mode) -> int:
        text_exts = {".json", ".lang", ".yaml", ".yml", ".toml", ".txt", ".md", ".snbt"}
        total = 0
        try:
            with zipfile.ZipFile(path, "r") as zf:
                for info in zf.infolist():
                    ext = os.path.splitext(info.filename)[1].lower()
                    if ext not in text_exts:
                        continue
                    try:
                        text = zf.read(info).decode("utf-8-sig", errors="replace")
                    except Exception:
                        continue
                    if ext == ".lang":
                        for line in text.split("\n"):
                            s = line.strip()
                            if not s or s.startswith("#") or "=" not in s:
                                continue
                            value = s.split("=", 1)[1].strip()
                            if not value or is_technical_term(value) or not looks_like_source_language(value):
                                continue
                            if mode != "force" and already_translated(value, target_regex):
                                continue
                            total += 1
                    else:
                        total += self._count_lines_in_text(text, None, target_regex, "text", mode)
        except (OSError, zipfile.BadZipFile):
            pass
        return total