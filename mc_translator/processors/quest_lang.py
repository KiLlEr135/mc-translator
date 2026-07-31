"""
Processor for FTB Quests' flat lang-key export files
(config/ftbquests/quests/lang/<code>.snbt).

Modern FTB Quests (this project targets MC 1.20+) can store ALL quest/
chapter/task text exclusively as lang keys here instead of as literal text
in quests/chapters/*.snbt -- confirmed on a real pack (ATM10, MC 1.21.1)
where every chapter/quest/reward-table .snbt had ZERO literal title/
description/subtitle text, while quests/lang/en_us.snbt alone held 8000+
translatable lines (quest.<id>.title/quest_desc/quest_subtitle,
task.<id>.title, chapter.<id>.title, chapter_group.<id>.title). SnbtProcessor
(snbt.py) only ever looks at quests/chapters/**, quests/chapter_groups.snbt,
quests/data.snbt and quests/reward_tables/** -- for a lang-key pack like
this, that means quest translation was silently a complete no-op (confirmed:
zero "[Квесты]" log lines across a real 6h45m run against this pack).

discover_snbt_files (discovery.py) deliberately excludes quests/lang/
entirely, and that exclusion must stay: it's FTBQuests' own multi-language
export tree (one flat <code>.snbt per language it ships), and a past attempt
to translate that whole tree corrupted the pack's own shipped es_es/it_it/
pt_br files -- every OTHER language's existing text also fails a
Cyrillic-target skip-check and would get blindly retranslated FROM that
other language INTO Russian if walked the same way SnbtProcessor walks
chapters/.

**Where the translation actually has to land (confirmed 2026-07-17 against a
real running ATM10 client, not guessed from decompiled strings):** editing
the flat quests/lang/<code>.snbt file alone is NOT enough -- the "FTB Quests
Lang Splitter" companion mod (github.com/pietro-lopes/FTB-Quests-Lang-
Splitter, confirmed via its real source/README) renders quest text from a
PER-CHAPTER split/merged cache under quests/lang/<code>/ (chapter.snbt,
chapter_group.snbt, file.snbt, reward_table.snbt, and chapters/<name>.snbt
per chapter file), not from the flat file directly. That cache is populated
by merging whichever of those split files are NOT already suffixed
`_merged` back into the flat file, automatically, on the next world load or
`/ftbquests reload` -- confirmed empty of any Russian content across every
one of these split/merged files on a real pack even after the flat file was
independently verified correct, and even after a full PC restart, until
this processor started writing the split files directly.

Critically, the mod's own `/langsplitter split` command always (re)builds
these split files from the ENGLISH reference (en_us.snbt), not from
whatever's currently translated -- running it after this processor has
already written a correct translation would silently regenerate English
templates that the next reload then merges over the real translation,
wiping it back to English. This is exactly how a real pack's ru_ru.snbt
ended up byte-identical to en_us.snbt. This processor writes ALREADY-
TRANSLATED split files directly instead, so a normal reload is enough and
`/langsplitter split` must never be run afterward.

This processor is deliberately narrower and safer than "translate the
lang/ tree": it only ever reads/writes files for the ONE target language
(e.g. ru_ru.snbt / lang/ru_ru/**), using en_us.snbt purely as the reference
key set to diff against -- never opening any other locale's file. For every
key present in en_us.snbt:
  - missing from a target file entirely -> translated and appended
  - present but the value still looks like source-language text -> replaced
    in place
  - present and already looks like the target language (the pack's own
    shipped partial translation, or a prior run's output) -> left untouched
  - entirely non-translatable markup (an {image:...} embed or {@pagebreak})
    -> copied verbatim, never sent to the AI at all (see
    _NON_TRANSLATABLE_MARKUP_RE below)
"""
from __future__ import annotations

import os
import re
import shutil

from mc_translator.engines.base import EngineCallbacks
from mc_translator.engines.service import TranslationService

# One entry is `dotted.key: "value"` or `dotted.key: ["v1" "v2" ...]` -- FTB
# Quests' SNBT dialect doesn't require commas between array elements (same
# as the object-list syntax seen in quests/chapters/*.snbt's `images: [ {...} {...} ]`).
_ENTRY_RE = re.compile(
    r'([A-Za-z0-9_.]+)\s*:\s*(?:"((?:[^"\\]|\\.)*)"|\[((?:\s*"(?:[^"\\]|\\.)*"\s*)*)\])'
)
_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')

# FTB Quests markup that carries zero natural-language text -- an image
# embed or a page-break marker, not a sentence that merely CONTAINS one of
# these (which still needs translating around it and is left alone here;
# this only matches when the ENTIRE value is exactly one of these forms).
# The existing shield system (text_processing.mask_protected_fragments)
# does NOT protect this syntax (its FORMAT_PATTERN only matches simple
# `{word}` tokens, not `{image:path width:N height:N align:X}` or
# `{@pagebreak}`), so sending these whole-value through an LLM as if they
# were prose risks it silently altering the layout parameters or dropping
# the tag -- confirmed on a real pack's pre-existing (not this tool's own)
# lang file, where several `{image:...}` tags had their width/height
# altered and one was entirely missing from a translated array.
_NON_TRANSLATABLE_MARKUP_RE = re.compile(r"^\{(?:image:[^}]*|@pagebreak)\}$")

# Matches a quest/task/chapter id field inside a real quests/chapters/*.snbt
# file -- deliberately requires the WHOLE quoted value to be hex, so item/
# icon ids like `id: "minecraft:coal"` never match.
_ID_FIELD_RE = re.compile(r'\bid:\s*"([A-Fa-f0-9]+)"')

# Lang-key prefixes that FTB Quests Lang Splitter groups into one aggregate
# file per prefix under quests/lang/<code>/ (NOT split further per-chapter --
# confirmed against a real pack's split output: chapter.snbt held every
# chapter's title in one file, not one file per chapter).
_TOP_LEVEL_PREFIXES = frozenset({"chapter", "chapter_group", "file", "reward_table"})


def _is_non_translatable_markup(value: str) -> bool:
    return bool(_NON_TRANSLATABLE_MARKUP_RE.match(value.strip()))


def _resolve_translated_value(candidate: str, sent: str, cur: str | None) -> str:
    """`TranslationService.translate_dict` falls back to echoing the exact
    untranslated string it was sent when the engine genuinely fails a key
    (service.py's documented contract: `result[key] = item.original`) -- a
    real translation essentially never reproduces its source byte-for-byte,
    so `candidate == sent` is a reliable "this one failed" signal.

    In mode="force", `sent` is always the English reference value, even for
    a key that already had a correctly-translated `cur` on disk -- so
    without this check, a single transient engine failure (routine per real
    OpenRouter logs: truncated/malformed responses happen regularly) would
    silently regress an already-good translation back to English. Confirmed
    on a real pack: this is exactly how quests/lang/ru_ru.snbt ended up
    byte-identical to en_us.snbt. Keep the pre-existing value instead."""
    if candidate == sent and cur is not None:
        return cur
    return candidate


def _unescape(value: str) -> str:
    return value.replace('\\"', '"').replace('\\\\', '\\')


def _escape(value: str) -> str:
    return value.replace('\\', '\\\\').replace('"', '\\"')


def parse_lang_entries(content: str) -> dict[str, str | list[str]]:
    """Parses every `key: value` pair out of a flat FTBQuests lang-export
    .snbt. Order-preserving; a repeated key keeps its LAST occurrence (same
    as FTB Quests' own SNBT reader would)."""
    entries: dict[str, str | list[str]] = {}
    for match in _ENTRY_RE.finditer(content):
        key = match.group(1)
        if match.group(2) is not None:
            entries[key] = _unescape(match.group(2))
        else:
            array_body = match.group(3) or ""
            entries[key] = [_unescape(m.group(1)) for m in _STRING_RE.finditer(array_body)]
    return entries


def _format_value(value: str | list[str]) -> str:
    if isinstance(value, list):
        items = " ".join(f'"{_escape(v)}"' for v in value)
        return f"[{items}]" if items else "[ ]"
    return f'"{_escape(value)}"'


def apply_lang_updates(content: str, updates: dict[str, str | list[str]]) -> str:
    """Replaces the value of every key in `updates` that already has a line
    in `content`, byte-for-byte untouched elsewhere -- same philosophy as
    snbt_extract.apply_snbt_translations. Keys with no existing line are
    left alone here (see append_new_lang_entries)."""

    def replace_one(match: re.Match) -> str:
        key = match.group(1)
        if key not in updates:
            return match.group(0)
        return f"{key}: {_format_value(updates[key])}"

    return _ENTRY_RE.sub(replace_one, content)


def append_new_lang_entries(content: str, new_entries: dict[str, str | list[str]]) -> str:
    """Appends brand-new `key: value` lines just before the closing `}` of
    the top-level object -- for keys the reference file has that the target
    file doesn't have at all yet."""
    if not new_entries:
        return content
    lines = "\n".join(f"\t{key}: {_format_value(value)}" for key, value in new_entries.items())
    stripped = content.rstrip()
    if stripped.endswith("}"):
        return stripped[:-1].rstrip() + "\n" + lines + "\n}\n"
    # Malformed/unexpected shape (shouldn't happen for a real FTBQuests
    # export) -- fail safe by appending rather than silently dropping
    # translated content.
    return content.rstrip() + "\n" + lines + "\n"


def build_chapter_id_map(chapters_dir: str) -> dict[str, str]:
    """Maps every quest/task/chapter id referenced inside each real
    quests/chapters/<name>.snbt to that chapter's own base filename (e.g.
    "theurgy") -- used to route quest.<id>.*/task.<id>.* lang keys to the
    split file FTB Quests Lang Splitter's own /langsplitter split would put
    them in (quests/lang/<code>/chapters/<name>.snbt). Picking up a few
    unrelated hex ids (reward ids, etc.) from the same file is harmless --
    those never appear as quest./task. lang keys so they're never looked up."""
    id_map: dict[str, str] = {}
    if not os.path.isdir(chapters_dir):
        return id_map
    for name in sorted(os.listdir(chapters_dir)):
        if not name.endswith(".snbt"):
            continue
        path = os.path.join(chapters_dir, name)
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue
        base = name[: -len(".snbt")]
        for match in _ID_FIELD_RE.finditer(content):
            id_map[match.group(1).upper()] = base
    return id_map


def bucket_for_key(key: str, id_map: dict[str, str]) -> tuple[str, str] | None:
    """Returns (bucket_kind, bucket_name) describing which FTB Quests Lang
    Splitter-shaped split file `key` belongs in, matching the mod's own
    real layout: reward_table./chapter_group./chapter./file. keys each land
    in one aggregate top-level file (bucket_kind="top"); quest./task. keys
    land in a per-chapter file under chapters/, keyed by whichever chapter
    actually defines that id (bucket_kind="chapter"). Returns None for a
    quest./task. key whose owning chapter couldn't be determined (id_map
    has no entry) or for any other unrecognized prefix -- callers should
    just skip split-file routing for that one key rather than guess wrong;
    the flat file still covers it."""
    segment, _, rest = key.partition(".")
    if segment in _TOP_LEVEL_PREFIXES:
        return ("top", segment)
    if segment in ("quest", "task"):
        id_part = rest.split(".", 1)[0]
        chapter = id_map.get(id_part.upper())
        if chapter is None:
            return None
        return ("chapter", chapter)
    return None


class QuestLangProcessor:
    """Fills gaps in FTB Quests' lang-key export for the CURRENT target
    language against quests/lang/en_us.snbt as the reference key set --
    both the flat quests/lang/<target>.snbt file and (when the FTB Quests
    Lang Splitter mod is present) the per-file split layout it actually
    renders quest text from. `file_path` is the flat TARGET file's path
    (e.g. .../lang/ru_ru.snbt); the reference en_us.snbt is always the
    sibling file in the same directory, and quests/chapters/ (two levels up
    from the lang dir) is read to map quest/task ids to their owning
    chapter's split file."""

    def __init__(
        self,
        service: TranslationService,
        state: "JobState",
        callbacks: EngineCallbacks,
    ) -> None:
        self.service = service
        self.state = state
        self.callbacks = callbacks

    def process(self, file_path: str, *, target_lang: dict, mode: str) -> None:
        reference_path = os.path.join(os.path.dirname(file_path), "en_us.snbt")
        try:
            with open(reference_path, encoding="utf-8") as f:
                reference = parse_lang_entries(f.read())
        except OSError as exc:
            self.callbacks.on_log(f"❌ Ошибка чтения {reference_path}: {exc}", "red")
            return
        if not reference:
            return

        # Captured BEFORE this run's own flat-file edits below, deliberately
        # -- this seeds a split file being generated for the first time
        # with whatever was ALREADY correctly translated there from a past
        # run, so we don't burn AI calls re-translating ~8000 lines a real
        # pack may already have covered. Seeding from this run's OWN fresh
        # edits instead would double-process anything just translated a
        # moment ago (it hasn't been vetted as "already good" by anything
        # other than the same translate call the split file would make
        # anyway) -- simpler and strictly safer to just let freshly
        # translated-this-run content go through its own independent call.
        try:
            with open(file_path, encoding="utf-8") as f:
                flat_existing = parse_lang_entries(f.read())
        except OSError:
            flat_existing = {}

        flat_written = self._process_file(file_path, reference, target_lang=target_lang, mode=mode)

        if not self._has_lang_splitter_mod(file_path):
            if flat_written:
                self._remind_generic(file_path)
            return

        lang_dir = os.path.dirname(file_path)
        chapters_dir = os.path.join(os.path.dirname(lang_dir), "chapters")
        id_map = build_chapter_id_map(chapters_dir)

        split_dir = os.path.join(lang_dir, target_lang["file"])
        buckets: dict[tuple[str, str], dict[str, str | list[str]]] = {}
        for key, value in reference.items():
            bucket = bucket_for_key(key, id_map)
            if bucket is None:
                continue
            buckets.setdefault(bucket, {})[key] = value

        any_split_written = False
        for (kind, name), bucket_reference in buckets.items():
            target_path = (
                os.path.join(split_dir, f"{name}.snbt")
                if kind == "top"
                else os.path.join(split_dir, "chapters", f"{name}.snbt")
            )
            if self._process_file(
                target_path,
                bucket_reference,
                target_lang=target_lang,
                mode=mode,
                seed_existing=flat_existing,
            ):
                any_split_written = True

        if any_split_written:
            self._remind_to_reload()
        elif flat_written:
            # Mod present but nothing could be split-file-routed (e.g. no
            # quests/chapters/ found, or every changed key was unmapped) --
            # fall back to the generic reminder so the flat-file edit isn't
            # silently unreported.
            self._remind_generic(file_path)

    def _process_file(
        self,
        file_path: str,
        reference: dict[str, str | list[str]],
        *,
        target_lang: dict,
        mode: str,
        seed_existing: dict[str, str | list[str]] | None = None,
    ) -> bool:
        """Fills gaps in `file_path` against `reference` -- shared by both
        the flat quests/lang/<locale>.snbt target and every FTB Quests Lang
        Splitter-shaped split file, so a single implementation safely
        populates whichever file(s) the pack's current FTBQuests + Lang
        Splitter setup actually reads at render time. Returns True if
        `file_path` was written.

        `seed_existing`, when given, is used as a starting point ONLY if
        `file_path` doesn't exist yet (a split file being generated for the
        first time) -- letting a fresh split file inherit whatever's already
        correctly translated elsewhere (the flat file) instead of treating
        every key as brand new and burning AI calls to re-translate text
        this run (or a past run) already covered."""
        if not reference:
            return False

        existing_content = ""
        if os.path.exists(file_path):
            backup = file_path + ".bak"
            if not os.path.exists(backup):
                shutil.copy2(file_path, backup)
            try:
                with open(file_path, encoding="utf-8") as f:
                    existing_content = f.read()
            except OSError as exc:
                self.callbacks.on_log(f"❌ Ошибка чтения {file_path}: {exc}", "red")
                return False
        existing = parse_lang_entries(existing_content) if existing_content else {}
        seeded_from_scratch = False
        if not existing and seed_existing:
            existing = {k: v for k, v in seed_existing.items() if k in reference}
            if existing:
                # Give the seeded keys real serialized text to live in, not
                # just entries in the `existing` dict -- otherwise an
                # already-good seeded value that needs no further
                # translation is never re-emitted anywhere below (it's only
                # ever written out via apply_lang_updates/
                # append_new_lang_entries mutating real existing text) and
                # would silently vanish from a brand-new split file.
                existing_content = append_new_lang_entries("{\n}\n", existing)
                seeded_from_scratch = True

        target_regex = target_lang["regex"]

        def needs_translation(value: str) -> bool:
            return (
                bool(value.strip())
                and not re.search(target_regex, value)
                and not _is_non_translatable_markup(value)
            )

        if mode == "skip" and existing:
            total = 0
            done = 0
            for key, ref_value in reference.items():
                items = ref_value if isinstance(ref_value, list) else [ref_value]
                cur = existing.get(key)
                cur_items = (cur if isinstance(cur, list) else [cur]) if cur is not None else [None] * len(items)
                for i, item in enumerate(items):
                    if not item.strip():
                        continue
                    total += 1
                    cur_item = cur_items[i] if i < len(cur_items) else None
                    if cur_item is not None and not needs_translation(cur_item):
                        done += 1
            if total and done >= total * 0.9:
                return False

        chunk: dict[str, str] = {}
        scalar_targets: dict[str, str] = {}
        list_targets: dict[str, list[str]] = {}
        list_placeholders: dict[str, dict[int, str]] = {}
        is_new: dict[str, bool] = {}
        # Keys/items that are missing but need no AI call at all (entirely
        # non-translatable markup, or an otherwise-empty value) -- still
        # added to the file, just copied straight from the reference
        # instead of being routed through translate_dict for nothing.
        direct_copy: dict[str, str | list[str]] = {}

        ph_counter = 0
        for key, ref_value in reference.items():
            cur = existing.get(key)
            is_missing = cur is None
            if not is_missing and mode != "force":
                if isinstance(cur, list):
                    if not any(needs_translation(v) for v in cur):
                        continue
                elif not needs_translation(cur):
                    continue
            base = ref_value if (is_missing or mode == "force") else cur
            if isinstance(base, list):
                touched: dict[int, str] = {}
                for i, item in enumerate(base):
                    if item.strip() and needs_translation(item):
                        ph_counter += 1
                        ph = str(ph_counter)
                        chunk[ph] = item
                        touched[i] = ph
                if touched:
                    list_targets[key] = list(base)
                    list_placeholders[key] = touched
                    is_new[key] = is_missing
                elif is_missing or mode == "force":
                    # Nothing here needed an AI call (empty / entirely
                    # non-translatable markup) -- still write it: either the
                    # key is new and must be added, or it's a force run
                    # resetting a possibly-mismatched existing value back to
                    # the clean reference.
                    direct_copy[key] = list(base)
                    is_new[key] = is_missing
            else:
                if base.strip() and needs_translation(base):
                    ph_counter += 1
                    ph = str(ph_counter)
                    chunk[ph] = base
                    scalar_targets[key] = ph
                    is_new[key] = is_missing
                elif (is_missing or mode == "force") and base.strip():
                    direct_copy[key] = base
                    is_new[key] = is_missing

        if not chunk and not direct_copy and not seeded_from_scratch:
            return False

        updates: dict[str, str | list[str]] = dict(direct_copy)

        if chunk:
            name = os.path.basename(file_path)
            self.callbacks.on_log(f"⚡ Перевод {name} [Квесты, lang-ключи] — {len(chunk)} строк", "yellow")
            translated = self.service.translate_dict(
                chunk, target_lang, self.callbacks, context="FTB Quests", usage_label=(name, "Квесты")
            )
            for key, ph in scalar_targets.items():
                sent = chunk[ph]
                candidate = translated.get(ph, sent)
                updates[key] = _resolve_translated_value(candidate, sent, existing.get(key))
            for key, template in list_targets.items():
                result = list(template)
                cur_list = existing.get(key)
                cur_list = cur_list if isinstance(cur_list, list) else None
                for i, ph in list_placeholders[key].items():
                    sent = chunk[ph]
                    candidate = translated.get(ph, sent)
                    cur_item = cur_list[i] if cur_list is not None and i < len(cur_list) else None
                    result[i] = _resolve_translated_value(candidate, sent, cur_item)
                updates[key] = result

        new_entries = {k: v for k, v in updates.items() if is_new[k]}
        replace_entries = {k: v for k, v in updates.items() if not is_new[k]}

        new_content = existing_content or "{\n}\n"
        if replace_entries:
            new_content = apply_lang_updates(new_content, replace_entries)
        new_content = append_new_lang_entries(new_content, new_entries)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        temp_path = file_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        os.replace(temp_path, file_path)
        return True

    def _has_lang_splitter_mod(self, file_path: str) -> bool:
        mods_dir = os.path.join(os.path.dirname(file_path), "..", "..", "..", "..", "mods")
        if not os.path.isdir(mods_dir):
            return False
        return any(name.lower().startswith("ftbquestslangsplitter") for name in os.listdir(mods_dir))

    def _remind_generic(self, file_path: str) -> None:
        """No FTB Quests Lang Splitter mod detected (or nothing could be
        routed into its split-file layout) -- fall back to the pre-split-
        file-support reminder: the flat file edit MAY be enough on its own
        for a pack without the splitter mod, but some FTBQuests setups
        cache lang data at load time regardless."""
        self.callbacks.on_log(
            f"💡 Квесты переведены в {os.path.basename(file_path)}. Если в игре текст квестов "
            f"всё ещё на английском, попробуйте перезайти в мир/перезапустить игру -- некоторые "
            f"сборки FTB Quests кэшируют переводы при загрузке и не видят изменений файла на лету.",
            "cyan",
        )

    def _remind_to_reload(self) -> None:
        """quests/lang/<code>/... holds FTB Quests Lang Splitter's per-file
        split templates -- confirmed by reading the mod's real source
        (github.com/pietro-lopes/FTB-Quests-Lang-Splitter): on the next
        world load or `/ftbquests reload`, the mod merges any split file
        here that ISN'T already suffixed `_merged` back into the flat
        quests/lang/<code>.snbt, and (confirmed against a real running
        client) that merged cache is what quest rendering actually reads --
        editing the flat file alone was NOT enough; real Russian text in it
        stayed invisible in-game until these split files existed too, even
        across a full PC restart. This processor writes ALREADY-TRANSLATED
        split files directly (unlike the mod's own /langsplitter split,
        which always regenerates them from the English reference and would
        silently wipe an existing translation back to English on the next
        merge -- confirmed the hard way on a real pack), so nothing more
        than a normal reload is needed here."""
        self.callbacks.on_log(
            "💡 Квесты переведены. В игре выйдите в мир и зайдите заново (или выполните "
            "/ftbquests reload) — FTB Quests Lang Splitter сам сольёт готовые файлы. "
            "НЕ запускайте /langsplitter split после этого: команда пересоздаёт файлы из "
            "английского оригинала и сотрёт перевод при следующей перезагрузке.",
            "cyan",
        )
