"""
Discovery functions for locating files to translate.
"""
import os
import re

from mc_translator.constants import LOOSE_JSON_SEARCH_DIRS


def discover_jar_files(mc_dir: str) -> list[str]:
    """Find all JAR files in the mods directory."""
    mods_dir = os.path.join(mc_dir, "mods")
    if not os.path.isdir(mods_dir):
        return []
    return [os.path.join(mods_dir, f) for f in os.listdir(mods_dir) if f.endswith(".jar")]


def discover_loose_lang_files(mc_dir: str) -> list[str]:
    """Find en_us.json lang override files in known locations (KubeJS/defaultconfigs/FTB Quests)."""
    found: list[str] = []
    for rel in LOOSE_JSON_SEARCH_DIRS:
        base = os.path.join(mc_dir, rel.replace("/", os.sep))
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for name in files:
                if name.lower() == "en_us.json":
                    found.append(os.path.join(root, name))
    return found


def _walk_quests_dir(quests: str):
    """os.walk over the FTBQuests quests root, pruning "lang/" -- FTBQuests'
    own multi-language export/reference tree (quests/lang/<code>/..., one
    subfolder per language -- en_us, es_es, it_it, pt_br, pl_pl, and more).
    That tree is never read by FTBQuests for in-game text, and was never
    consulted by SnbtProcessor either (append mode's "already translated"
    check re-reads the SAME live file being written, not a separate
    existing-translation lookup) -- so walking into it only meant
    already-translated (or already-OTHER-LANGUAGE, e.g. Spanish/Italian/
    Portuguese) quest text got fed through Google Translate as if it were
    English source: at best wasted API calls, at worst corrupted real
    foreign-language quest text shipped with the pack for other players
    (confirmed on a real modpack -- garbled/duplicated words in the
    "translated" es_es/it_it/pt_br output). The live, editable quest source
    (quests/chapters/*.snbt, quests/chapter_groups.snbt, quests/data.snbt,
    quests/reward_tables/*.snbt) lives directly under `quests`, outside
    "lang/", and is unaffected by this exclusion.
    """
    for root, dirs, files in os.walk(quests):
        if root == quests and "lang" in dirs:
            dirs.remove("lang")
        yield root, dirs, files


def discover_snbt_files(mc_dir: str) -> list[str]:
    """Find FTBQuests .snbt files (the live quest source -- see
    _walk_quests_dir for why quests/lang/ is excluded)."""
    quests = os.path.join(mc_dir, "config", "ftbquests", "quests")
    if not os.path.isdir(quests):
        return []
    result: list[str] = []
    for root, _, files in _walk_quests_dir(quests):
        for name in files:
            if name.endswith(".snbt"):
                result.append(os.path.join(root, name))
    return result


def discover_quest_lang_target(mc_dir: str, target_lang: dict) -> list[str]:
    """Find the FTBQuests lang-key export file for `target_lang` (e.g.
    quests/lang/ru_ru.snbt), if this pack stores quest text as lang keys
    (quests/lang/en_us.snbt exists as the reference) -- see quest_lang.py's
    module docstring for why this needs a dedicated processor instead of
    just un-excluding quests/lang/ from discover_snbt_files above.

    Returns a single-item list (QuestLangProcessor.process's file_path,
    matching every other processor's per-file bucket shape) or [] when
    there's no lang-key reference to fill from, or the target language IS
    English (nothing to fill it from itself)."""
    lang_dir = os.path.join(mc_dir, "config", "ftbquests", "quests", "lang")
    reference_path = os.path.join(lang_dir, "en_us.snbt")
    if not os.path.isfile(reference_path):
        return []
    target_path = os.path.join(lang_dir, f"{target_lang['file']}.snbt")
    if os.path.normcase(target_path) == os.path.normcase(reference_path):
        return []
    return [target_path]


def discover_mcfunction_files(mc_dir: str) -> list[str]:
    """Find .mcfunction files in data directory (data/<namespace>/functions/...)."""
    data_dir = os.path.join(mc_dir, "data")
    if not os.path.isdir(data_dir):
        return []
    result: list[str] = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".mcfunction"):
                result.append(os.path.join(root, f))
    return result


# (bucket_name, matching extensions) per root directory, for discover_mod_content_files.
# "mods"/"resourcepacks"/"config" were previously each walked once per bucket below
# (up to 8 redundant os.walk passes over the same tree); this walks each root exactly once.
_MOD_CONTENT_ROOT_RULES: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "mods": [
        ("generic_json", (".json",)),
        ("lang", (".lang",)),
        ("yaml_toml", (".yaml", ".yml", ".toml")),
        ("text", (".txt", ".md")),
        ("cfg", (".cfg",)),
        ("xml", (".xml",)),
        ("bat", (".bat",)),
        ("archive", (".zip", ".mcpack", ".mcaddon", ".litemod")),
    ],
    "resourcepacks": [
        ("generic_json", (".json",)),
        ("lang", (".lang",)),
        ("yaml_toml", (".yaml", ".yml", ".toml")),
        ("text", (".txt", ".md")),
        ("cfg", (".cfg",)),
        ("xml", (".xml",)),
        ("bat", (".bat",)),
        ("archive", (".zip", ".mcpack", ".mcaddon", ".litemod")),
    ],
    "config": [
        ("generic_json", (".json",)),
        ("lang", (".lang",)),
        ("yaml_toml", (".yaml", ".yml", ".toml")),
        ("text", (".txt", ".md")),
        ("cfg", (".cfg",)),
        ("xml", (".xml",)),
        ("bat", (".bat",)),
        ("nbt", (".nbt", ".dat")),
    ],
    "saves": [
        ("nbt", (".nbt", ".dat")),
        ("archive", (".zip", ".mcpack", ".mcaddon", ".litemod")),
    ],
    "datapacks": [
        ("nbt", (".nbt", ".dat")),
        ("archive", (".zip", ".mcpack", ".mcaddon", ".litemod")),
    ],
    "assets": [
        ("xml", (".xml",)),
    ],
}


def _own_output_name_patterns(pack_name: str) -> tuple[re.Pattern, re.Pattern]:
    """(resourcepack_regex, datapack_regex) matching the basenames PackWriter
    would use for THIS run's own output, given the currently-configured
    pack_name -- mirrors PackWriter's own sanitization (output/pack_writer.py)
    exactly, including the "_1"/"_2"... suffix PackWriter._unique_path can
    produce when a same-named file already exists and can't be removed."""
    safe = re.sub(r'[\\/*?:"<>|]', "", (pack_name or "").strip() or "MCTranslator_Pack")
    if not safe.lower().endswith(".zip"):
        safe += ".zip"
    stem = re.escape(safe[: -len(".zip")])
    rp_re = re.compile(rf"^{stem}(_\d+)?\.zip$", re.IGNORECASE)
    dp_re = re.compile(rf"^{stem}(_\d+)?_Datapack\.zip$", re.IGNORECASE)
    return rp_re, dp_re


def discover_mod_content_files(mc_dir: str, *, pack_name: str | None = None) -> dict[str, list[str]]:
    """Single-walk discovery for generic_json/lang/yaml_toml/text/cfg/xml/bat/nbt/archive.

    Walks each of mods/resourcepacks/config/saves/datapacks/assets exactly once
    and buckets every matching file by type, instead of one independent os.walk
    per file type (previously up to 8 redundant passes over the same trees).

    `pack_name`, when given, excludes the tool's OWN previous output from the
    "archive" bucket -- PackWriter (output/pack_writer.py) writes its
    resourcepack zip directly into "<mc_dir>/resourcepacks/" and its datapack
    zip directly into "<mc_dir>/config/openloader/data/", both roots this
    function already walks for the "archive" bucket (.zip/.mcpack/.mcaddon/
    .litemod). With no exclusion, a prior run's own output sitting in
    resourcepacks/ was silently rediscovered as "an archive to translate" on
    every later run -- confirmed on a real modpack, where it produced a
    misleading "archive found, skipped" log line even before archives gained
    resourcepack-mode support, and would otherwise start extracting and
    re-merging the tool's own prior output into itself once that support
    existed. Passing None (the default) preserves old behavior exactly for
    any caller that hasn't been updated to pass it."""
    buckets: dict[str, list[str]] = {
        "generic_json": [], "lang": [], "yaml_toml": [], "text": [],
        "cfg": [], "xml": [], "bat": [], "nbt": [], "archive": [],
    }
    # LOOSE_JSON_SEARCH_DIRS' "config/ftbquests/lang" sits *inside* the
    # "config" root walked below, so an en_us.json there would otherwise be
    # discovered here (bucketed as "generic_json") AND separately by
    # discover_loose_lang_files() (bucketed as "loose") -- double-counted in
    # the analyzer/estimator, translated twice (double API cost), and in
    # inplace mode GenericJsonProcessor's write clobbers LooseJsonProcessor's
    # lang-aware one since both resolve to the same on-disk target path.
    # Exclude exactly the files discover_loose_lang_files() already owns.
    loose_lang_prefixes = tuple(
        os.path.normcase(os.path.join(mc_dir, rel.replace("/", os.sep)) + os.sep)
        for rel in LOOSE_JSON_SEARCH_DIRS
    )
    rp_re = dp_re = None
    rp_root = dp_root = None
    if pack_name is not None:
        rp_re, dp_re = _own_output_name_patterns(pack_name)
        rp_root = os.path.normcase(os.path.join(mc_dir, "resourcepacks"))
        # Currently a no-op (the "config" root below has no "archive" bucket
        # rule), but cheap and keeps this exclusion correct if that ever
        # changes -- the datapack zip's own real location is exactly this.
        dp_root = os.path.normcase(os.path.join(mc_dir, "config", "openloader", "data"))
    for root_name, rules in _MOD_CONTENT_ROOT_RULES.items():
        base = os.path.join(mc_dir, root_name)
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for f in files:
                fl = f.lower()
                full_path = os.path.join(root, f)
                if fl == "en_us.json" and os.path.normcase(full_path).startswith(loose_lang_prefixes):
                    continue  # owned by discover_loose_lang_files() instead
                if rp_re is not None and fl.endswith(".zip"):
                    parent = os.path.normcase(root)
                    # Only files sitting DIRECTLY in resourcepacks/ or
                    # config/openloader/data/ -- PackWriter never nests its
                    # output in a subfolder, so a same-named zip found one
                    # level deeper is a real third-party pack, not ours.
                    if parent == rp_root and rp_re.match(f):
                        continue
                    if parent == dp_root and dp_re.match(f):
                        continue
                for bucket, exts in rules:
                    if not fl.endswith(exts):
                        continue
                    if bucket == "lang" and fl != "en_us.lang":
                        # LangProcessor requires an ENGLISH source (it
                        # computes the translated sidecar FROM en_us.lang,
                        # same convention as en_us.json elsewhere) -- an
                        # unrelated .lang file loose in the tree (someone's
                        # existing ru_ru.lang/de_de.lang, a legacy target
                        # locale another mod already ships) is not one and
                        # must never be fed through as if it were.
                        break
                    buckets[bucket].append(full_path)
                    break
    return buckets
