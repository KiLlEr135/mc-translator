"""File discovery + per-file-type bucket-descriptor building for
TranslationJob.run_translation (mc_translator/runtime/job.py). Split out of
job.py purely to keep that file under the project's 500-line guideline --
this is pure code motion, not a behavior change; every function here used to
be inline in run_translation."""
from __future__ import annotations

from dataclasses import dataclass, field

from mc_translator.processors.discovery import (
    discover_jar_files,
    discover_loose_lang_files,
    discover_mcfunction_files,
    discover_mod_content_files,
    discover_quest_lang_target,
    discover_snbt_files,
)
from mc_translator.runtime.manifest import jar_fingerprint, load_estimate_cache, load_jar_manifest


@dataclass
class DiscoveredFiles:
    """Every file list run_translation's buckets/estimator/manifest
    bookkeeping need, plus the bits of jar-skip state that outlive
    discovery (all_jars/jar_manifest/estimate_cache are still needed in
    run_translation's `finally` block to prune manifests after the run)."""

    jars: list = field(default_factory=list)
    all_jars: list = field(default_factory=list)
    jar_manifest: dict = field(default_factory=dict)
    skipped_unchanged: int = 0
    loose: list = field(default_factory=list)
    snbt: list = field(default_factory=list)
    quest_lang: list = field(default_factory=list)
    generic_json: list = field(default_factory=list)
    lang_files: list = field(default_factory=list)
    nbt_files: list = field(default_factory=list)
    yaml_toml_files: list = field(default_factory=list)
    text_files: list = field(default_factory=list)
    archives: list = field(default_factory=list)
    cfg_files: list = field(default_factory=list)
    mcfunction_files: list = field(default_factory=list)
    xml_files: list = field(default_factory=list)
    bat_files: list = field(default_factory=list)
    estimate_cache: dict = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (
            self.jars or self.loose or self.snbt or self.quest_lang or self.generic_json or self.lang_files
            or self.nbt_files or self.yaml_toml_files or self.text_files or self.archives
            or self.cfg_files or self.mcfunction_files or self.xml_files or self.bat_files
        )

    def total_items(self) -> int:
        return (
            len(self.jars) + len(self.loose) + len(self.snbt) + len(self.quest_lang) + len(self.generic_json)
            + len(self.lang_files) + len(self.nbt_files) + len(self.yaml_toml_files)
            + len(self.text_files) + len(self.archives)
            + len(self.cfg_files) + len(self.mcfunction_files) + len(self.xml_files)
            + len(self.bat_files)
        )


def discover_translation_files(options, lang: dict, on_log) -> DiscoveredFiles:
    """All file discovery + the jar unchanged-skip filter + per-file-type
    gating (GUI's "Типы файлов" tab) for run_translation."""
    all_jars = discover_jar_files(options.mc_dir) if (options.translate_mods or options.translate_books) else []

    # Skip jars that are byte-identical to last run AND were processed with
    # the same options (mode="force" always reprocesses everything).
    # Only safe in "inplace" output: in "resourcepack" mode, jar_manifest's
    # skip-translation path has never been wired to also copy that jar's
    # PREVIOUS entries into the new pack -- PackWriter now seeds each new
    # pack from whatever a prior run wrote at the same path (see
    # PackWriter._read_existing_entries), so unmodified jars' old output does
    # survive a normal rerun untouched, but a skipped jar still wouldn't get
    # re-checked against jar_manifest's fingerprint here without that wiring,
    # so leave this gated to "inplace" until it's actually implemented.
    skip_unchanged = (
        bool(all_jars) and options.process_mode != "force" and options.output_mode == "inplace"
    )
    jar_manifest = load_jar_manifest(options.mc_dir, lang["file"]) if skip_unchanged else {}
    # Per-mc_dir, per-language cache of StringEstimator's per-jar string
    # count -- see manifest.load_estimate_cache's docstring. Unlike
    # jar_manifest above, this isn't gated on output_mode/process_mode: it
    # only speeds up the "Подсчёт строк" pass (total_strings), never what
    # gets translated, so it's safe to reuse in every mode -- including
    # "resourcepack", where jar_manifest's skip-translation path is
    # unavailable (PackWriter always rebuilds the pack from scratch) but
    # re-parsing every jar just to count strings is still pure wasted work
    # when the jar hasn't changed.
    estimate_cache = load_estimate_cache(options.mc_dir, lang["file"]) if all_jars else {}
    jars = []
    skipped_unchanged = 0
    for path in all_jars:
        fp = jar_fingerprint(
            path,
            translate_mods=options.translate_mods,
            translate_books=options.translate_books,
            output_mode=options.output_mode,
        )
        if skip_unchanged and fp is not None and jar_manifest.get(path) == fp:
            skipped_unchanged += 1
            continue
        jars.append(path)
    if skipped_unchanged:
        on_log(f"⏭ Без изменений с прошлого раза: {skipped_unchanged} мод(ов) — пропущено", "dim")

    loose = discover_loose_lang_files(options.mc_dir) if (options.translate_mods or options.translate_quests) else []
    snbt = discover_snbt_files(options.mc_dir) if options.translate_quests else []
    quest_lang = discover_quest_lang_target(options.mc_dir, lang) if options.translate_quests else []
    # New discoveries under translate_mods flag (single-walk, bucketed by type)
    mod_content = (
        discover_mod_content_files(options.mc_dir, pack_name=options.pack_name)
        if options.translate_mods else {}
    )
    generic_json = mod_content.get("generic_json", [])
    lang_files = mod_content.get("lang", [])
    nbt_files = mod_content.get("nbt", [])
    yaml_toml_files = mod_content.get("yaml_toml", [])
    text_files = mod_content.get("text", [])
    archives = mod_content.get("archive", [])
    cfg_files = mod_content.get("cfg", [])
    mcfunction_files = discover_mcfunction_files(options.mc_dir) if options.translate_mods else []
    xml_files = mod_content.get("xml", [])
    bat_files = mod_content.get("bat", [])

    # Per-file-type gating (GUI's "Типы файлов" settings tab) -- an
    # additional AND-filter on top of translate_mods/books/quests above, not
    # a replacement for them. Missing key defaults to enabled, so existing
    # settings.ini files without a [FILETYPES] section (or a scope that
    # predates a newly-added type) behave exactly as before.
    file_types = options.file_types or {}
    if not file_types.get("generic_json", True):
        generic_json = []
    if not file_types.get("lang", True):
        lang_files = []
    if not file_types.get("nbt", True):
        nbt_files = []
    if not file_types.get("yaml_toml", True):
        yaml_toml_files = []
    if not file_types.get("text", True):
        text_files = []
    if not file_types.get("archive", True):
        archives = []
    if not file_types.get("cfg", True):
        cfg_files = []
    if not file_types.get("mcfunction", True):
        mcfunction_files = []
    if not file_types.get("xml", True):
        xml_files = []
    if not file_types.get("bat", True):
        bat_files = []

    return DiscoveredFiles(
        jars=jars, all_jars=all_jars, jar_manifest=jar_manifest, skipped_unchanged=skipped_unchanged,
        loose=loose, snbt=snbt, quest_lang=quest_lang, generic_json=generic_json, lang_files=lang_files,
        nbt_files=nbt_files, yaml_toml_files=yaml_toml_files, text_files=text_files, archives=archives,
        cfg_files=cfg_files, mcfunction_files=mcfunction_files, xml_files=xml_files, bat_files=bat_files,
        estimate_cache=estimate_cache,
    )


def build_buckets(found: DiscoveredFiles, lang: dict, options, pack_writer, processors: dict, on_jar_done) -> list:
    """Build the (items, process_one, label, on_item_done) descriptor list
    run_translation's bucket loop iterates over -- one entry per file type.
    `processors` is a dict of already-constructed processor instances keyed
    by short name (jar/loose/snbt/quest_lang/generic_json/lang/nbt/
    yaml_toml/text/archive/cfg/mcfunction/xml/bat)."""
    return [
        (
            found.jars,
            lambda path: processors["jar"].process(
                path, target_lang=lang, mode=options.process_mode, output_mode=options.output_mode,
                translate_mods=options.translate_mods, translate_books=options.translate_books,
                pack_writer=pack_writer,
            ),
            "Модов", on_jar_done,
        ),
        (
            found.loose,
            lambda path: processors["loose"].process(
                path, options.mc_dir, target_lang=lang, mode=options.process_mode,
                output_mode=options.output_mode, pack_writer=pack_writer,
            ),
            "Словарей", None,
        ),
        (
            found.snbt,
            lambda path: processors["snbt"].process(path, target_lang=lang, mode=options.process_mode),
            "Квестов", None,
        ),
        (
            found.quest_lang,
            lambda path: processors["quest_lang"].process(path, target_lang=lang, mode=options.process_mode),
            "Квестов (lang-ключи)", None,
        ),
        (
            found.generic_json,
            lambda path: processors["generic_json"].process(
                path, target_lang=lang, mode=options.process_mode, output_mode=options.output_mode,
                pack_writer=pack_writer, mc_dir=options.mc_dir,
            ),
            "JSON", None,
        ),
        (
            found.lang_files,
            lambda path: processors["lang"].process(
                path, target_lang=lang, mode=options.process_mode, output_mode=options.output_mode,
                pack_writer=pack_writer, mc_dir=options.mc_dir, mc_version=options.mc_version,
            ),
            "Lang", None,
        ),
        (
            found.nbt_files,
            lambda path: processors["nbt"].process(
                path, target_lang=lang, mode=options.process_mode, output_mode=options.output_mode,
                pack_writer=pack_writer, mc_dir=options.mc_dir,
            ),
            "NBT", None,
        ),
        (
            found.yaml_toml_files,
            lambda path: processors["yaml_toml"].process(
                path, target_lang=lang, mode=options.process_mode, output_mode=options.output_mode,
                pack_writer=pack_writer, mc_dir=options.mc_dir,
            ),
            "YAML/TOML", None,
        ),
        (
            found.text_files,
            lambda path: processors["text"].process(
                path, target_lang=lang, mode=options.process_mode, output_mode=options.output_mode,
                pack_writer=pack_writer, mc_dir=options.mc_dir,
            ),
            "Text", None,
        ),
        (
            found.archives,
            lambda path: processors["archive"].process(
                path, target_lang=lang, mode=options.process_mode, output_mode=options.output_mode,
                pack_writer=pack_writer, mc_dir=options.mc_dir,
            ),
            "Архив", None,
        ),
        (
            found.cfg_files,
            lambda path: processors["cfg"].process(
                path, target_lang=lang, mode=options.process_mode, output_mode=options.output_mode,
                pack_writer=pack_writer, mc_dir=options.mc_dir,
            ),
            "CFG", None,
        ),
        (
            found.mcfunction_files,
            lambda path: processors["mcfunction"].process(
                path, target_lang=lang, mode=options.process_mode, output_mode=options.output_mode,
                pack_writer=pack_writer, mc_dir=options.mc_dir,
            ),
            "McFunction", None,
        ),
        (
            found.xml_files,
            lambda path: processors["xml"].process(
                path, target_lang=lang, mode=options.process_mode, output_mode=options.output_mode,
                pack_writer=pack_writer, mc_dir=options.mc_dir,
            ),
            "XML", None,
        ),
        (
            found.bat_files,
            lambda path: processors["bat"].process(
                path, target_lang=lang, mode=options.process_mode, output_mode=options.output_mode,
                pack_writer=pack_writer, mc_dir=options.mc_dir,
            ),
            "BAT", None,
        ),
    ]
