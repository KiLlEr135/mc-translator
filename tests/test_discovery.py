"""Tests for mc_translator.processors.discovery — locating files to translate."""
import json
import os

from mc_translator.processors.discovery import (
    discover_loose_lang_files,
    discover_mod_content_files,
    discover_quest_lang_target,
    discover_snbt_files,
)

_RU = {"file": "ru_ru", "api": "ru", "deepl": "RU", "name": "Russian", "regex": r"[А-Яа-яЁё]"}
_EN = {"file": "en_us", "api": "en", "deepl": "EN", "name": "English", "regex": r"[a-zA-Z]"}


def test_config_ftbquests_lang_not_double_counted(tmp_path):
    """Regression test: config/ftbquests/lang/en_us.json sits inside both
    LOOSE_JSON_SEARCH_DIRS and the "config" root discover_mod_content_files
    walks. Without excluding it from the "config" walk, the same file was
    discovered by BOTH discover_loose_lang_files (as "loose") and
    discover_mod_content_files (as "generic_json"), so the analyzer/estimator
    double-counted it and the two processors translated (and paid for) it
    twice, with the second silently clobbering the first's output."""
    lang_dir = tmp_path / "config" / "ftbquests" / "lang"
    lang_dir.mkdir(parents=True)
    (lang_dir / "en_us.json").write_text(json.dumps({"a": "b"}), encoding="utf-8")

    loose = discover_loose_lang_files(str(tmp_path))
    mod_content = discover_mod_content_files(str(tmp_path))

    assert len(loose) == 1
    assert not any("ftbquests" in p for p in mod_content["generic_json"])


def test_unrelated_config_json_still_discovered(tmp_path):
    """The exclusion must be scoped to the loose-lang search dirs only --
    an ordinary config/<mod>/settings.json must still be found."""
    mod_dir = tmp_path / "config" / "somemod"
    mod_dir.mkdir(parents=True)
    (mod_dir / "settings.json").write_text(json.dumps({"x": 1}), encoding="utf-8")

    mod_content = discover_mod_content_files(str(tmp_path))
    assert any("settings.json" in p for p in mod_content["generic_json"])


def test_only_en_us_lang_lands_in_lang_bucket(tmp_path):
    """The "lang" bucket feeds LangProcessor, which requires an ENGLISH
    source (same convention as en_us.json elsewhere) -- it computes the
    translated sidecar FROM en_us.lang. A stray non-English .lang file
    loose in the tree (someone's existing ru_ru.lang, a legacy target
    locale another mod already ships) is not a translation source and must
    never be swept in and fed through as if it were -- that would silently
    "translate" already-translated/foreign-language text as if it were
    English (the same corruption class as the FTBQuests quests/lang/ export
    tree, see test_discover_snbt_files_excludes_lang_export_tree)."""
    mod_dir = tmp_path / "mods" / "somemod"
    mod_dir.mkdir(parents=True)
    (mod_dir / "en_us.lang").write_text("key=value", encoding="utf-8")
    (mod_dir / "ru_ru.lang").write_text("key=значение", encoding="utf-8")

    mod_content = discover_mod_content_files(str(tmp_path))
    assert any("en_us.lang" in p for p in mod_content["lang"])
    assert not any("ru_ru.lang" in p for p in mod_content["lang"])


def test_discover_loose_lang_files_missing_dirs_returns_empty(tmp_path):
    assert discover_loose_lang_files(str(tmp_path)) == []


def test_discover_mod_content_files_missing_dirs_returns_empty_buckets(tmp_path):
    result = discover_mod_content_files(str(tmp_path))
    assert all(v == [] for v in result.values())


def test_discover_snbt_files_excludes_lang_export_tree(tmp_path):
    """Regression test (found on a real ATM10 modpack run): FTBQuests' own
    multi-language export tree (quests/lang/<code>/..., one subfolder per
    language including non-English ones) was swept up by the unfiltered
    quests-dir walk and fed through Google Translate as if it were English
    source -- wasting API calls at best, and (confirmed via real corrupted
    output: duplicated/dropped words) corrupting real Spanish/Italian/
    Portuguese quest text shipped with the pack for other players at worst.
    The live quest source (quests/chapters/*.snbt) must still be found."""
    quests = tmp_path / "config" / "ftbquests" / "quests"
    chapters = quests / "chapters"
    chapters.mkdir(parents=True)
    (chapters / "silent_gear.snbt").write_text('{title: "Silent Gear"}', encoding="utf-8")

    for code in ("en_us", "es_es", "it_it", "pt_br"):
        lang_chapters = quests / "lang" / code / "chapters"
        lang_chapters.mkdir(parents=True)
        (lang_chapters / "silent_gear.snbt").write_text('{title: "..."}', encoding="utf-8")
        (quests / "lang" / code / "chapter.snbt").write_text('{title: "..."}', encoding="utf-8")

    found = discover_snbt_files(str(tmp_path))
    assert any("chapters" in p and "silent_gear.snbt" in p and "lang" not in p.split(os.sep) for p in found)
    assert not any("lang" in p.split(os.sep) for p in found)
    assert len(found) == 1


def test_discover_quest_lang_target_finds_target_file_next_to_reference(tmp_path):
    lang_dir = tmp_path / "config" / "ftbquests" / "quests" / "lang"
    lang_dir.mkdir(parents=True)
    (lang_dir / "en_us.snbt").write_text('{\n\tquest.ABC.title: "Hi"\n}\n', encoding="utf-8")

    found = discover_quest_lang_target(str(tmp_path), _RU)

    assert found == [str(lang_dir / "ru_ru.snbt")]


def test_discover_quest_lang_target_empty_without_reference_file(tmp_path):
    lang_dir = tmp_path / "config" / "ftbquests" / "quests" / "lang"
    lang_dir.mkdir(parents=True)
    # No en_us.snbt -- this pack doesn't use FTBQuests' lang-key export at
    # all (quest text lives directly in quests/chapters/*.snbt instead,
    # handled by discover_snbt_files/SnbtProcessor).
    assert discover_quest_lang_target(str(tmp_path), _RU) == []


def test_discover_quest_lang_target_empty_when_target_language_is_english(tmp_path):
    lang_dir = tmp_path / "config" / "ftbquests" / "quests" / "lang"
    lang_dir.mkdir(parents=True)
    (lang_dir / "en_us.snbt").write_text('{\n\tquest.ABC.title: "Hi"\n}\n', encoding="utf-8")

    assert discover_quest_lang_target(str(tmp_path), _EN) == []


def test_buckets_by_extension(tmp_path):
    mod_dir = tmp_path / "mods" / "somemod"
    mod_dir.mkdir(parents=True)
    (mod_dir / "config.yaml").write_text("a: b", encoding="utf-8")
    (mod_dir / "readme.txt").write_text("hello", encoding="utf-8")
    (mod_dir / "settings.cfg").write_text("# comment", encoding="utf-8")

    result = discover_mod_content_files(str(tmp_path))
    assert any("config.yaml" in p for p in result["yaml_toml"])
    assert any("readme.txt" in p for p in result["text"])
    assert any("settings.cfg" in p for p in result["cfg"])


def test_own_resourcepack_output_excluded_from_archive_bucket(tmp_path):
    """Regression test (found on a real ATM10 modpack instance): PackWriter
    writes its resourcepack zip directly into "<mc_dir>/resourcepacks/",
    which is also one of the roots discover_mod_content_files walks for the
    "archive" bucket -- with no exclusion, the tool rediscovered its OWN
    prior output as "an archive to translate" on every later run. A
    same-named zip elsewhere (a real third-party pack) must still be found."""
    rp_dir = tmp_path / "resourcepacks"
    rp_dir.mkdir()
    (rp_dir / "MCTranslator_Pack.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    (rp_dir / "OtherPack.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)

    result = discover_mod_content_files(str(tmp_path), pack_name="MCTranslator_Pack")

    assert not any("MCTranslator_Pack.zip" in p for p in result["archive"])
    assert any("OtherPack.zip" in p for p in result["archive"])


def test_own_output_not_excluded_when_pack_name_omitted(tmp_path):
    """pack_name=None (the default) must change nothing, for any caller not
    yet updated to pass it."""
    rp_dir = tmp_path / "resourcepacks"
    rp_dir.mkdir()
    (rp_dir / "MCTranslator_Pack.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)

    result = discover_mod_content_files(str(tmp_path))

    assert any("MCTranslator_Pack.zip" in p for p in result["archive"])


def test_own_output_numbered_variant_excluded(tmp_path):
    """PackWriter._unique_path falls back to a "_1"/"_2"... suffix when the
    same-named file exists and can't be removed -- that variant must also
    be excluded, not just the exact base name."""
    rp_dir = tmp_path / "resourcepacks"
    rp_dir.mkdir()
    (rp_dir / "MCTranslator_Pack_1.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)

    result = discover_mod_content_files(str(tmp_path), pack_name="MCTranslator_Pack")

    assert not any("MCTranslator_Pack_1.zip" in p for p in result["archive"])


def test_own_output_nested_subfolder_not_excluded(tmp_path):
    """PackWriter never nests its output in a subfolder of resourcepacks/ --
    a same-named zip found one level deeper is a real third-party pack and
    must still be discovered."""
    nested = tmp_path / "resourcepacks" / "somepack"
    nested.mkdir(parents=True)
    (nested / "MCTranslator_Pack.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)

    result = discover_mod_content_files(str(tmp_path), pack_name="MCTranslator_Pack")

    assert any("MCTranslator_Pack.zip" in p for p in result["archive"])
