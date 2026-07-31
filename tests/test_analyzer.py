"""Tests for mc_translator.processors.analyzer.ModpackAnalyzer -- covers two
regressions found and fixed in this project:

- _analyze_books' is_jb (JSON) and XML-book branch used to omit the
  is_localized_guide_path guard that their MD sibling already applied, so an
  already-localized quest/research JSON or XML bundled in a jar (under a
  "_xx_xx/" path segment) was counted as translatable English source on the
  Analyze screen, even though the real translation run (jar.py) never
  touches such paths.
- The "lang" discovery bucket (real key=value en_us.lang files) used to be
  routed through _analyze_lang_simple, a JSON parser -- it always raised on
  the first line and was silently swallowed, so the file's content never
  appeared on the Analyze screen at all.
"""
import json
import os
import zipfile

import pytest

from mc_translator.output.pack_writer import expected_pack_paths
from mc_translator.processors.analyzer import ModpackAnalyzer
from mc_translator.processors.resourcepack_path import compute_target_paths


def _lang():
    return {"file": "ru_ru", "regex": r"[А-Яа-яЁё]"}


def _write_pack_zip(zip_path, entries):
    """Build a resourcepack/datapack zip at `zip_path` (as
    expected_pack_paths(mc_dir, pack_name) would compute) containing
    {internal_path: bytes} -- the same shape PackWriter.write() produces,
    built directly with zipfile rather than a real PackWriter so tests don't
    need a full translation run to populate one."""
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as z:
        for internal_path, data in entries.items():
            z.writestr(internal_path, data)


def _write_resourcepack_zip(mc_dir, pack_name, entries):
    """Convenience wrapper around _write_pack_zip: resolves the resourcepack
    (not datapack) zip path for `pack_name` under `mc_dir` via
    expected_pack_paths and writes `entries` into it."""
    rp_path, _ = expected_pack_paths(str(mc_dir), pack_name)
    _write_pack_zip(rp_path, entries)


def test_analyze_books_skips_localized_guide_json(tmp_path, job_state):
    """A jar-bundled quest/research JSON already living under a "_xx_xx/"
    locale segment must not be counted as translatable English source."""
    jar_path = tmp_path / "testmod.jar"
    with zipfile.ZipFile(jar_path, "w") as z:
        z.writestr(
            "assets/testmod/quests/_es_es/chapter1.json",
            json.dumps({"title": "Capitulo Uno", "pages": [{"text": "Algun texto en espanol"}]}),
        )

    rows = []
    analyzer = ModpackAnalyzer(job_state)
    en, tr = analyzer.analyze(
        str(tmp_path),
        target_lang=_lang(),
        translate_mods=False,
        translate_books=True,
        translate_quests=False,
        on_row=lambda *a: rows.append(a),
        on_log=lambda *a: None,
        on_status=lambda *a: None,
    )

    assert en == 0
    assert tr == 0
    assert rows == []


def test_analyze_lang_bucket_parses_key_value_format(tmp_path, job_state):
    """A real (pre-1.13) key=value en_us.lang file must be parsed as such,
    not fed through the JSON parser used for loose en_us.json files."""
    lang_dir = tmp_path / "mods" / "testmod" / "lang"
    lang_dir.mkdir(parents=True)
    (lang_dir / "en_us.lang").write_text(
        "# a comment\nitem.testmod.thing=A Cool Thing\n", encoding="utf-8"
    )

    rows = []
    analyzer = ModpackAnalyzer(job_state)
    en, tr = analyzer.analyze(
        str(tmp_path),
        target_lang=_lang(),
        translate_mods=True,
        translate_books=False,
        translate_quests=False,
        on_row=lambda *a: rows.append(a),
        on_log=lambda *a: None,
        on_status=lambda *a: None,
    )

    assert en == 1
    assert tr == 0
    assert len(rows) == 1


def test_analyze_lang_bucket_counts_existing_sidecar_translation(tmp_path, job_state):
    lang_dir = tmp_path / "mods" / "testmod" / "lang"
    lang_dir.mkdir(parents=True)
    (lang_dir / "en_us.lang").write_text("item.testmod.thing=A Cool Thing\n", encoding="utf-8")
    (lang_dir / "ru_ru.lang").write_text("item.testmod.thing=Классная штука\n", encoding="utf-8")

    rows = []
    analyzer = ModpackAnalyzer(job_state)
    en, tr = analyzer.analyze(
        str(tmp_path),
        target_lang=_lang(),
        translate_mods=True,
        translate_books=False,
        translate_quests=False,
        on_row=lambda *a: rows.append(a),
        on_log=lambda *a: None,
        on_status=lambda *a: None,
    )

    assert en == 1
    assert tr == 1


def test_analyze_threads_pack_name_into_discovery(tmp_path, job_state, monkeypatch):
    """analyze() must forward pack_name to discover_mod_content_files so the
    "Analyze" screen's file counts exclude the tool's own prior resourcepack
    output the same way a real translation run does (see
    tests/test_discovery.py's self-consumption regression tests)."""
    import mc_translator.processors.analyzer as analyzer_module

    captured = {}
    real = analyzer_module.discover_mod_content_files

    def spy(mc_dir, **kwargs):
        captured.update(kwargs)
        return real(mc_dir, **kwargs)

    monkeypatch.setattr(analyzer_module, "discover_mod_content_files", spy)

    analyzer = ModpackAnalyzer(job_state)
    analyzer.analyze(
        str(tmp_path),
        target_lang=_lang(),
        translate_mods=True,
        translate_books=False,
        translate_quests=False,
        pack_name="MCTranslator_Pack",
        on_row=lambda *a: None,
        on_log=lambda *a: None,
        on_status=lambda *a: None,
    )

    assert captured.get("pack_name") == "MCTranslator_Pack"


# --- output_mode="resourcepack" fallback (_PackIndex) -----------------------
#
# The tests above all rely on an on-disk sidecar/inplace file next to the
# source (output_mode="inplace"). The tests below cover the actual
# production case (settings.ini output_mode=resourcepack, the app's default
# and recommended mode): no sidecar is ever written on disk, so translated
# content only exists inside the current resourcepack/datapack zip. Each
# test writes translated content ONLY into the pack zip (never a sidecar) so
# a false-positive pass would require the analyzer to still be finding a
# sidecar it shouldn't -- it isn't there.


def test_analyze_lang_simple_resourcepack_fallback(tmp_path, job_state):
    """Loose en_us.json (KubeJS/defaultconfigs/FTB Quests location, see
    LOOSE_JSON_SEARCH_DIRS) with no on-disk sidecar: the translated
    counterpart only exists in the current resourcepack zip. Must be found
    via the same internal-path formula the real writer (loose_json.py) uses
    (compute_target_paths), not report 0%."""
    mc_dir = tmp_path
    src_dir = mc_dir / "kubejs" / "assets" / "mymod" / "lang"
    src_dir.mkdir(parents=True)
    src_path = src_dir / "en_us.json"
    src_path.write_text(json.dumps({"item.testmod.thing": "A Cool Thing"}), encoding="utf-8")

    pack_name = "MyPack"
    _, _, internal_path = compute_target_paths(str(src_path), "ru_ru", str(mc_dir))
    rp_path, _ = expected_pack_paths(str(mc_dir), pack_name)
    _write_pack_zip(
        rp_path,
        {internal_path: json.dumps({"item.testmod.thing": "Классная штука"}, ensure_ascii=False).encode("utf-8")},
    )
    assert not (src_dir / "ru_ru.json").exists()  # no sidecar -- proves the pack, not disk, is the source

    rows = []
    analyzer = ModpackAnalyzer(job_state)
    en, tr = analyzer.analyze(
        str(mc_dir),
        target_lang=_lang(),
        translate_mods=True,
        translate_books=False,
        translate_quests=False,
        pack_name=pack_name,
        on_row=lambda *a: rows.append(a),
        on_log=lambda *a: None,
        on_status=lambda *a: None,
    )

    assert en == 1
    assert tr == 1
    assert len(rows) == 1


def test_analyze_legacy_lang_file_resourcepack_fallback(tmp_path, job_state):
    """Real key=value en_us.lang, no on-disk sidecar -- translated content
    only exists in the current resourcepack zip."""
    mc_dir = tmp_path
    lang_dir = mc_dir / "mods" / "testmod" / "lang"
    lang_dir.mkdir(parents=True)
    src_path = lang_dir / "en_us.lang"
    src_path.write_text("item.testmod.thing=A Cool Thing\n", encoding="utf-8")

    pack_name = "MyPack"
    _, _, internal_path = compute_target_paths(str(src_path), "ru_ru", str(mc_dir))
    assert not internal_path.lower().startswith("data/")  # .lang -> resourcepack zip, not datapack
    rp_path, _ = expected_pack_paths(str(mc_dir), pack_name)
    _write_pack_zip(
        rp_path,
        {internal_path: "item.testmod.thing=Классная штука\n".encode()},
    )
    assert not (lang_dir / "ru_ru.lang").exists()  # no sidecar

    rows = []
    analyzer = ModpackAnalyzer(job_state)
    en, tr = analyzer.analyze(
        str(mc_dir),
        target_lang=_lang(),
        translate_mods=True,
        translate_books=False,
        translate_quests=False,
        pack_name=pack_name,
        on_row=lambda *a: rows.append(a),
        on_log=lambda *a: None,
        on_status=lambda *a: None,
    )

    assert en == 1
    assert tr == 1


@pytest.mark.parametrize(
    "rel_source, source_content, translated_content, kind",
    [
        (
            "mods/testmod/data.json",
            '{\n  "greeting": "Hello World"\n}\n',
            '{\n  "greeting": "Привет Мир"\n}\n',
            "json",
        ),
        (
            "mods/testmod/config.yaml",
            "greeting: Hello World\n",
            "greeting: Привет Мир\n",
            "yaml_toml",
        ),
        (
            "mods/testmod/strings.xml",
            '<string name="greeting">Hello World</string>\n',
            '<string name="greeting">Привет Мир</string>\n',
            "xml",
        ),
        (
            "mods/testmod/notes.txt",
            "Hello World\n",
            "Привет Мир\n",
            "text",
        ),
        (
            "mods/testmod/settings.cfg",
            "# Hello World comment\nsomekey=value\n",
            "# Привет Мир comment\nsomekey=value\n",
            "cfg",
        ),
        (
            "mods/testmod/run.bat",
            "REM Hello World comment\necho hi\n",
            "REM Привет Мир comment\necho hi\n",
            "bat",
        ),
    ],
)
def test_analyze_generic_file_resourcepack_fallback(
    tmp_path, job_state, rel_source, source_content, translated_content, kind
):
    """Loose generic files (json/yaml_toml/xml/text/cfg/bat) discovered
    outside any jar (discover_mod_content_files), no on-disk sidecar --
    translated content only exists in the current resourcepack zip. Must be
    found via compute_target_paths' internal_path, the same one
    generic_json.py/yaml_toml.py/xml.py/line_processor.py actually write to
    in resourcepack mode."""
    mc_dir = tmp_path
    src_path = mc_dir / rel_source
    src_path.parent.mkdir(parents=True, exist_ok=True)
    src_path.write_text(source_content, encoding="utf-8")

    pack_name = "MyPack"
    _, _, internal_path = compute_target_paths(str(src_path), "ru_ru", str(mc_dir))
    assert not internal_path.lower().startswith("data/")  # these all live under assets/mods, not data/
    rp_path, _ = expected_pack_paths(str(mc_dir), pack_name)
    _write_pack_zip(rp_path, {internal_path: translated_content.encode("utf-8")})

    rows = []
    analyzer = ModpackAnalyzer(job_state)
    en, tr = analyzer.analyze(
        str(mc_dir),
        target_lang=_lang(),
        translate_mods=True,
        translate_books=False,
        translate_quests=False,
        pack_name=pack_name,
        on_row=lambda *a: rows.append(a),
        on_log=lambda *a: None,
        on_status=lambda *a: None,
    )

    assert en >= 1, f"expected at least one translatable line for kind={kind}"
    assert tr >= 1, f"pack fallback did not find the translation for kind={kind}"


def test_analyze_mcfunction_datapack_fallback(tmp_path, job_state):
    """mcfunction files live under data/<namespace>/functions/ -- per
    PackWriter.handle_for_path, a "data/"-prefixed internal_path routes to
    the DATAPACK zip, not the resourcepack zip. The analyzer's pack fallback
    (_PackIndex._zip_for) must agree, or resourcepack-mode readiness for
    mcfunction content would always read 0% even with real translated
    content sitting in the datapack zip."""
    mc_dir = tmp_path
    func_dir = mc_dir / "data" / "testns" / "functions"
    func_dir.mkdir(parents=True)
    src_path = func_dir / "foo.mcfunction"
    src_path.write_text("# A comment line here\nsay hello\n", encoding="utf-8")

    pack_name = "MyPack"
    _, _, internal_path = compute_target_paths(str(src_path), "ru_ru", str(mc_dir))
    assert internal_path.lower().startswith("data/")  # sanity: this is exactly the case under test
    _, dp_path = expected_pack_paths(str(mc_dir), pack_name)
    _write_pack_zip(
        dp_path,
        {internal_path: "# Комментарий тут\nsay hello\n".encode()},
    )

    rows = []
    analyzer = ModpackAnalyzer(job_state)
    en, tr = analyzer.analyze(
        str(mc_dir),
        target_lang=_lang(),
        translate_mods=True,
        translate_books=False,
        translate_quests=False,
        pack_name=pack_name,
        on_row=lambda *a: rows.append(a),
        on_log=lambda *a: None,
        on_status=lambda *a: None,
    )

    assert en == 1
    assert tr == 1


def test_analyze_mcfunction_ignores_resourcepack_zip(tmp_path, job_state):
    """Mirror image of the above: the SAME translated content sitting in the
    resourcepack zip (the wrong zip for a "data/"-prefixed path) must NOT be
    picked up. This guards against a _zip_for implementation that ignores
    the "data/" prefix and always checks the resourcepack zip (which would
    make the datapack-routing test above pass for the wrong reason if it
    also happened to fall back to the rp zip)."""
    mc_dir = tmp_path
    func_dir = mc_dir / "data" / "testns" / "functions"
    func_dir.mkdir(parents=True)
    src_path = func_dir / "foo.mcfunction"
    src_path.write_text("# A comment line here\nsay hello\n", encoding="utf-8")

    pack_name = "MyPack"
    _, _, internal_path = compute_target_paths(str(src_path), "ru_ru", str(mc_dir))
    rp_path, _ = expected_pack_paths(str(mc_dir), pack_name)
    _write_pack_zip(
        rp_path,  # deliberately the WRONG zip for a "data/"-prefixed path
        {internal_path: "# Комментарий тут\nsay hello\n".encode()},
    )

    rows = []
    analyzer = ModpackAnalyzer(job_state)
    en, tr = analyzer.analyze(
        str(mc_dir),
        target_lang=_lang(),
        translate_mods=True,
        translate_books=False,
        translate_quests=False,
        pack_name=pack_name,
        on_row=lambda *a: rows.append(a),
        on_log=lambda *a: None,
        on_status=lambda *a: None,
    )

    assert en == 1
    assert tr == 0


def test_analyze_no_pack_name_and_no_existing_pack_does_not_crash(tmp_path, job_state):
    """pack_name=None (never passed at all): expected_pack_paths defaults to
    "MCTranslator_Pack", and since no prior run ever created that zip, the
    analyzer must fall back to "no translation found" (not crash) for every
    content kind."""
    mc_dir = tmp_path
    lang_dir = mc_dir / "mods" / "testmod" / "lang"
    lang_dir.mkdir(parents=True)
    (lang_dir / "en_us.lang").write_text("item.testmod.thing=A Cool Thing\n", encoding="utf-8")

    kubejs_dir = mc_dir / "kubejs" / "assets" / "mymod" / "lang"
    kubejs_dir.mkdir(parents=True)
    (kubejs_dir / "en_us.json").write_text(json.dumps({"item.testmod.thing": "A Cool Thing"}), encoding="utf-8")

    func_dir = mc_dir / "data" / "testns" / "functions"
    func_dir.mkdir(parents=True)
    (func_dir / "foo.mcfunction").write_text("# A comment line here\nsay hello\n", encoding="utf-8")

    rp_path, dp_path = expected_pack_paths(str(mc_dir), None)
    assert not os.path.exists(rp_path)
    assert not os.path.exists(dp_path)

    rows = []
    analyzer = ModpackAnalyzer(job_state)
    en, tr = analyzer.analyze(
        str(mc_dir),
        target_lang=_lang(),
        translate_mods=True,
        translate_books=False,
        translate_quests=False,
        on_row=lambda *a: rows.append(a),
        on_log=lambda *a: None,
        on_status=lambda *a: None,
    )

    assert en == 3
    assert tr == 0
    assert len(rows) == 3


# --- output_mode="resourcepack" fallback for JAR-EMBEDDED content -----------
#
# The tests above cover loose/generic/mcfunction files. The tests below cover
# content that lives INSIDE a mod jar -- mod interface strings (_analyze_
# mods_ui) and Patchouli/GuideME-style books (_analyze_books, JSON/MD/XML
# variants). Real writers (jar.py's _process_lang_entry/_process_legacy_lang_
# entry, jar_books.py's _process_book_json/_process_book_md/_process_book_xml)
# never modify the source jar at all in resourcepack mode -- every test here
# builds the jar with ONLY the English source and writes the translated
# counterpart directly into a resourcepack zip built at the exact path/
# internal-path a real PackWriter-backed run would use, so a false-positive
# pass would require the analyzer to still be finding something inside the
# untouched jar -- it isn't there.


def test_analyze_mods_ui_resourcepack_fallback(tmp_path, job_state):
    """Jar-embedded mod interface (assets/<modid>/lang/en_us.json), no
    translated locale file inside the jar -- the translation only exists in
    the current resourcepack zip. Must be found via the same tr_path formula
    the real writer (jar.py's _process_lang_entry) uses: re.sub(r"en_us\\.
    json$", target_file, item.filename, IGNORECASE)."""
    mc_dir = tmp_path
    mods_dir = mc_dir / "mods"
    mods_dir.mkdir(parents=True)
    jar_path = mods_dir / "testmod.jar"
    en_internal = "assets/testmod/lang/en_us.json"
    with zipfile.ZipFile(jar_path, "w") as z:
        z.writestr(
            en_internal,
            json.dumps({"item.testmod.thing": "A Cool Thing", "item.testmod.other": "Another Thing"}),
        )

    pack_name = "MyPack"
    tr_internal = "assets/testmod/lang/ru_ru.json"
    rp_path, _ = expected_pack_paths(str(mc_dir), pack_name)
    _write_pack_zip(
        rp_path,
        {
            tr_internal: json.dumps(
                {"item.testmod.thing": "Классная штука", "item.testmod.other": "Другая вещь"},
                ensure_ascii=False,
            ).encode("utf-8")
        },
    )

    rows = []
    analyzer = ModpackAnalyzer(job_state)
    en, tr = analyzer.analyze(
        str(mc_dir),
        target_lang=_lang(),
        translate_mods=True,
        translate_books=False,
        translate_quests=False,
        pack_name=pack_name,
        on_row=lambda *a: rows.append(a),
        on_log=lambda *a: None,
        on_status=lambda *a: None,
    )

    assert en == 2
    assert tr == 2
    assert len(rows) == 1
    assert rows[0][2] == "Интерфейс"


def test_analyze_mods_ui_no_resourcepack_zip_is_graceful(tmp_path, job_state):
    """First-ever analyze before any translation run: no resourcepack zip
    exists on disk at all. Must not crash and must behave like the old
    (source-file-only) behavior -- en counted, tr stays 0."""
    mc_dir = tmp_path
    mods_dir = mc_dir / "mods"
    mods_dir.mkdir(parents=True)
    jar_path = mods_dir / "testmod.jar"
    with zipfile.ZipFile(jar_path, "w") as z:
        z.writestr(
            "assets/testmod/lang/en_us.json",
            json.dumps({"item.testmod.thing": "A Cool Thing"}),
        )

    pack_name = "MyPack"
    rp_path, dp_path = expected_pack_paths(str(mc_dir), pack_name)
    assert not os.path.exists(rp_path)
    assert not os.path.exists(dp_path)

    rows = []
    analyzer = ModpackAnalyzer(job_state)
    en, tr = analyzer.analyze(
        str(mc_dir),
        target_lang=_lang(),
        translate_mods=True,
        translate_books=True,
        translate_quests=False,
        pack_name=pack_name,
        on_row=lambda *a: rows.append(a),
        on_log=lambda *a: None,
        on_status=lambda *a: None,
    )

    assert en == 1
    assert tr == 0


def test_analyze_mods_ui_case_sensitive_jar_path_matches_pack_case(tmp_path, job_state):
    """A jar with a mixed-case internal path (assets/SomeMod/Lang/en_us.json)
    must still find its pack translation, which the real writer stores under
    the ORIGINAL-case path (jar.py computes tr_path from item.filename, not a
    lowercased copy) -- not silently miss it due to a case-mismatch bug in
    the analyzer's pack lookup."""
    mc_dir = tmp_path
    mods_dir = mc_dir / "mods"
    mods_dir.mkdir(parents=True)
    jar_path = mods_dir / "testmod.jar"
    en_internal = "assets/SomeMod/Lang/en_us.json"
    with zipfile.ZipFile(jar_path, "w") as z:
        z.writestr(en_internal, json.dumps({"item.somemod.thing": "A Cool Thing"}))

    pack_name = "MyPack"
    # Correctly-cased path, exactly as jar.py's _process_lang_entry would
    # compute it (re.sub on the ORIGINAL-case item.filename).
    tr_internal_correct_case = "assets/SomeMod/Lang/ru_ru.json"
    rp_path, _ = expected_pack_paths(str(mc_dir), pack_name)
    _write_pack_zip(
        rp_path,
        {
            tr_internal_correct_case: json.dumps(
                {"item.somemod.thing": "Классная штука"}, ensure_ascii=False
            ).encode("utf-8")
        },
    )

    rows = []
    analyzer = ModpackAnalyzer(job_state)
    en, tr = analyzer.analyze(
        str(mc_dir),
        target_lang=_lang(),
        translate_mods=True,
        translate_books=False,
        translate_quests=False,
        pack_name=pack_name,
        on_row=lambda *a: rows.append(a),
        on_log=lambda *a: None,
        on_status=lambda *a: None,
    )

    assert en == 1
    assert tr == 1


def test_analyze_books_json_resourcepack_fallback(tmp_path, job_state):
    """Patchouli-style book JSON (assets/<modid>/patchouli_books/.../en_us/
    entries/entry1.json), no translated variant inside the jar -- only in the
    current resourcepack zip. Must be found via the real writer's (jar_books.
    py's _process_book_json) tr_path formula: re.sub(r"/en_us/", "/<lang>/",
    item.filename, IGNORECASE)."""
    mc_dir = tmp_path
    mods_dir = mc_dir / "mods"
    mods_dir.mkdir(parents=True)
    jar_path = mods_dir / "testmod.jar"
    en_internal = "assets/testmod/patchouli_books/testbook/en_us/entries/entry1.json"
    with zipfile.ZipFile(jar_path, "w") as z:
        z.writestr(
            en_internal,
            json.dumps({"title": "Some English Title", "pages": [{"text": "Some page text in english"}]}),
        )

    pack_name = "MyPack"
    tr_internal = "assets/testmod/patchouli_books/testbook/ru_ru/entries/entry1.json"
    rp_path, _ = expected_pack_paths(str(mc_dir), pack_name)
    _write_pack_zip(
        rp_path,
        {
            tr_internal: json.dumps(
                {"title": "Какой-то русский текст", "pages": [{"text": "Какой-то текст на странице"}]},
                ensure_ascii=False,
            ).encode("utf-8")
        },
    )

    rows = []
    analyzer = ModpackAnalyzer(job_state)
    en, tr = analyzer.analyze(
        str(mc_dir),
        target_lang=_lang(),
        translate_mods=False,
        translate_books=True,
        translate_quests=False,
        pack_name=pack_name,
        on_row=lambda *a: rows.append(a),
        on_log=lambda *a: None,
        on_status=lambda *a: None,
    )

    assert en == 2
    assert tr == 2
    assert len(rows) == 1
    assert rows[0][2] == "Книга(JSON)"


def test_analyze_books_json_no_resourcepack_zip_is_graceful(tmp_path, job_state):
    """Book counterpart of test_analyze_mods_ui_no_resourcepack_zip_is_
    graceful: first-ever analyze before any translation run, no resourcepack
    zip on disk at all. Must not crash and must behave like the old
    (source-file-only) behavior for _analyze_books' JSON branch specifically
    -- en counted, tr stays 0."""
    mc_dir = tmp_path
    mods_dir = mc_dir / "mods"
    mods_dir.mkdir(parents=True)
    jar_path = mods_dir / "testmod.jar"
    en_internal = "assets/testmod/patchouli_books/testbook/en_us/entries/entry1.json"
    with zipfile.ZipFile(jar_path, "w") as z:
        z.writestr(
            en_internal,
            json.dumps({"title": "Some English Title", "pages": [{"text": "Some page text in english"}]}),
        )

    pack_name = "MyPack"
    rp_path, dp_path = expected_pack_paths(str(mc_dir), pack_name)
    assert not os.path.exists(rp_path)
    assert not os.path.exists(dp_path)

    rows = []
    analyzer = ModpackAnalyzer(job_state)
    en, tr = analyzer.analyze(
        str(mc_dir),
        target_lang=_lang(),
        translate_mods=False,
        translate_books=True,
        translate_quests=False,
        pack_name=pack_name,
        on_row=lambda *a: rows.append(a),
        on_log=lambda *a: None,
        on_status=lambda *a: None,
    )

    assert en == 2
    assert tr == 0


def test_analyze_books_md_resourcepack_fallback(tmp_path, job_state):
    """GuideME-style MD book (assets/<modid>/guide/page1.md, no "/en_us/"
    subtree -- pages live at the guide root), no translated variant inside
    the jar -- only in the current resourcepack zip under the "_ru_ru/"
    convention. Must be found via the real writer's (jar_books.py's
    _process_book_md) tr_path formula: guide_target_path(fl, lang_file)."""
    mc_dir = tmp_path
    mods_dir = mc_dir / "mods"
    mods_dir.mkdir(parents=True)
    jar_path = mods_dir / "testmod.jar"
    en_internal = "assets/testmod/guide/page1.md"
    en_text = "# Title\n\nSome english prose text right here.\n"
    with zipfile.ZipFile(jar_path, "w") as z:
        z.writestr(en_internal, en_text)

    pack_name = "MyPack"
    tr_internal = "assets/testmod/guide/_ru_ru/page1.md"
    tr_text = "# Заголовок\n\nКакой-то русский текст прямо тут.\n"
    rp_path, _ = expected_pack_paths(str(mc_dir), pack_name)
    _write_pack_zip(rp_path, {tr_internal: tr_text.encode("utf-8")})

    rows = []
    analyzer = ModpackAnalyzer(job_state)
    en, tr = analyzer.analyze(
        str(mc_dir),
        target_lang=_lang(),
        translate_mods=False,
        translate_books=True,
        translate_quests=False,
        pack_name=pack_name,
        on_row=lambda *a: rows.append(a),
        on_log=lambda *a: None,
        on_status=lambda *a: None,
    )

    assert en == 2  # heading block + prose paragraph block
    assert tr == 2
    assert len(rows) == 1
    assert rows[0][2] == "Книга(MD)"


def test_analyze_books_xml_resourcepack_fallback(tmp_path, job_state):
    """GuideME-style XML book (assets/<modid>/guide/page1.xml, no "/en_us/"
    subtree), no translated variant inside the jar -- only in the current
    resourcepack zip under the "_ru_ru/" convention. Must be found via the
    same tr_path formula as the MD case (guide_target_path).

    Also covers a second, pre-existing bug found during this fix's own
    verification pass (unrelated to the pack-index fallback itself): the
    XML branch of _analyze_books used to compute its own local x_en/x_tr
    and never fold them into _analyze_books' b_en/b_tr running totals the
    way the JSON/MD branches do, so analyze()'s own (total_en, total_tr)
    aggregate silently dropped ALL XML book content from the readiness
    percentage in every output_mode, even though the per-file Analyze row
    was always correct. Fixed by accumulating into a dedicated xml_en/
    xml_tr pair and including it in the final return -- this test now
    asserts the CORRECT aggregate (en==2, tr==2), not the old buggy 0/0."""
    mc_dir = tmp_path
    mods_dir = mc_dir / "mods"
    mods_dir.mkdir(parents=True)
    jar_path = mods_dir / "testmod.jar"
    en_internal = "assets/testmod/guide/page1.xml"
    en_xml = (
        "<book><chapter title=\"Intro\">"
        "<page>Some prose text here for testing.</page>"
        "</chapter></book>"
    )
    with zipfile.ZipFile(jar_path, "w") as z:
        z.writestr(en_internal, en_xml)

    pack_name = "MyPack"
    tr_internal = "assets/testmod/guide/_ru_ru/page1.xml"
    tr_xml = (
        "<book><chapter title=\"Начало\">"
        "<page>Какой-то текст на странице для теста.</page>"
        "</chapter></book>"
    )
    rp_path, _ = expected_pack_paths(str(mc_dir), pack_name)
    _write_pack_zip(rp_path, {tr_internal: tr_xml.encode("utf-8")})

    rows = []
    analyzer = ModpackAnalyzer(job_state)
    en, tr = analyzer.analyze(
        str(mc_dir),
        target_lang=_lang(),
        translate_mods=False,
        translate_books=True,
        translate_quests=False,
        pack_name=pack_name,
        on_row=lambda *a: rows.append(a),
        on_log=lambda *a: None,
        on_status=lambda *a: None,
    )

    # The per-file row IS correct -- the pack fallback located the
    # translation and counted it right (title attribute + page text, both
    # translated).
    assert len(rows) == 1
    icon, mod_name, kind, row_tr, row_en, pct = rows[0]
    assert kind == "Книга(XML)"
    assert row_en == 2
    assert row_tr == 2
    assert pct == 100

    # analyze()'s own aggregate return value now includes it too (see the
    # docstring above -- this used to be the buggy 0/0).
    assert en == 2
    assert tr == 2
