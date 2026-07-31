"""Tests for mc_translator.processors.jar.JarProcessor -- covers two of the
most severe regressions found and fixed in this project: (1) zipfile.write
str(ZipInfo, ...) mutating the shared ZipInfo object in place, corrupting
zin's bookkeeping so a second read of the same entry raised BadZipFile
(broke "inplace" output for essentially any mod with a translatable
en_us.json), and (2) the writes_own_path/own_path dedup logic that stops a
book entry matched via a non-"/en_us/" marker (e.g. research JSON) from
being written twice (once by the raw copy-through, once by the book
processor) into the same zip."""
import json
import zipfile

try:
    from lxml import etree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from mc_translator.processors.jar import JarProcessor, _parse_legacy_lang, _serialize_legacy_lang


def test_parse_legacy_lang_ignores_comments_and_blank_lines():
    data = b"# a comment\n\nkey.one=Value One\nkey.two=Value Two\n"
    assert _parse_legacy_lang(data) == {"key.one": "Value One", "key.two": "Value Two"}


def test_serialize_legacy_lang_round_trips_through_parse():
    data = {"key.one": "Value One", "key.two": "Value Two"}
    assert _parse_legacy_lang(_serialize_legacy_lang(data)) == data


def build_test_jar(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "assets/testmod/lang/en_us.json",
            json.dumps({"item.testmod.thing": "A Cool Thing"}),
        )
        # Research JSON: matched via RESEARCH_PATH_MARKERS, no "/en_us/" in
        # its path -- own_path == item.filename, the case that used to
        # produce a duplicate zip entry (raw copy-through + book write).
        z.writestr(
            "assets/testmod/quests/research/foo.json",
            json.dumps({"name": "Foo Research", "pages": [{"text": "Some research text"}]}),
        )
        z.writestr("fabric.mod.json", json.dumps({"name": "TestMod"}))


def build_guide_jar(path):
    """A jar shaped like AE2's GuideME guide: pages live at the guide root
    (no per-language subtree in the source), and the mod also ships someone
    else's finished translation under a "_es_es/" subfolder -- the exact
    layout that used to get clobbered (see guide_md.py's module docstring)."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "assets/testmod/guide/channels.md",
            "# Channels\n\nSome introductory text about channels here.\n",
        )
        z.writestr(
            "assets/testmod/guide/_es_es/channels.md",
            "# Canales\n\nAlgun texto introductorio sobre los canales aqui.\n",
        )


def build_book_xml_jar(path):
    """The XML-book counterpart of build_guide_jar: a Patchouli/GuideME-style
    XML book living at the guide root (no "/en_us/" subtree in the source),
    plus another mod-shipped locale under "_es_es/" -- matches the same
    BOOK_PATH_MARKERS gate as the English source and, before the is_book_xml
    gate/writes_own_path fixes, used to get clobbered by the translation.

    No "encoding=" in the XML declaration: _process_book_xml decodes the zip
    entry to a `str` before handing it to lxml, and lxml refuses to parse a
    `str` whose declaration names an encoding (it only accepts that combo for
    `bytes` input) -- unrelated to the bug fixed here, so the fixture just
    avoids tripping it.
    """
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "assets/testmod/guide/book.xml",
            '<?xml version="1.0"?>\n'
            "<book>\n"
            '  <page title="Introduction">Some introductory text about channels here.</page>\n'
            "</book>\n",
        )
        z.writestr(
            "assets/testmod/guide/_es_es/book.xml",
            '<?xml version="1.0"?>\n'
            "<book>\n"
            '  <page title="Introduccion">Algun texto introductorio sobre los canales aqui.</page>\n'
            "</book>\n",
        )


def _rewrite_zip_entry(jar_path, entry_name, new_bytes):
    """Replace a single entry's content in an existing zip archive, keeping
    every other entry byte-for-byte -- used to simulate a modpack update
    changing the English source between two processor runs."""
    with zipfile.ZipFile(jar_path) as zin:
        entries = [(i.filename, zin.read(i)) for i in zin.infolist()]
    with zipfile.ZipFile(jar_path, "w") as zout:
        for name, data in entries:
            zout.writestr(name, new_bytes if name == entry_name else data)


def test_inplace_run_produces_no_duplicate_zip_entries(tmp_path, fake_service, job_state, fake_callbacks):
    jar_path = str(tmp_path / "testmod.jar")
    build_test_jar(jar_path)

    proc = JarProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        jar_path,
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="append",
        output_mode="inplace",
        translate_mods=True,
        translate_books=True,
        pack_writer=None,
    )

    with zipfile.ZipFile(jar_path) as z:
        names = z.namelist()
        for name in set(names):
            assert names.count(name) == 1, f"duplicate zip entry: {name}"


def test_inplace_run_translates_lang_and_research_json(tmp_path, fake_service, job_state, fake_callbacks):
    jar_path = str(tmp_path / "testmod.jar")
    build_test_jar(jar_path)

    proc = JarProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        jar_path,
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="append",
        output_mode="inplace",
        translate_mods=True,
        translate_books=True,
        pack_writer=None,
    )

    with zipfile.ZipFile(jar_path) as z:
        names = z.namelist()
        # Original English source must survive alongside the translation.
        assert "assets/testmod/lang/en_us.json" in names
        lang_out = json.loads(z.read("assets/testmod/lang/ru_ru.json"))
        assert "TR[A Cool Thing]" in lang_out["item.testmod.thing"]

        # Research JSON has no distinct target-locale path -- translated in
        # place at its own path (own_path case), not duplicated.
        research_out = json.loads(z.read("assets/testmod/quests/research/foo.json"))
        assert "TR[Foo Research]" in research_out["name"]
        assert "TR[Some research text]" in research_out["pages"][0]["text"]
        # A one-time backup of the true English source must be created
        # alongside it, since the live entry above was just overwritten.
        assert "assets/testmod/quests/research/foo.json.bak" in names
        backup = json.loads(z.read("assets/testmod/quests/research/foo.json.bak"))
        assert backup == {"name": "Foo Research", "pages": [{"text": "Some research text"}]}


def test_research_json_force_recovers_from_backup(tmp_path, fake_service, job_state, fake_callbacks):
    """Regression test: research JSON (RESEARCH_PATH_MARKERS, no "/en_us/"
    segment) is overwritten in place at its own path on every inplace run.
    Without a preserved English backup, a later "force" run would read
    already-Russian text from the live entry, fail looks_like_source_
    language, and silently send nothing for translation -- "force" mode's
    "redo everything" contract would be completely defeated. With the
    backup, force must read the true English from the ".bak" entry and
    genuinely re-translate."""
    jar_path = str(tmp_path / "testmod.jar")
    build_test_jar(jar_path)
    target_lang = {"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"}

    proc = JarProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        jar_path, target_lang=target_lang, mode="append", output_mode="inplace",
        translate_mods=False, translate_books=True, pack_writer=None,
    )

    fake_service.calls.clear()
    proc.process(
        jar_path, target_lang=target_lang, mode="force", output_mode="inplace",
        translate_mods=False, translate_books=True, pack_writer=None,
    )

    force_run_strings: set = set()
    for strings, label in fake_service.calls:
        if label and label[1] == "Книга(JSON)":
            force_run_strings.update(strings.values())
    assert "Foo Research" in force_run_strings
    assert "Some research text" in force_run_strings

    with zipfile.ZipFile(jar_path) as z:
        research_out = json.loads(z.read("assets/testmod/quests/research/foo.json"))
        assert "TR[Foo Research]" in research_out["name"]


def test_research_json_backup_is_one_time(tmp_path, fake_service, job_state, fake_callbacks):
    jar_path = str(tmp_path / "testmod.jar")
    build_test_jar(jar_path)
    target_lang = {"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"}

    proc = JarProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        jar_path, target_lang=target_lang, mode="append", output_mode="inplace",
        translate_mods=False, translate_books=True, pack_writer=None,
    )
    proc.process(
        jar_path, target_lang=target_lang, mode="append", output_mode="inplace",
        translate_mods=False, translate_books=True, pack_writer=None,
    )

    with zipfile.ZipFile(jar_path) as z:
        names = z.namelist()
        assert names.count("assets/testmod/quests/research/foo.json.bak") == 1
        backup = json.loads(z.read("assets/testmod/quests/research/foo.json.bak"))
        # Still the ORIGINAL English -- never overwritten with translated content.
        assert backup == {"name": "Foo Research", "pages": [{"text": "Some research text"}]}


def test_research_json_append_preserves_existing_translation_with_backup(
    tmp_path, fake_service, job_state, fake_callbacks
):
    """The backup mechanism must not regress append mode's existing-
    translation preservation: a second append run over an unchanged English
    source must not re-send already-translated content."""
    jar_path = str(tmp_path / "testmod.jar")
    build_test_jar(jar_path)
    target_lang = {"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"}

    proc = JarProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        jar_path, target_lang=target_lang, mode="append", output_mode="inplace",
        translate_mods=False, translate_books=True, pack_writer=None,
    )

    fake_service.calls.clear()
    proc.process(
        jar_path, target_lang=target_lang, mode="append", output_mode="inplace",
        translate_mods=False, translate_books=True, pack_writer=None,
    )

    second_run_strings: set = set()
    for strings, label in fake_service.calls:
        if label and label[1] == "Книга(JSON)":
            second_run_strings.update(strings.values())
    assert "Foo Research" not in second_run_strings
    assert "Some research text" not in second_run_strings

    with zipfile.ZipFile(jar_path) as z:
        research_out = json.loads(z.read("assets/testmod/quests/research/foo.json"))
        assert "TR[Foo Research]" in research_out["name"]


def test_research_json_resourcepack_mode_creates_no_backup(tmp_path, fake_service, job_state, fake_callbacks):
    """In resourcepack mode the source jar is never modified, so
    zin.read(item) is always genuinely English already -- no backup is
    needed, and none should be created."""
    jar_path = str(tmp_path / "testmod.jar")
    build_test_jar(jar_path)
    target_lang = {"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"}

    written: dict[str, bytes] = {}

    class FakePackWriter:
        def write(self, internal_path, data):
            written[internal_path] = data

    proc = JarProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        jar_path, target_lang=target_lang, mode="append", output_mode="resourcepack",
        translate_mods=False, translate_books=True, pack_writer=FakePackWriter(),
    )

    assert "assets/testmod/quests/research/foo.json" in written
    assert not any(p.endswith(".bak") for p in written)
    with zipfile.ZipFile(jar_path) as z:
        assert "assets/testmod/quests/research/foo.json.bak" not in z.namelist()
        # Source jar itself is untouched.
        assert json.loads(z.read("assets/testmod/quests/research/foo.json")) == {
            "name": "Foo Research", "pages": [{"text": "Some research text"}]
        }


def test_second_read_of_same_entry_does_not_raise_badzipfile(tmp_path, fake_service, job_state, fake_callbacks):
    """Regression test for the ZipInfo-mutation bug: zout.writestr(ZipInfo,
    ...) rewrites header_offset/CRC on that exact object in place. Since
    `item` is the same object zin's own bookkeeping uses, writing it
    unmodified to zout used to corrupt zin's copy -- any LATER read of that
    same entry (e.g. the lang processor's own zin.read(item) call, which
    runs right after the unconditional copy-through for non-target-locale
    items) raised zipfile.BadZipFile. A jar with a translatable en_us.json
    always hits this path in "inplace" mode, so this covers the single most
    impactful bug found in this project."""
    jar_path = str(tmp_path / "testmod.jar")
    build_test_jar(jar_path)

    proc = JarProcessor(fake_service, job_state, fake_callbacks)
    # Must not raise.
    proc.process(
        jar_path,
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="append",
        output_mode="inplace",
        translate_mods=True,
        translate_books=True,
        pack_writer=None,
    )
    error_logs = [msg for msg, tag in fake_callbacks.logs if tag == "red"]
    assert not error_logs, f"unexpected errors logged: {error_logs}"


def test_resourcepack_mode_uses_full_internal_paths(tmp_path, fake_service, job_state, fake_callbacks):
    from mc_translator.output.pack_writer import PackWriter

    jar_path = str(tmp_path / "testmod.jar")
    build_test_jar(jar_path)
    mc_dir = tmp_path / "mc"
    mc_dir.mkdir()

    pw = PackWriter(str(mc_dir), "TestPack", "1.20.1", "Russian")
    proc = JarProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        jar_path,
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="append",
        output_mode="resourcepack",
        translate_mods=True,
        translate_books=True,
        pack_writer=pw,
    )
    rp, dp = pw.close()

    with zipfile.ZipFile(rp) as z:
        names = z.namelist()
        assert "assets/testmod/lang/ru_ru.json" in names
        # Research JSON's translated output must land at its real path, not
        # collapsed to a bare basename at the zip root.
        assert "assets/testmod/quests/research/foo.json" in names
        assert "foo.json" not in names


def test_guide_md_resourcepack_writes_lang_subfolder_and_spares_other_locales(
    tmp_path, fake_service, job_state, fake_callbacks
):
    """Regression test for the AE2/GuideME guide corruption: guide pages live
    at the guide root in the source jar (no "/en_us/" subtree), and the mod
    ships a finished Spanish translation under "_es_es/". The old code wrote
    the Russian translation straight into the (English) root path and, on a
    prior run, would have treated "_es_es/channels.md" as more English source
    to translate -- clobbering the Spanish text with Russian."""
    from mc_translator.output.pack_writer import PackWriter

    jar_path = str(tmp_path / "testmod.jar")
    build_guide_jar(jar_path)
    mc_dir = tmp_path / "mc"
    mc_dir.mkdir()

    pw = PackWriter(str(mc_dir), "TestPack", "1.20.1", "Russian")
    proc = JarProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        jar_path,
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="append",
        output_mode="resourcepack",
        translate_mods=True,
        translate_books=True,
        pack_writer=pw,
    )
    rp, dp = pw.close()

    with zipfile.ZipFile(rp) as z:
        names = z.namelist()
        assert "assets/testmod/guide/_ru_ru/channels.md" in names
        # The English root and the Spanish translation are never a write
        # target -- only the new "_ru_ru/" page belongs in the overlay pack.
        assert "assets/testmod/guide/channels.md" not in names
        assert "assets/testmod/guide/_es_es/channels.md" not in names

        text = z.read("assets/testmod/guide/_ru_ru/channels.md").decode("utf-8")
        lines = text.split("\n")
        # Heading must stay on its own line, not glued into the paragraph.
        assert lines[0] == "# TR[Channels]"
        assert lines[1] == ""
        assert lines[2] == "TR[Some introductory text about channels here.]"


def test_guide_md_inplace_preserves_root_and_other_locale(
    tmp_path, fake_service, job_state, fake_callbacks
):
    jar_path = str(tmp_path / "testmod.jar")
    build_guide_jar(jar_path)
    original_es = zipfile.ZipFile(jar_path).read("assets/testmod/guide/_es_es/channels.md")
    original_en = zipfile.ZipFile(jar_path).read("assets/testmod/guide/channels.md")

    proc = JarProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        jar_path,
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="append",
        output_mode="inplace",
        translate_mods=True,
        translate_books=True,
        pack_writer=None,
    )

    with zipfile.ZipFile(jar_path) as z:
        names = z.namelist()
        for name in set(names):
            assert names.count(name) == 1, f"duplicate zip entry: {name}"

        assert "assets/testmod/guide/_ru_ru/channels.md" in names
        # The English root and the Spanish translation must survive
        # byte-for-byte -- neither is a valid write target for this run.
        assert z.read("assets/testmod/guide/channels.md") == original_en
        assert z.read("assets/testmod/guide/_es_es/channels.md") == original_es

        ru_text = z.read("assets/testmod/guide/_ru_ru/channels.md").decode("utf-8")
        assert ru_text.split("\n")[0] == "# TR[Channels]"


def test_book_xml_resourcepack_writes_lang_subfolder_and_spares_other_locales(
    tmp_path, fake_service, job_state, fake_callbacks
):
    """Regression test for the XML-book counterpart of the guide_md bug: a
    Patchouli/GuideME-style XML book at the guide root, with another
    mod-shipped locale under "_es_es/". Before the is_book_xml gate/
    writes_own_path fixes, the Spanish file matched the same
    BOOK_PATH_MARKERS gate as the English source and was treated as
    translatable English -- clobbering the Spanish text with Russian."""
    from mc_translator.output.pack_writer import PackWriter

    jar_path = str(tmp_path / "testmod.jar")
    build_book_xml_jar(jar_path)
    mc_dir = tmp_path / "mc"
    mc_dir.mkdir()

    pw = PackWriter(str(mc_dir), "TestPack", "1.20.1", "Russian")
    proc = JarProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        jar_path,
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="append",
        output_mode="resourcepack",
        translate_mods=False,
        translate_books=True,
        pack_writer=pw,
    )
    rp, dp = pw.close()

    with zipfile.ZipFile(rp) as z:
        names = z.namelist()
        assert "assets/testmod/guide/_ru_ru/book.xml" in names
        # The English root and the Spanish translation are never a write
        # target -- only the new "_ru_ru/" page belongs in the overlay pack.
        assert "assets/testmod/guide/book.xml" not in names
        assert "assets/testmod/guide/_es_es/book.xml" not in names

        root = ET.fromstring(z.read("assets/testmod/guide/_ru_ru/book.xml"))
        page = root[0]
        assert page.attrib["title"] == "TR[Introduction]"
        assert page.text.strip() == "TR[Some introductory text about channels here.]"


def test_book_xml_inplace_preserves_root_and_other_locale(
    tmp_path, fake_service, job_state, fake_callbacks
):
    jar_path = str(tmp_path / "testmod.jar")
    build_book_xml_jar(jar_path)
    original_es = zipfile.ZipFile(jar_path).read("assets/testmod/guide/_es_es/book.xml")
    original_en = zipfile.ZipFile(jar_path).read("assets/testmod/guide/book.xml")

    proc = JarProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        jar_path,
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="append",
        output_mode="inplace",
        translate_mods=False,
        translate_books=True,
        pack_writer=None,
    )

    with zipfile.ZipFile(jar_path) as z:
        names = z.namelist()
        for name in set(names):
            assert names.count(name) == 1, f"duplicate zip entry: {name}"

        assert "assets/testmod/guide/_ru_ru/book.xml" in names
        # The English root and the Spanish translation must survive
        # byte-for-byte -- neither is a valid write target for this run.
        assert z.read("assets/testmod/guide/book.xml") == original_en
        assert z.read("assets/testmod/guide/_es_es/book.xml") == original_es

        root = ET.fromstring(z.read("assets/testmod/guide/_ru_ru/book.xml"))
        page = root[0]
        assert page.attrib["title"] == "TR[Introduction]"
        assert page.text.strip() == "TR[Some introductory text about channels here.]"


def test_book_xml_append_mode_preserves_translation_when_source_gains_new_page(
    tmp_path, fake_service, job_state, fake_callbacks
):
    """Regression test for the existing_map seed-back in _process_book_xml
    (mirrors the nbt/yaml_toml/xml.py fix): a second append-mode run, after
    the mod's English source gains a new page, must keep the first page's
    existing translation untouched and only translate the new page -- not
    revert the first page back to English."""
    jar_path = str(tmp_path / "testmod.jar")
    build_book_xml_jar(jar_path)

    target_lang = {"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"}
    proc = JarProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        jar_path, target_lang=target_lang, mode="append", output_mode="inplace",
        translate_mods=False, translate_books=True, pack_writer=None,
    )

    # Simulate a modpack update: the English source gains a second page.
    _rewrite_zip_entry(
        jar_path,
        "assets/testmod/guide/book.xml",
        (
            b'<?xml version="1.0"?>\n'
            b"<book>\n"
            b'  <page title="Introduction">Some introductory text about channels here.</page>\n'
            b'  <page title="Advanced">Some advanced text about channels here.</page>\n'
            b"</book>\n"
        ),
    )

    calls_before_second_run = len(fake_service.calls)
    proc.process(
        jar_path, target_lang=target_lang, mode="append", output_mode="inplace",
        translate_mods=False, translate_books=True, pack_writer=None,
    )

    with zipfile.ZipFile(jar_path) as z:
        names = z.namelist()
        for name in set(names):
            assert names.count(name) == 1, f"duplicate zip entry: {name}"
        ru_xml = z.read("assets/testmod/guide/_ru_ru/book.xml")

    root = ET.fromstring(ru_xml)
    pages = list(root)
    # Already-translated first page must survive unchanged, not reverted.
    assert pages[0].attrib["title"] == "TR[Introduction]"
    assert pages[0].text.strip() == "TR[Some introductory text about channels here.]"
    # New page must have been freshly translated.
    assert pages[1].attrib["title"] == "TR[Advanced]"
    assert pages[1].text.strip() == "TR[Some advanced text about channels here.]"

    # The already-translated first page's source strings must not have been
    # sent through translate_dict a second time -- only the new page's.
    second_run_strings: set = set()
    for strings, label in fake_service.calls[calls_before_second_run:]:
        if label and label[1] == "Книга(XML)":
            second_run_strings.update(strings.values())
    assert "Introduction" not in second_run_strings
    assert "Some introductory text about channels here." not in second_run_strings


def test_book_xml_force_mode_retranslates_already_translated_page(
    tmp_path, fake_service, job_state, fake_callbacks
):
    """Regression test: collect_translatable used to run AFTER seeding root
    with the existing translation (existing_map), so looks_like_source_
    language saw the already-Cyrillic text and excluded it regardless of
    mode -- silently defeating "force" mode's "retranslate everything"
    contract on any page that already had a prior translation."""
    jar_path = str(tmp_path / "testmod.jar")
    build_book_xml_jar(jar_path)

    target_lang = {"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"}
    proc = JarProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        jar_path, target_lang=target_lang, mode="append", output_mode="inplace",
        translate_mods=False, translate_books=True, pack_writer=None,
    )

    fake_service.calls.clear()
    proc.process(
        jar_path, target_lang=target_lang, mode="force", output_mode="inplace",
        translate_mods=False, translate_books=True, pack_writer=None,
    )

    with zipfile.ZipFile(jar_path) as z:
        ru_xml = z.read("assets/testmod/guide/_ru_ru/book.xml")
    root = ET.fromstring(ru_xml)
    page = root[0]
    assert page.attrib["title"] == "TR[Introduction]"
    assert page.text.strip() == "TR[Some introductory text about channels here.]"

    # Force mode must have actually re-sent the already-translated page's
    # strings, not silently skipped them because they already looked done.
    force_run_strings: set = set()
    for strings, label in fake_service.calls:
        if label and label[1] == "Книга(XML)":
            force_run_strings.update(strings.values())
    assert "Introduction" in force_run_strings
    assert "Some introductory text about channels here." in force_run_strings


def build_legacy_lang_jar(path):
    """A jar shaped like a pre-1.13 mod: assets/<modid>/lang/en_US.lang in
    the plain key=value format (uppercase-region convention), no JSON lang
    file at all."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "assets/testmod/lang/en_US.lang",
            "item.testmod.thing=A Cool Thing\nitem.testmod.other=Another Thing\n",
        )
        z.writestr("fabric.mod.json", json.dumps({"name": "TestMod"}))


def test_legacy_lang_resourcepack_translates_and_converts_case(tmp_path, fake_service, job_state, fake_callbacks):
    jar_path = str(tmp_path / "testmod.jar")
    build_legacy_lang_jar(jar_path)

    from mc_translator.output.pack_writer import PackWriter

    pack_writer = PackWriter(str(tmp_path), "TestPack", "1.12.2", "Russian")
    proc = JarProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        jar_path,
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="force",
        output_mode="resourcepack",
        translate_mods=True,
        translate_books=True,
        pack_writer=pack_writer,
    )
    pack_writer.close()

    with zipfile.ZipFile(pack_writer.rp_zip_path) as z:
        names = z.namelist()
        # Legacy-region-case convention: ru_RU, not ru_ru.
        assert "assets/testmod/lang/ru_RU.lang" in names
        content = z.read("assets/testmod/lang/ru_RU.lang").decode("utf-8")
    assert "item.testmod.thing=TR[A Cool Thing]" in content
    assert "item.testmod.other=TR[Another Thing]" in content


def test_legacy_lang_inplace_translates_and_preserves_source(tmp_path, fake_service, job_state, fake_callbacks):
    jar_path = str(tmp_path / "testmod.jar")
    build_legacy_lang_jar(jar_path)

    proc = JarProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        jar_path,
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="force",
        output_mode="inplace",
        translate_mods=True,
        translate_books=True,
        pack_writer=None,
    )

    with zipfile.ZipFile(jar_path) as z:
        names = z.namelist()
        for name in set(names):
            assert names.count(name) == 1, f"duplicate zip entry: {name}"
        assert "assets/testmod/lang/en_US.lang" in names
        assert "assets/testmod/lang/ru_RU.lang" in names
        content = z.read("assets/testmod/lang/ru_RU.lang").decode("utf-8")
    assert "item.testmod.thing=TR[A Cool Thing]" in content


def test_legacy_lang_append_mode_preserves_existing_translation(tmp_path, fake_service, job_state, fake_callbacks):
    """Regression test: a second run with a new key added to the English
    source must translate only the new key and must NOT revert the
    already-translated one back to English."""
    jar_path = str(tmp_path / "testmod.jar")
    with zipfile.ZipFile(jar_path, "w") as z:
        z.writestr(
            "assets/testmod/lang/en_US.lang",
            "item.testmod.thing=A Cool Thing\nitem.testmod.other=Another Thing\n",
        )
        z.writestr("assets/testmod/lang/ru_RU.lang", "item.testmod.thing=Классная штука\n")

    proc = JarProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        jar_path,
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="append",
        output_mode="inplace",
        translate_mods=True,
        translate_books=True,
        pack_writer=None,
    )

    with zipfile.ZipFile(jar_path) as z:
        names = z.namelist()
        for name in set(names):
            assert names.count(name) == 1, f"duplicate zip entry: {name}"
        content = z.read("assets/testmod/lang/ru_RU.lang").decode("utf-8")
    assert "item.testmod.thing=Классная штука" in content, "already-translated key must survive untouched"
    assert "item.testmod.other=TR[Another Thing]" in content

    assert len(fake_service.calls) == 1
    assert list(fake_service.calls[0][0].values()) == ["Another Thing"]
