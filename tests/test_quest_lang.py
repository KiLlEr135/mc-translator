"""Tests for mc_translator.processors.quest_lang -- the FTB Quests lang-key
export translator (quests/lang/<code>.snbt), used when a pack stores ALL
quest/chapter/task text as lang keys instead of literal text in
quests/chapters/*.snbt (confirmed on a real pack, ATM10 MC 1.21.1, where
SnbtProcessor's target files had ZERO literal text and quest translation was
a silent no-op). Diffs quests/lang/en_us.snbt (reference) against the
target-language file, translating only missing/still-source-language keys --
never touching an already-translated key or any OTHER locale's file."""
from mc_translator.processors.quest_lang import (
    QuestLangProcessor,
    append_new_lang_entries,
    apply_lang_updates,
    bucket_for_key,
    build_chapter_id_map,
    parse_lang_entries,
)
from mc_translator.processors.quest_lang import (
    _is_non_translatable_markup as is_non_translatable_markup,
)

# ---------------------------------------------------------------------
# parse_lang_entries
# ---------------------------------------------------------------------


def test_parse_lang_entries_scalar_values():
    content = '{\n\tchapter.ABC.title: "Theurgy"\n\tquest.DEF.title: "Strong Shield"\n}\n'
    assert parse_lang_entries(content) == {
        "chapter.ABC.title": "Theurgy",
        "quest.DEF.title": "Strong Shield",
    }


def test_parse_lang_entries_single_line_array():
    content = '{\n\tquest.ABC.quest_desc: ["Hold a Shield in your off hand."]\n}\n'
    assert parse_lang_entries(content) == {"quest.ABC.quest_desc": ["Hold a Shield in your off hand."]}


def test_parse_lang_entries_multi_line_array_no_commas():
    """Real FTB Quests export format: array elements on their own lines,
    with no comma separators (same juxtaposition style as quests/chapters/
    *.snbt's `images: [ {...} {...} ]`)."""
    content = (
        "{\n"
        "\tquest.ABC.quest_desc: [\n"
        '\t\t"Allows you to see details.\\n"\n'
        '\t\t"{image:atm:textures/foo.png}"\n'
        "\t]\n"
        "}\n"
    )
    assert parse_lang_entries(content) == {
        "quest.ABC.quest_desc": ["Allows you to see details.\\n", "{image:atm:textures/foo.png}"]
    }


def test_parse_lang_entries_unescapes_quotes_and_backslashes():
    content = '{\n\tquest.ABC.title: "She said \\"hi\\" \\\\ bye"\n}\n'
    assert parse_lang_entries(content) == {"quest.ABC.title": 'She said "hi" \\ bye'}


def test_parse_lang_entries_empty_object():
    assert parse_lang_entries("{\n}\n") == {}


# ---------------------------------------------------------------------
# apply_lang_updates / append_new_lang_entries
# ---------------------------------------------------------------------


def test_apply_lang_updates_replaces_only_matching_keys_in_place():
    content = '{\n\ta.title: "old a"\n\tb.title: "old b"\n}\n'
    result = apply_lang_updates(content, {"a.title": "new a"})
    assert 'a.title: "new a"' in result
    assert 'b.title: "old b"' in result  # untouched


def test_apply_lang_updates_replaces_array_value():
    content = '{\n\ta.quest_desc: ["old"]\n}\n'
    result = apply_lang_updates(content, {"a.quest_desc": ["new one", "new two"]})
    assert parse_lang_entries(result) == {"a.quest_desc": ["new one", "new two"]}


def test_append_new_lang_entries_inserts_before_closing_brace():
    content = '{\n\ta.title: "existing"\n}\n'
    result = append_new_lang_entries(content, {"b.title": "brand new"})
    assert parse_lang_entries(result) == {"a.title": "existing", "b.title": "brand new"}


def test_append_new_lang_entries_noop_when_nothing_to_add():
    content = '{\n\ta.title: "existing"\n}\n'
    assert append_new_lang_entries(content, {}) == content


# ---------------------------------------------------------------------
# QuestLangProcessor.process
# ---------------------------------------------------------------------


def _write(path, content):
    path.write_text(content, encoding="utf-8")
    return path


def test_no_reference_file_is_a_harmless_noop(tmp_path, fake_service, fake_callbacks, job_state, lang):
    target = tmp_path / "ru_ru.snbt"
    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(target), target_lang=lang, mode="append")
    assert not target.exists()
    assert fake_service.calls == []


def test_missing_target_file_is_created_from_scratch(tmp_path, fake_service, fake_callbacks, job_state, lang):
    _write(tmp_path / "en_us.snbt", '{\n\tchapter.ABC.title: "Theurgy"\n}\n')
    target = tmp_path / "ru_ru.snbt"

    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(target), target_lang=lang, mode="append")

    assert target.exists()
    assert parse_lang_entries(target.read_text(encoding="utf-8")) == {"chapter.ABC.title": "TR[Theurgy]"}


def test_missing_keys_are_translated_and_appended(tmp_path, fake_service, fake_callbacks, job_state, lang):
    _write(
        tmp_path / "en_us.snbt",
        '{\n\tchapter.ABC.title: "Theurgy"\n\tchapter.XYZ.title: "Create"\n}\n',
    )
    _write(tmp_path / "ru_ru.snbt", '{\n\tchapter.ABC.title: "Теургия"\n}\n')

    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(tmp_path / "ru_ru.snbt"), target_lang=lang, mode="append")

    entries = parse_lang_entries((tmp_path / "ru_ru.snbt").read_text(encoding="utf-8"))
    assert entries["chapter.ABC.title"] == "Теургия"  # already Russian -- untouched
    assert entries["chapter.XYZ.title"] == "TR[Create]"  # missing -- added
    assert len(fake_service.calls) == 1
    assert list(fake_service.calls[0][0].values()) == ["Create"]


def test_existing_source_language_value_is_replaced_in_place(tmp_path, fake_service, fake_callbacks, job_state, lang):
    _write(tmp_path / "en_us.snbt", '{\n\tquest.ABC.title: "Strong Shield"\n}\n')
    _write(tmp_path / "ru_ru.snbt", '{\n\tquest.ABC.title: "Strong Shield"\n}\n')  # never translated

    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(tmp_path / "ru_ru.snbt"), target_lang=lang, mode="append")

    entries = parse_lang_entries((tmp_path / "ru_ru.snbt").read_text(encoding="utf-8"))
    assert entries["quest.ABC.title"] == "TR[Strong Shield]"


def test_already_translated_value_is_never_sent_to_the_service(tmp_path, fake_service, fake_callbacks, job_state, lang):
    _write(tmp_path / "en_us.snbt", '{\n\tquest.ABC.title: "Strong Shield"\n}\n')
    _write(tmp_path / "ru_ru.snbt", '{\n\tquest.ABC.title: "Крепкий щит"\n}\n')

    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(tmp_path / "ru_ru.snbt"), target_lang=lang, mode="append")

    assert fake_service.calls == []
    assert (tmp_path / "ru_ru.snbt").read_text(encoding="utf-8") == '{\n\tquest.ABC.title: "Крепкий щит"\n}\n'


def test_array_field_translates_only_missing_elements(tmp_path, fake_service, fake_callbacks, job_state, lang):
    _write(
        tmp_path / "en_us.snbt",
        '{\n\tquest.ABC.quest_desc: ["First line." "Second line."]\n}\n',
    )
    # Target already has the first line translated; second is still English.
    _write(
        tmp_path / "ru_ru.snbt",
        '{\n\tquest.ABC.quest_desc: ["Первая строка." "Second line."]\n}\n',
    )

    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(tmp_path / "ru_ru.snbt"), target_lang=lang, mode="append")

    entries = parse_lang_entries((tmp_path / "ru_ru.snbt").read_text(encoding="utf-8"))
    assert entries["quest.ABC.quest_desc"] == ["Первая строка.", "TR[Second line.]"]
    assert list(fake_service.calls[0][0].values()) == ["Second line."]


def test_force_mode_retranslates_from_reference_even_if_already_translated(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    _write(tmp_path / "en_us.snbt", '{\n\tquest.ABC.title: "Strong Shield"\n}\n')
    _write(tmp_path / "ru_ru.snbt", '{\n\tquest.ABC.title: "Крепкий щит"\n}\n')

    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(tmp_path / "ru_ru.snbt"), target_lang=lang, mode="force")

    entries = parse_lang_entries((tmp_path / "ru_ru.snbt").read_text(encoding="utf-8"))
    assert entries["quest.ABC.title"] == "TR[Strong Shield]"
    assert list(fake_service.calls[0][0].values()) == ["Strong Shield"]


class _FailingService:
    """Simulates TranslationService.translate_dict's documented contract for
    a genuine per-key engine failure -- echoes the sent text back unchanged
    (service.py: `result[key] = item.original`), same as a real dead-backend
    or malformed-response fallback would produce."""

    def __init__(self):
        self.calls = []

    def translate_dict(self, strings, target_lang, callbacks, *, context="", usage_label=None):
        self.calls.append((dict(strings), usage_label))
        return dict(strings)


def test_force_mode_engine_failure_does_not_regress_an_already_translated_value(
    tmp_path, fake_callbacks, job_state, lang
):
    """Regression test: this exact scenario (mode="force" + a transient
    engine failure) silently overwrote the real ru_ru.snbt with a byte-for-
    byte copy of en_us.snbt on the user's actual ATM10 pack."""
    _write(tmp_path / "en_us.snbt", '{\n\tquest.ABC.title: "Strong Shield"\n}\n')
    _write(tmp_path / "ru_ru.snbt", '{\n\tquest.ABC.title: "Крепкий щит"\n}\n')

    proc = QuestLangProcessor(_FailingService(), job_state, fake_callbacks)
    proc.process(str(tmp_path / "ru_ru.snbt"), target_lang=lang, mode="force")

    entries = parse_lang_entries((tmp_path / "ru_ru.snbt").read_text(encoding="utf-8"))
    assert entries["quest.ABC.title"] == "Крепкий щит"


def test_force_mode_engine_failure_does_not_regress_an_already_translated_array_item(
    tmp_path, fake_callbacks, job_state, lang
):
    _write(
        tmp_path / "en_us.snbt",
        '{\n\tquest.ABC.quest_desc: ["First line." "Second line."]\n}\n',
    )
    _write(
        tmp_path / "ru_ru.snbt",
        '{\n\tquest.ABC.quest_desc: ["Первая строка." "Вторая строка."]\n}\n',
    )

    proc = QuestLangProcessor(_FailingService(), job_state, fake_callbacks)
    proc.process(str(tmp_path / "ru_ru.snbt"), target_lang=lang, mode="force")

    entries = parse_lang_entries((tmp_path / "ru_ru.snbt").read_text(encoding="utf-8"))
    assert entries["quest.ABC.quest_desc"] == ["Первая строка.", "Вторая строка."]


def test_force_mode_engine_failure_still_fills_a_missing_key_with_english_fallback(
    tmp_path, fake_callbacks, job_state, lang
):
    """No pre-existing value to protect here -- the documented English-
    fallback behavior for a missing key is unchanged."""
    _write(tmp_path / "en_us.snbt", '{\n\tquest.NEW.title: "Brand New"\n}\n')
    target = tmp_path / "ru_ru.snbt"

    proc = QuestLangProcessor(_FailingService(), job_state, fake_callbacks)
    proc.process(str(target), target_lang=lang, mode="force")

    entries = parse_lang_entries(target.read_text(encoding="utf-8"))
    assert entries["quest.NEW.title"] == "Brand New"


def test_backup_is_created_once_and_preserves_original(tmp_path, fake_service, fake_callbacks, job_state, lang):
    _write(tmp_path / "en_us.snbt", '{\n\tquest.ABC.title: "Strong Shield"\n}\n')
    _write(tmp_path / "ru_ru.snbt", '{\n\tquest.ABC.title: "Strong Shield"\n}\n')
    target = tmp_path / "ru_ru.snbt"
    backup = tmp_path / "ru_ru.snbt.bak"

    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(target), target_lang=lang, mode="append")
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == '{\n\tquest.ABC.title: "Strong Shield"\n}\n'

    # A later run must not overwrite the backup with the now-translated content.
    _write(tmp_path / "en_us.snbt", '{\n\tquest.ABC.title: "Strong Shield"\n\tquest.DEF.title: "New"\n}\n')
    proc.process(str(target), target_lang=lang, mode="append")
    assert backup.read_text(encoding="utf-8") == '{\n\tquest.ABC.title: "Strong Shield"\n}\n'


def test_other_locale_files_in_the_same_directory_are_never_touched(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    """The exact corruption this design avoids: processing ru_ru.snbt must
    never read or write es_es.snbt/it_it.snbt/etc. sitting right next to
    it -- only the one file_path passed in, plus the en_us.snbt reference."""
    _write(tmp_path / "en_us.snbt", '{\n\tquest.ABC.title: "Strong Shield"\n}\n')
    _write(tmp_path / "ru_ru.snbt", '{\n\tquest.ABC.title: "Strong Shield"\n}\n')
    es_content = '{\n\tquest.ABC.title: "Escudo Fuerte"\n}\n'
    _write(tmp_path / "es_es.snbt", es_content)

    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(tmp_path / "ru_ru.snbt"), target_lang=lang, mode="append")

    assert (tmp_path / "es_es.snbt").read_text(encoding="utf-8") == es_content
    assert not (tmp_path / "es_es.snbt.bak").exists()


def test_skip_mode_bails_when_at_least_90_percent_already_translated(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    en_lines = "".join(f'\tquest.{i:03d}.title: "Item {i}"\n' for i in range(10))
    _write(tmp_path / "en_us.snbt", "{\n" + en_lines + "}\n")
    ru_lines = "".join(f'\tquest.{i:03d}.title: "Перевод {i}"\n' for i in range(9))
    ru_lines += '\tquest.009.title: "Item 9"\n'  # 1 of 10 still English == 90% done
    _write(tmp_path / "ru_ru.snbt", "{\n" + ru_lines + "}\n")

    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(tmp_path / "ru_ru.snbt"), target_lang=lang, mode="skip")

    assert fake_service.calls == []


def test_skip_mode_translates_remaining_below_90_percent_threshold(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    en_lines = "".join(f'\tquest.{i:03d}.title: "Item {i}"\n' for i in range(4))
    _write(tmp_path / "en_us.snbt", "{\n" + en_lines + "}\n")
    ru_lines = '\tquest.000.title: "Перевод 0"\n' + "".join(
        f'\tquest.{i:03d}.title: "Item {i}"\n' for i in range(1, 4)
    )
    _write(tmp_path / "ru_ru.snbt", "{\n" + ru_lines + "}\n")

    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(tmp_path / "ru_ru.snbt"), target_lang=lang, mode="skip")

    assert len(fake_service.calls) == 1
    entries = parse_lang_entries((tmp_path / "ru_ru.snbt").read_text(encoding="utf-8"))
    assert entries["quest.000.title"] == "Перевод 0"
    assert entries["quest.001.title"] == "TR[Item 1]"


# ---------------------------------------------------------------------
# Non-translatable markup filter ({image:...}, {@pagebreak})
# ---------------------------------------------------------------------
# Regression coverage: the shield system (mask_protected_fragments) does NOT
# protect this FTB-Quests-specific syntax, so sending it whole-value through
# an LLM as if it were prose risks the model silently altering the layout
# parameters or dropping the tag -- confirmed on a real pack's PRE-EXISTING
# (not this tool's own) lang file, where several {image:...} tags had their
# width/height altered. These entries carry no natural-language text, so
# they should never reach translate_dict at all.


def test_is_non_translatable_markup_matches_image_and_pagebreak():
    assert is_non_translatable_markup('{image:atm:textures/foo.png width:100 height:50 align:center}')
    assert is_non_translatable_markup("{@pagebreak}")


def test_is_non_translatable_markup_does_not_match_prose_containing_a_tag():
    """Only the entire value being exactly one of these forms counts --
    prose that merely mentions or embeds a tag still needs translating."""
    assert not is_non_translatable_markup('Look at this: {image:foo.png} nice, right?')
    assert not is_non_translatable_markup("Strong Shield")


def test_missing_scalar_markup_value_is_copied_without_calling_the_service(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    _write(tmp_path / "en_us.snbt", '{\n\tquest.ABC.icon_note: "{@pagebreak}"\n}\n')
    target = tmp_path / "ru_ru.snbt"

    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(target), target_lang=lang, mode="append")

    assert fake_service.calls == []
    entries = parse_lang_entries(target.read_text(encoding="utf-8"))
    assert entries["quest.ABC.icon_note"] == "{@pagebreak}"


def test_missing_array_of_only_markup_is_copied_without_calling_the_service(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    _write(
        tmp_path / "en_us.snbt",
        '{\n\tquest.ABC.quest_desc: ['
        '"{image:atm:textures/a.png width:100 height:100 align:center}" '
        '"{@pagebreak}"'
        "]\n}\n",
    )
    target = tmp_path / "ru_ru.snbt"

    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(target), target_lang=lang, mode="append")

    assert fake_service.calls == []
    entries = parse_lang_entries(target.read_text(encoding="utf-8"))
    assert entries["quest.ABC.quest_desc"] == [
        "{image:atm:textures/a.png width:100 height:100 align:center}",
        "{@pagebreak}",
    ]


def test_mixed_array_translates_text_but_copies_markup_verbatim(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    _write(
        tmp_path / "en_us.snbt",
        '{\n\tquest.ABC.quest_desc: ['
        '"Some real text." '
        '"{image:atm:textures/a.png width:100 height:100 align:center}"'
        "]\n}\n",
    )
    target = tmp_path / "ru_ru.snbt"

    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(target), target_lang=lang, mode="append")

    entries = parse_lang_entries(target.read_text(encoding="utf-8"))
    assert entries["quest.ABC.quest_desc"] == [
        "TR[Some real text.]",
        "{image:atm:textures/a.png width:100 height:100 align:center}",
    ]
    # Only the real text line was ever sent to the service -- the image tag
    # never appeared in any translate_dict call.
    assert len(fake_service.calls) == 1
    assert list(fake_service.calls[0][0].values()) == ["Some real text."]


def test_existing_untranslated_markup_item_is_left_untouched(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    """An already-present markup value that doesn't match the Cyrillic
    regex (true of every {image:...}/{@pagebreak} tag, translated or not)
    must never be sent for translation, whether it's correct or a
    pre-existing mismatch from some other source -- repairing pre-existing
    structural issues is out of scope for this processor."""
    _write(
        tmp_path / "en_us.snbt",
        '{\n\tquest.ABC.quest_desc: ['
        '"Real text." '
        '"{image:atm:textures/a.png width:100 height:100 align:center}"'
        "]\n}\n",
    )
    _write(
        tmp_path / "ru_ru.snbt",
        '{\n\tquest.ABC.quest_desc: ['
        '"Настоящий текст." '
        '"{image:atm:textures/a.png width:999 height:999 align:center}"'
        "]\n}\n",
    )

    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(tmp_path / "ru_ru.snbt"), target_lang=lang, mode="append")

    assert fake_service.calls == []
    entries = parse_lang_entries((tmp_path / "ru_ru.snbt").read_text(encoding="utf-8"))
    # Left exactly as it was -- not "fixed" to match the reference, not sent
    # to the AI, just untouched.
    assert entries["quest.ABC.quest_desc"][1] == "{image:atm:textures/a.png width:999 height:999 align:center}"


def test_force_mode_resets_markup_to_the_reference_value(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    """Force mode rebuilds from the reference for everything, so it's the
    one path that CAN heal a pre-existing mismatched markup tag -- by
    resetting it back to the clean source, with no AI call involved."""
    _write(
        tmp_path / "en_us.snbt",
        '{\n\tquest.ABC.quest_desc: ["{image:atm:textures/a.png width:100 height:100 align:center}"]\n}\n',
    )
    _write(
        tmp_path / "ru_ru.snbt",
        '{\n\tquest.ABC.quest_desc: ["{image:atm:textures/a.png width:999 height:999 align:center}"]\n}\n',
    )

    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(tmp_path / "ru_ru.snbt"), target_lang=lang, mode="force")

    entries = parse_lang_entries((tmp_path / "ru_ru.snbt").read_text(encoding="utf-8"))
    assert entries["quest.ABC.quest_desc"] == ["{image:atm:textures/a.png width:100 height:100 align:center}"]


# ---------------------------------------------------------------------
# build_chapter_id_map / bucket_for_key -- routing lang keys into the same
# per-file split layout FTB Quests Lang Splitter's own /langsplitter split
# would produce (chapter.snbt/chapter_group.snbt/file.snbt/reward_table.snbt
# as one aggregate file each, quest./task. keys per-chapter under chapters/).
# ---------------------------------------------------------------------


def test_build_chapter_id_map_maps_every_id_in_a_chapter_file_to_its_filename(tmp_path):
    chapters_dir = tmp_path / "chapters"
    chapters_dir.mkdir()
    _write(
        chapters_dir / "theurgy.snbt",
        '{\n\tid: "007B547630FF0478"\n\tquests: [\n\t\t{\n\t\t\tid: "02593CD4B4AE814C"\n'
        '\t\t\ttasks: [{\n\t\t\t\tid: "2BC2C407C46E8F62"\n\t\t\t}]\n\t\t}\n\t]\n}\n',
    )

    id_map = build_chapter_id_map(str(chapters_dir))

    assert id_map == {
        "007B547630FF0478": "theurgy",
        "02593CD4B4AE814C": "theurgy",
        "2BC2C407C46E8F62": "theurgy",
    }


def test_build_chapter_id_map_ignores_non_hex_id_fields(tmp_path):
    """Item/icon ids like `id: "minecraft:coal"` must never be mistaken for
    a quest/task/chapter id -- the regex requires the whole quoted value to
    be hex, so only the real chapter id here should end up in the map."""
    chapters_dir = tmp_path / "chapters"
    chapters_dir.mkdir()
    _write(
        chapters_dir / "aether.snbt",
        '{\n\ticon: { id: "minecraft:coal" }\n\tid: "ABCDEF0123456789"\n}\n',
    )

    assert build_chapter_id_map(str(chapters_dir)) == {"ABCDEF0123456789": "aether"}


def test_build_chapter_id_map_empty_when_chapters_dir_missing(tmp_path):
    assert build_chapter_id_map(str(tmp_path / "does_not_exist")) == {}


def test_bucket_for_key_routes_top_level_prefixes_regardless_of_id_map():
    assert bucket_for_key("chapter.ABC.title", {}) == ("top", "chapter")
    assert bucket_for_key("chapter_group.ABC.title", {}) == ("top", "chapter_group")
    assert bucket_for_key("file.0000000000000001.title", {}) == ("top", "file")
    assert bucket_for_key("reward_table.ABC.title", {}) == ("top", "reward_table")


def test_bucket_for_key_routes_quest_and_task_keys_to_their_owning_chapter():
    id_map = {"ABC": "theurgy", "DEF": "theurgy"}
    assert bucket_for_key("quest.ABC.title", id_map) == ("chapter", "theurgy")
    assert bucket_for_key("task.DEF.title", id_map) == ("chapter", "theurgy")


def test_bucket_for_key_none_for_a_quest_id_with_no_known_chapter():
    assert bucket_for_key("quest.UNKNOWN.title", {}) is None


def test_bucket_for_key_none_for_an_unrecognized_prefix():
    assert bucket_for_key("something_else.ABC.title", {}) is None


# ---------------------------------------------------------------------
# QuestLangProcessor.process -- FTB Quests Lang Splitter split-file writing
# (the actual fix for quest text not appearing in-game: editing the flat
# file alone was confirmed insufficient on a real pack -- see module
# docstring). Only kicks in once the mod is detected under mods/.
# ---------------------------------------------------------------------


def _full_mc_dir_layout(tmp_path, *, chapter_id, quest_id, chapter_name="theurgy"):
    """Real ATM10-shaped tree: mods/ftbquestslangsplitter present, plus a
    quests/chapters/<chapter_name>.snbt actually defining chapter_id/
    quest_id, so QuestLangProcessor can route chapter./quest. lang keys to
    their real split file the same way the mod itself would. Returns the
    lang dir (.../quests/lang)."""
    lang_dir = tmp_path / "config" / "ftbquests" / "quests" / "lang"
    lang_dir.mkdir(parents=True)
    chapters_dir = tmp_path / "config" / "ftbquests" / "quests" / "chapters"
    chapters_dir.mkdir(parents=True)
    _write(
        chapters_dir / f"{chapter_name}.snbt",
        f'{{\n\tid: "{chapter_id}"\n\tquests: [\n\t\t{{\n\t\t\tid: "{quest_id}"\n\t\t}}\n\t]\n}}\n',
    )
    mods_dir = tmp_path / "mods"
    mods_dir.mkdir()
    (mods_dir / "ftbquestslangsplitter-1.0.5.jar").write_bytes(b"")
    return lang_dir


def test_process_writes_split_files_matching_the_real_lang_splitter_layout(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    lang_dir = _full_mc_dir_layout(tmp_path, chapter_id="AAAA000000000001", quest_id="BBBB000000000002")
    _write(
        lang_dir / "en_us.snbt",
        '{\n\tchapter.AAAA000000000001.title: "Theurgy"\n\tquest.BBBB000000000002.title: "Strong Shield"\n}\n',
    )

    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(lang_dir / "ru_ru.snbt"), target_lang=lang, mode="append")

    split_dir = lang_dir / "ru_ru"
    chapter_top = parse_lang_entries((split_dir / "chapter.snbt").read_text(encoding="utf-8"))
    assert chapter_top == {"chapter.AAAA000000000001.title": "TR[Theurgy]"}

    chapter_quests = parse_lang_entries((split_dir / "chapters" / "theurgy.snbt").read_text(encoding="utf-8"))
    assert chapter_quests == {"quest.BBBB000000000002.title": "TR[Strong Shield]"}

    # The flat file is still written too -- unchanged old behavior, still
    # useful as a human-readable reference even though it's not what the
    # mod actually renders from.
    flat = parse_lang_entries((lang_dir / "ru_ru.snbt").read_text(encoding="utf-8"))
    assert flat == {
        "chapter.AAAA000000000001.title": "TR[Theurgy]",
        "quest.BBBB000000000002.title": "TR[Strong Shield]",
    }


def test_process_seeds_a_fresh_split_file_from_the_already_translated_flat_file(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    """If the flat ru_ru.snbt already has a real translation (e.g. from a
    past run, or the pack's own shipped partial translation), generating
    the split file for the first time must reuse it instead of re-sending
    it to the AI -- a real pack has ~8000 such lines, re-translating them
    all again would be a huge, pointless cost."""
    lang_dir = _full_mc_dir_layout(tmp_path, chapter_id="AAAA000000000001", quest_id="BBBB000000000002")
    _write(
        lang_dir / "en_us.snbt",
        '{\n\tchapter.AAAA000000000001.title: "Theurgy"\n\tquest.BBBB000000000002.title: "Strong Shield"\n}\n',
    )
    # The flat file is ALREADY translated (as if from a previous run) --
    # process() should still write it (unchanged, already-good values are
    # left alone) but must seed the brand-new split files from it.
    _write(
        lang_dir / "ru_ru.snbt",
        '{\n\tchapter.AAAA000000000001.title: "Теургия"\n\tquest.BBBB000000000002.title: "Крепкий щит"\n}\n',
    )

    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(lang_dir / "ru_ru.snbt"), target_lang=lang, mode="append")

    assert fake_service.calls == []  # nothing needed AI translation at all

    split_dir = lang_dir / "ru_ru"
    chapter_top = parse_lang_entries((split_dir / "chapter.snbt").read_text(encoding="utf-8"))
    assert chapter_top == {"chapter.AAAA000000000001.title": "Теургия"}
    chapter_quests = parse_lang_entries((split_dir / "chapters" / "theurgy.snbt").read_text(encoding="utf-8"))
    assert chapter_quests == {"quest.BBBB000000000002.title": "Крепкий щит"}


def test_process_reminds_to_reload_not_to_run_split_when_split_files_are_written(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    """Regression coverage for the real corruption: telling the user to run
    /langsplitter split after this processor already wrote a correct
    translation would regenerate English templates that the next reload
    then merges over it, wiping the translation back to English."""
    lang_dir = _full_mc_dir_layout(tmp_path, chapter_id="AAAA000000000001", quest_id="BBBB000000000002")
    _write(lang_dir / "en_us.snbt", '{\n\tchapter.AAAA000000000001.title: "Theurgy"\n}\n')

    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(lang_dir / "ru_ru.snbt"), target_lang=lang, mode="append")

    messages = [msg for msg, _tag in fake_callbacks.logs]
    assert not any("выполните команду" in m and "/langsplitter split" in m for m in messages)
    assert any("зайдите заново" in m or "/ftbquests reload" in m for m in messages)


def test_process_does_not_regress_an_already_translated_split_file_in_force_mode(
    tmp_path, fake_callbacks, job_state, lang
):
    """The exact scenario that corrupted a real pack's ru_ru.snbt: force
    mode plus a transient engine failure must never regress an already-
    correct split file back to English -- same guarantee the flat file
    already had, now extended to the files that actually matter in-game."""
    lang_dir = _full_mc_dir_layout(tmp_path, chapter_id="AAAA000000000001", quest_id="BBBB000000000002")
    _write(lang_dir / "en_us.snbt", '{\n\tchapter.AAAA000000000001.title: "Theurgy"\n}\n')
    split_dir = lang_dir / "ru_ru"
    split_dir.mkdir()
    _write(split_dir / "chapter.snbt", '{\n\tchapter.AAAA000000000001.title: "Теургия"\n}\n')

    proc = QuestLangProcessor(_FailingService(), job_state, fake_callbacks)
    proc.process(str(lang_dir / "ru_ru.snbt"), target_lang=lang, mode="force")

    entries = parse_lang_entries((split_dir / "chapter.snbt").read_text(encoding="utf-8"))
    assert entries["chapter.AAAA000000000001.title"] == "Теургия"


def test_process_skips_split_file_writing_when_lang_splitter_mod_is_absent(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    lang_dir = tmp_path / "config" / "ftbquests" / "quests" / "lang"
    lang_dir.mkdir(parents=True)
    (tmp_path / "mods").mkdir()  # present but no langsplitter jar
    _write(lang_dir / "en_us.snbt", '{\n\tquest.ABC.title: "Hi"\n}\n')

    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(lang_dir / "ru_ru.snbt"), target_lang=lang, mode="append")

    # The flat file is still translated -- only the split-file layout is
    # skipped when the splitter mod isn't there to read it.
    assert len(fake_service.calls) == 1
    assert not (lang_dir / "ru_ru").exists()
    messages = [msg for msg, _tag in fake_callbacks.logs]
    assert any("перезайти в мир" in m or "перезапустить игру" in m for m in messages)


# ---------------------------------------------------------------------
# Post-write reminder (FTB Quests Lang Splitter needs an explicit
# /langsplitter split <locale> command -- see quest_lang.py's
# _remind_to_split docstring for how this was confirmed by decompiling the
# companion mod's classes)
# ---------------------------------------------------------------------


def _mc_dir_layout(tmp_path):
    """Real ATM10-shaped tree: <mc_dir>/config/ftbquests/quests/lang/ and
    <mc_dir>/mods/ -- needed because _remind_to_split walks up from
    file_path's directory to find mods/, so a flat tmp_path (used by every
    other test in this file) can't exercise the detection."""
    lang_dir = tmp_path / "config" / "ftbquests" / "quests" / "lang"
    lang_dir.mkdir(parents=True)
    return lang_dir


def test_falls_back_to_generic_reminder_when_mod_present_but_nothing_split_routed(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    """Mod present, but no quests/chapters/ tree exists to map quest.ABC to
    a chapter -- nothing can be split-file-routed, so this must fall back
    to the generic reminder rather than staying silent. Also locks in that
    the old "/langsplitter split" suggestion is gone: running that command
    is what corrupted a real pack's translation (see module docstring)."""
    lang_dir = _mc_dir_layout(tmp_path)
    mods_dir = tmp_path / "mods"
    mods_dir.mkdir()
    (mods_dir / "ftbquestslangsplitter-1.0.5.jar").write_bytes(b"")
    _write(lang_dir / "en_us.snbt", '{\n\tquest.ABC.title: "Hi"\n}\n')

    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(lang_dir / "ru_ru.snbt"), target_lang=lang, mode="append")

    messages = [msg for msg, _tag in fake_callbacks.logs]
    assert not any("/langsplitter split" in m for m in messages)
    assert any("перезайти в мир" in m or "перезапустить игру" in m for m in messages)


def test_reminds_generically_when_the_mod_is_absent(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    lang_dir = _mc_dir_layout(tmp_path)
    (tmp_path / "mods").mkdir()  # present but empty -- no langsplitter jar
    _write(lang_dir / "en_us.snbt", '{\n\tquest.ABC.title: "Hi"\n}\n')

    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(lang_dir / "ru_ru.snbt"), target_lang=lang, mode="append")

    messages = [msg for msg, _tag in fake_callbacks.logs]
    assert not any("/langsplitter" in m for m in messages)
    assert any("перезайти в мир" in m or "перезапустить игру" in m for m in messages)


def test_no_reminder_when_nothing_was_translated(tmp_path, fake_service, fake_callbacks, job_state, lang):
    lang_dir = _mc_dir_layout(tmp_path)
    _write(lang_dir / "en_us.snbt", '{\n\tquest.ABC.title: "Привет"\n}\n')
    _write(lang_dir / "ru_ru.snbt", '{\n\tquest.ABC.title: "Привет"\n}\n')  # already fully translated

    proc = QuestLangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(lang_dir / "ru_ru.snbt"), target_lang=lang, mode="append")

    assert fake_callbacks.logs == []
