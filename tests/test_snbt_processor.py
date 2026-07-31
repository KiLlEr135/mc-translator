"""Tests for mc_translator.processors.snbt.SnbtProcessor -- the FTB Quests
.snbt file translator. Writes directly to the live quest file (via a
.tmp + os.replace swap), with a one-time .bak snapshot of the original --
the highest-risk untested surface identified in the pipeline audit before
this test file existed."""
from mc_translator.processors.snbt import SnbtProcessor


def test_force_mode_translates_and_creates_backup(tmp_path, fake_service, fake_callbacks, job_state, lang):
    src = tmp_path / "chapter.snbt"
    src.write_text('title: "Hello World"\n', encoding="utf-8")

    proc = SnbtProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(src), target_lang=lang, mode="force")

    assert src.read_text(encoding="utf-8") == 'title: "TR[Hello World]"\n'
    backup = tmp_path / "chapter.snbt.bak"
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == 'title: "Hello World"\n'


def test_backup_is_only_created_once(tmp_path, fake_service, fake_callbacks, job_state, lang):
    """A second run must not overwrite the .bak with the now-translated
    content -- the backup is meant to always hold the ORIGINAL English."""
    src = tmp_path / "chapter.snbt"
    src.write_text('title: "Hello World"\n', encoding="utf-8")
    backup = tmp_path / "chapter.snbt.bak"

    proc = SnbtProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(src), target_lang=lang, mode="force")
    first_backup_content = backup.read_text(encoding="utf-8")

    # Force mode re-reads from the .bak (source_path = backup for non-append
    # modes), so a second force run re-translates from the SAME original.
    proc.process(str(src), target_lang=lang, mode="force")

    assert backup.read_text(encoding="utf-8") == first_backup_content
    assert backup.read_text(encoding="utf-8") == 'title: "Hello World"\n'


def test_append_mode_skips_already_target_language_strings(tmp_path, fake_service, fake_callbacks, job_state, lang):
    src = tmp_path / "chapter.snbt"
    src.write_text('title: "Привет мир"\ndesc: "New description"\n', encoding="utf-8")

    proc = SnbtProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(src), target_lang=lang, mode="append")

    content = src.read_text(encoding="utf-8")
    assert 'title: "Привет мир"' in content  # already Russian -- untouched
    assert 'desc: "TR[New description]"' in content
    assert len(fake_service.calls) == 1
    assert list(fake_service.calls[0][0].values()) == ["New description"]


def test_no_translatable_strings_leaves_file_untouched(tmp_path, fake_service, fake_callbacks, job_state, lang):
    src = tmp_path / "chapter.snbt"
    original = 'name: "quest.category.intro"\n'
    src.write_text(original, encoding="utf-8")

    proc = SnbtProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(src), target_lang=lang, mode="force")

    assert src.read_text(encoding="utf-8") == original
    assert fake_service.calls == []


def test_force_mode_does_not_drop_quest_added_after_backup(tmp_path, fake_service, fake_callbacks, job_state, lang):
    """A stale .bak snapshot must not silently delete live-only content: if
    the live file gained a new field since the backup was taken (e.g. a
    modpack update added a quest), force mode must still translate it
    instead of reconstructing the file from the outdated backup alone."""
    src = tmp_path / "chapter.snbt"
    src.write_text('title: "Quest One"\n', encoding="utf-8")

    proc = SnbtProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(src), target_lang=lang, mode="force")
    assert src.read_text(encoding="utf-8") == 'title: "TR[Quest One]"\n'

    # Simulate a modpack update adding a second quest to the live file --
    # the .bak still only knows about the first one.
    src.write_text('title: "Quest One"\ntitle: "Quest Two"\n', encoding="utf-8")

    proc.process(str(src), target_lang=lang, mode="force")

    content = src.read_text(encoding="utf-8")
    assert 'title: "TR[Quest One]"' in content
    assert 'title: "TR[Quest Two]"' in content
    backup = tmp_path / "chapter.snbt.bak"
    assert 'Quest Two' in backup.read_text(encoding="utf-8")


def test_skip_mode_translates_remaining_below_90_percent_threshold(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    """Below the 90%-done threshold, skip mode must translate the
    remaining untranslated strings (like append) instead of bailing out
    entirely just because *some* string in the file is already translated."""
    src = tmp_path / "chapter.snbt"
    src.write_text('title: "Привет мир"\ndesc: "New description"\n', encoding="utf-8")

    proc = SnbtProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(src), target_lang=lang, mode="skip")

    content = src.read_text(encoding="utf-8")
    assert 'title: "Привет мир"' in content
    assert 'desc: "TR[New description]"' in content
    assert len(fake_service.calls) == 1


def test_skip_mode_bails_when_at_least_90_percent_already_translated(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    # 9 distinct already-translated titles + 1 still-English desc == 90%
    # done (extract_snbt_strings dedupes by value, so each must be unique).
    src_lines = [f'title: "Перевод {i}"\n' for i in range(9)] + ['desc: "Still English"\n']
    src = tmp_path / "chapter.snbt"
    src.write_text("".join(src_lines), encoding="utf-8")

    proc = SnbtProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(src), target_lang=lang, mode="skip")

    assert fake_service.calls == []
    assert src.read_text(encoding="utf-8") == "".join(src_lines)
