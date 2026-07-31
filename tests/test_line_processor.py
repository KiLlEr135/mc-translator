"""Tests for mc_translator.processors.line_processor -- the shared
cfg/bat/mcfunction/text line-based translator used by CfgProcessor,
BatProcessor, McfunctionProcessor and TextProcessor."""
from mc_translator.processors.cfg import CfgProcessor
from mc_translator.processors.line_processor import (
    bat_candidates,
    cfg_candidates,
    mcfunction_candidates,
    text_candidates,
)
from mc_translator.processors.text import TextProcessor


def test_cfg_candidates_matches_hash_and_slash_comments():
    lines = ["# comment one\n", "normal = value\n", "// comment two\n", "\n"]
    assert cfg_candidates(lines) == {0, 2}


def test_bat_candidates_matches_rem_and_double_colon():
    lines = ["REM a comment\n", "echo hi\n", ":: another comment\n"]
    assert bat_candidates(lines) == {0, 2}


def test_mcfunction_candidates_matches_hash_only():
    lines = ["# a comment\n", "say hello\n"]
    assert mcfunction_candidates(lines) == {0}


def test_text_candidates_skips_fenced_code_block_content():
    lines = [
        "Normal text\n",
        "```\n",
        "code here, not a candidate\n",
        "```\n",
        "More normal text\n",
    ]
    assert text_candidates(lines) == {0, 4}


def test_cfg_processor_translates_comment_lines_only(tmp_path, fake_service, fake_callbacks, job_state, lang):
    src = tmp_path / "en_us.cfg"
    src.write_text("# Hello world\nkey = value\n", encoding="utf-8")
    proc = CfgProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(src), target_lang=lang, mode="force", output_mode="inplace", mc_dir=str(tmp_path))

    out_path = tmp_path / "ru_ru.cfg"
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "TR[# Hello world]" in content
    assert "key = value" in content  # not a comment -- left untouched


def test_crlf_line_endings_are_preserved(tmp_path, fake_service, fake_callbacks, job_state, lang):
    src = tmp_path / "en_us.cfg"
    src.write_bytes(b"# Hello world\r\nkey = value\r\n")
    proc = CfgProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(src), target_lang=lang, mode="force", output_mode="inplace", mc_dir=str(tmp_path))

    out_bytes = (tmp_path / "ru_ru.cfg").read_bytes()
    assert b"TR[# Hello world]\r\n" in out_bytes


def test_text_processor_skips_translating_inside_code_fence(tmp_path, fake_service, fake_callbacks, job_state, lang):
    src = tmp_path / "en_us.txt"
    src.write_text("Some prose here\n```\ncode.example()\n```\nMore prose\n", encoding="utf-8")
    proc = TextProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(src), target_lang=lang, mode="force", output_mode="inplace", mc_dir=str(tmp_path))

    content = (tmp_path / "ru_ru.txt").read_text(encoding="utf-8")
    assert "TR[Some prose here]" in content
    assert "TR[More prose]" in content
    assert "code.example()" in content  # inside the fence -- untouched


def test_append_mode_preserves_already_translated_lines(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    """append mode's "already handled" check must be POSITIONAL -- the
    existing target's line at index idx corresponds to the source's line at
    the same idx (both files share the same structure). A previous version
    compared the English source line against the *set* of existing target
    lines, which never matched real translated content -- see
    test_append_mode_retranslates_lines_still_english_in_existing_target for
    the regression this caused."""
    src = tmp_path / "en_us.cfg"
    src.write_text("# Already done\n# Needs translation\n", encoding="utf-8")
    (tmp_path / "ru_ru.cfg").write_text("# Уже сделано\n# Needs translation\n", encoding="utf-8")

    proc = CfgProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(src), target_lang=lang, mode="append", output_mode="inplace", mc_dir=str(tmp_path))

    assert len(fake_service.calls) == 1
    assert list(fake_service.calls[0][0].values()) == ["# Needs translation"]

    content = (tmp_path / "ru_ru.cfg").read_text(encoding="utf-8")
    assert "# Уже сделано" in content  # preserved, not reverted to English
    assert "TR[# Needs translation]" in content


def test_append_mode_retranslates_lines_still_english_in_existing_target(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    """Regression test: before the positional fix, append mode compared the
    English source line against the set of existing target lines, so a
    sidecar that (aborted first run, copy-paste seed, etc.) still has the
    literal English text at a position was wrongly treated as "already
    translated" and never queued -- permanently stuck untranslated. The fix
    compares positionally: identical-to-source at that index means NOT
    translated yet."""
    src = tmp_path / "en_us.cfg"
    src.write_text("# Still English\n", encoding="utf-8")
    (tmp_path / "ru_ru.cfg").write_text("# Still English\n", encoding="utf-8")

    proc = CfgProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(src), target_lang=lang, mode="append", output_mode="inplace", mc_dir=str(tmp_path))

    content = (tmp_path / "ru_ru.cfg").read_text(encoding="utf-8")
    assert "TR[# Still English]" in content


def test_append_mode_translates_lines_not_in_existing_target(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    src = tmp_path / "en_us.cfg"
    src.write_text("# Brand new comment\n", encoding="utf-8")

    proc = CfgProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(src), target_lang=lang, mode="append", output_mode="inplace", mc_dir=str(tmp_path))

    content = (tmp_path / "ru_ru.cfg").read_text(encoding="utf-8")
    assert "TR[# Brand new comment]" in content


def test_append_mode_falls_back_to_translating_all_when_line_counts_differ(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    """If the source's line count changed since the last run (e.g. a mod
    update added/removed lines), the existing target is no longer
    positionally aligned with the source -- comparing by index would compare
    unrelated lines. Safe fallback: translate every candidate, same as if
    there were no existing target at all."""
    src = tmp_path / "en_us.cfg"
    src.write_text("# One\n# Two\n# Three\n", encoding="utf-8")
    (tmp_path / "ru_ru.cfg").write_text("# Один\n# Два\n", encoding="utf-8")

    proc = CfgProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(src), target_lang=lang, mode="append", output_mode="inplace", mc_dir=str(tmp_path))

    assert len(fake_service.calls) == 1
    assert set(fake_service.calls[0][0].values()) == {"# One", "# Two", "# Three"}


def test_skip_mode_respects_target_language_regex(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    src = tmp_path / "en_us.cfg"
    src.write_text("# Already ru\n# Still english\n", encoding="utf-8")
    (tmp_path / "ru_ru.cfg").write_text("# Уже переведено\n# Still english\n", encoding="utf-8")

    proc = CfgProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(src), target_lang=lang, mode="skip", output_mode="inplace", mc_dir=str(tmp_path))

    assert len(fake_service.calls) == 1
    assert list(fake_service.calls[0][0].values()) == ["# Still english"]

    content = (tmp_path / "ru_ru.cfg").read_text(encoding="utf-8")
    assert "# Уже переведено" in content
    assert "TR[# Still english]" in content


def test_force_mode_does_not_overwrite_existing_output_when_stopped_mid_run(
    tmp_path, fake_callbacks, job_state, lang
):
    """Regression test (H3): in force mode, out_lines starts from the
    English placeholder for every candidate line (unlike append/skip, which
    pre-seed already-translated lines) and is only overwritten for lines
    translate_dict actually returned. If the job is Stopped mid-translation
    (should_run() flips False), translate_dict returns a PARTIAL result --
    so without a should_run() guard on the write, a previously
    fully-translated output file gets clobbered with a mix of fresh
    translations and English placeholders for the untranslated tail."""
    src = tmp_path / "en_us.cfg"
    src.write_text("# Line one\n# Line two\n# Line three\n", encoding="utf-8")

    out_path = tmp_path / "ru_ru.cfg"
    previously_translated = "# СтрокаОдин\n# СтрокаДва\n# СтрокаТри\n"
    out_path.write_text(previously_translated, encoding="utf-8")

    class StoppingService:
        def translate_dict(self, strings, target_lang, callbacks, *, context="", usage_label=None):
            # Simulate a Stop arriving mid-translation: should_run() flips
            # False partway through, so only the first requested key comes
            # back translated (mirroring TranslationService's own
            # should_run()-gated loop).
            job_state.is_running = False
            first_key = next(iter(strings))
            return {first_key: f"TR[{strings[first_key]}]"}

    proc = CfgProcessor(StoppingService(), job_state, fake_callbacks)
    proc.process(str(src), target_lang=lang, mode="force", output_mode="inplace", mc_dir=str(tmp_path))

    assert out_path.read_text(encoding="utf-8") == previously_translated
