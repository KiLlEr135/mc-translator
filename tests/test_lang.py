"""Tests for mc_translator.processors.lang.LangProcessor -- the .lang
(key=value) localization file translator."""
from mc_translator.processors.lang import LangProcessor


def test_translates_new_keys_inplace(tmp_path, fake_service, fake_callbacks, job_state, lang):
    src = tmp_path / "en_us.lang"
    src.write_text("greeting=Hello\nfarewell=Goodbye\n", encoding="utf-8")
    proc = LangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(src), target_lang=lang, mode="force", output_mode="inplace", mc_dir=str(tmp_path))

    content = (tmp_path / "ru_ru.lang").read_text(encoding="utf-8")
    assert "greeting=TR[Hello]" in content
    assert "farewell=TR[Goodbye]" in content


def test_append_mode_does_not_revert_already_translated_keys_when_new_keys_are_added(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    """Regression test: a mod update adds new keys to en_us.lang. append
    mode must translate only the new keys while leaving the already-
    translated ones exactly as they were in the existing target file --
    NOT rebuild them from the (still-English) in-memory kv_values, which
    silently reverts previously translated entries back to English the
    moment there's at least one new key to translate in the same run."""
    src = tmp_path / "en_us.lang"
    src.write_text(
        "greeting=Hello\nfarewell=Goodbye\nnew_key=Brand new string\n",
        encoding="utf-8",
    )
    (tmp_path / "ru_ru.lang").write_text(
        "greeting=Привет\nfarewell=Пока\n",
        encoding="utf-8",
    )

    proc = LangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(src), target_lang=lang, mode="append", output_mode="inplace", mc_dir=str(tmp_path))

    content = (tmp_path / "ru_ru.lang").read_text(encoding="utf-8")
    assert "greeting=Привет" in content, "already-translated key must survive untouched"
    assert "farewell=Пока" in content, "already-translated key must survive untouched"
    assert "new_key=TR[Brand new string]" in content

    # Only the genuinely new key should have been sent for translation.
    assert len(fake_service.calls) == 1
    assert list(fake_service.calls[0][0].values()) == ["Brand new string"]


def test_skip_mode_does_not_revert_already_translated_keys_when_new_keys_are_added(
    tmp_path, fake_service, fake_callbacks, job_state, lang
):
    src = tmp_path / "en_us.lang"
    src.write_text(
        "greeting=Hello\nnew_key=Brand new string\n",
        encoding="utf-8",
    )
    (tmp_path / "ru_ru.lang").write_text("greeting=Привет\n", encoding="utf-8")

    proc = LangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(src), target_lang=lang, mode="skip", output_mode="inplace", mc_dir=str(tmp_path))

    content = (tmp_path / "ru_ru.lang").read_text(encoding="utf-8")
    assert "greeting=Привет" in content
    assert "new_key=TR[Brand new string]" in content


def test_comments_and_blank_lines_preserved(tmp_path, fake_service, fake_callbacks, job_state, lang):
    src = tmp_path / "en_us.lang"
    src.write_text("# a comment\n\ngreeting=Hello\n", encoding="utf-8")
    proc = LangProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(src), target_lang=lang, mode="force", output_mode="inplace", mc_dir=str(tmp_path))

    content = (tmp_path / "ru_ru.lang").read_text(encoding="utf-8")
    assert "# a comment" in content
    assert "\n\n" in content
