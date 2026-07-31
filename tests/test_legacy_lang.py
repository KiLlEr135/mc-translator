"""Tests for mc_translator.utils.legacy_lang -- shared pre-1.13 lang-file
conventions used by lang.py, jar.py and pack_writer.py."""
from mc_translator.utils.legacy_lang import is_legacy_version, legacy_lang_code, legacy_lang_filename


def test_is_legacy_version_true_for_1_12_and_earlier():
    assert is_legacy_version("1.12.2") is True
    assert is_legacy_version("1.7.10") is True


def test_is_legacy_version_false_for_1_13_and_later():
    assert is_legacy_version("1.20.1") is False
    assert is_legacy_version("1.13.0") is False


def test_is_legacy_version_false_for_unparseable_string():
    assert is_legacy_version("not-a-version") is False


def test_legacy_lang_code_converts_region_case():
    assert legacy_lang_code("ru_ru") == "ru_RU"


def test_legacy_lang_code_idempotent_on_already_legacy_case():
    assert legacy_lang_code("ru_RU") == "ru_RU"


def test_legacy_lang_code_leaves_non_lang_region_shape_unchanged():
    assert legacy_lang_code("en") == "en"


def test_legacy_lang_filename_converts_stem_only():
    assert legacy_lang_filename("ru_ru.lang") == "ru_RU.lang"
    assert legacy_lang_filename("ru_ru.json") == "ru_RU.json"


def test_legacy_lang_filename_idempotent():
    assert legacy_lang_filename("ru_RU.lang") == "ru_RU.lang"
