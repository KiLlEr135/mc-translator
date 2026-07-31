"""Tests for mc_translator.processors.guide_md -- the structure-preserving
guide/manual markdown translator that replaced jar.py's old
`_process_book_md`. Covers the exact regressions found in a real ATM10 run
against Applied Energistics 2's shipped guide (see guide_md.py's module
docstring): headings glued into body text, missing whitespace around inline
component tags, collapsed blank lines/CRLF frontmatter lists, and Russian
output overwriting the English root (and other languages' shipped guide
translations) instead of landing in a "_ru_ru/" subfolder."""
from mc_translator.processors.guide_md import (
    count_book_md_blocks,
    count_guide_blocks,
    guide_target_path,
    is_localized_guide_path,
    is_target_locale_path,
    translate_guide_markdown,
)
from mc_translator.text_processing import mask_protected_fragments, unmask_translation

LANG = {"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"}


def _shield_upper(pending: dict[str, str]) -> dict[str, str]:
    """Stand-in for TranslationService.translate_dict that still runs the
    real Titanium Shield mask/unmask around a trivial uppercase
    "translation". This exercises the actual division of labor these tests
    care about: guide_md decides WHAT to send as one translation unit; the
    shield (exercised for real here, not stubbed) decides what inside that
    unit is protected and restores it byte-for-byte."""
    out = {}
    for key, value in pending.items():
        masked, mapping = mask_protected_fragments(value)
        out[key] = unmask_translation(masked.upper(), mapping)
    return out


# ---------------------------------------------------------------------------
# guide_target_path / is_localized_guide_path / is_target_locale_path
# ---------------------------------------------------------------------------


def test_guide_target_path_inserts_lang_subfolder_for_guideme_layout():
    path = "assets/ae2/ae2guide/ae2-mechanics/channels.md"
    assert guide_target_path(path, "ru_ru") == "assets/ae2/ae2guide/_ru_ru/ae2-mechanics/channels.md"


def test_guide_target_path_substitutes_en_us_for_patchouli_layout():
    path = "assets/mymod/patchouli_books/mybook/en_us/entries/foo.md"
    assert guide_target_path(path, "ru_ru") == "assets/mymod/patchouli_books/mybook/ru_ru/entries/foo.md"


def test_guide_target_path_returns_none_for_already_localized_source():
    # Own prior output.
    assert guide_target_path("assets/ae2/ae2guide/_ru_ru/x.md", "ru_ru") is None
    # Another mod-shipped locale -- must never be treated as an English
    # source and overwritten (this is exactly what corrupted the ATM10 pack's
    # Spanish/Portuguese/Japanese AE2 guide translations).
    assert guide_target_path("assets/advanced_ae/ae2guide/_es_es/x.md", "ru_ru") is None


def test_is_localized_guide_path():
    assert is_localized_guide_path("assets/ae2/ae2guide/_es_mx/x.md")
    assert not is_localized_guide_path("assets/ae2/ae2guide/x.md")


def test_is_target_locale_path_matches_both_conventions():
    assert is_target_locale_path("assets/x/lang/ru_ru/y.json", "ru_ru")
    assert is_target_locale_path("assets/ae2/ae2guide/_ru_ru/x.md", "ru_ru")
    assert not is_target_locale_path("assets/ae2/ae2guide/_es_es/x.md", "ru_ru")


# ---------------------------------------------------------------------------
# translate_guide_markdown -- structure preservation
# ---------------------------------------------------------------------------


def test_heading_is_not_glued_into_following_paragraph():
    en = "# Channels\n\nApplied Energistics requires channels for devices.\n"
    out = translate_guide_markdown(en, "", "force", LANG, _shield_upper)
    lines = out.split("\n")
    assert lines[0] == "# CHANNELS"
    assert lines[1] == ""
    assert lines[2] == "APPLIED ENERGISTICS REQUIRES CHANNELS FOR DEVICES."


def test_blank_line_between_paragraphs_is_preserved():
    en = "First paragraph line one.\nFirst paragraph line two.\n\nSecond paragraph.\n"
    out = translate_guide_markdown(en, "", "force", LANG, _shield_upper)
    lines = out.split("\n")
    assert lines[1] == ""
    # en ends in "\n", so split("\n") has a trailing "" element too --
    # translate_guide_markdown must preserve that trailing newline exactly.
    assert len(lines) == 4
    assert lines[-1] == ""


def test_space_before_inline_tag_is_preserved():
    en = 'Some devices are <ItemLink id="me_p2p_tunnel" /> and more text.\n'
    out = translate_guide_markdown(en, "", "force", LANG, _shield_upper)
    # The core regression: a manual tag-split used to .strip() each fragment,
    # fusing the preceding word straight onto the tag ("are<ItemLink...").
    # A real single space must survive on both sides of the tag.
    assert ' <ItemLink id="me_p2p_tunnel" /> ' in out


def test_pure_tag_lines_are_left_completely_untouched():
    en = (
        '<GameScene zoom="7" interactive={true}>\n'
        '  <ImportStructure src="../assets/x.snbt" />\n'
        "</GameScene>\n"
    )
    out = translate_guide_markdown(en, "", "force", LANG, _shield_upper)
    # Lines with no prose outside their tag(s) are never sent for
    # translation at all -- byte-identical output, indentation included.
    assert out == en


def test_annotation_line_with_inline_text_is_translated_tags_preserved():
    en = '<DiamondAnnotation pos="1 1 1" color="#ff0000">\nAll channels used.\n</DiamondAnnotation>\n'
    out = translate_guide_markdown(en, "", "force", LANG, _shield_upper)
    lines = out.split("\n")
    assert lines[0] == '<DiamondAnnotation pos="1 1 1" color="#ff0000">'
    assert lines[1] == "ALL CHANNELS USED."
    assert lines[2] == "</DiamondAnnotation>"


def test_crlf_yaml_frontmatter_list_is_untouched():
    en = (
        "---\r\n"
        "navigation:\r\n"
        "  title: Quantum Computer\r\n"
        "item_ids:\r\n"
        "  - advanced_ae:quantum_unit\r\n"
        "  - advanced_ae:quantum_core\r\n"
        "---\r\n"
        "\r\n"
        "Some body text.\r\n"
    )
    out = translate_guide_markdown(en, "", "force", LANG, _shield_upper)
    assert "item_ids:\r\n  - advanced_ae:quantum_unit\r\n  - advanced_ae:quantum_core\r\n" in out


def test_frontmatter_title_is_translated_other_keys_untouched():
    en = "---\nnavigation:\n  parent: x.md\n  title: Channels\n  icon: controller\n---\nBody.\n"
    out = translate_guide_markdown(en, "", "force", LANG, _shield_upper)
    assert "title: CHANNELS" in out
    assert "parent: x.md" in out
    assert "icon: controller" in out


def test_list_items_stay_on_separate_lines():
    en = "Intro line.\n\n- First item text\n- Second item text\n"
    out = translate_guide_markdown(en, "", "force", LANG, _shield_upper)
    lines = out.split("\n")
    assert lines[2] == "- FIRST ITEM TEXT"
    assert lines[3] == "- SECOND ITEM TEXT"


# ---------------------------------------------------------------------------
# translate_guide_markdown -- mode-aware reuse (append/skip)
# ---------------------------------------------------------------------------


def test_append_mode_reuses_existing_paragraph_translation():
    en = "# Title\n\nHello world, this is a test paragraph.\n"
    tr = "# ЗАГОЛОВОК\n\nПривет мир, это тестовый абзац.\n"
    calls = []

    def _batch(pending):
        calls.append(dict(pending))
        return _shield_upper(pending)

    out = translate_guide_markdown(en, tr, "append", LANG, _batch)
    assert not calls  # nothing should have been sent for (re)translation
    assert "Привет мир, это тестовый абзац." in out
    assert "ЗАГОЛОВОК" in out


def test_append_mode_translates_new_paragraph_not_present_before():
    en = "# Title\n\nFirst paragraph.\n\nSecond paragraph is new.\n"
    tr = "# ЗАГОЛОВОК\n\nПервый абзац.\n"
    out = translate_guide_markdown(en, tr, "append", LANG, _shield_upper)
    assert "Первый абзац." in out
    assert "SECOND PARAGRAPH IS NEW." in out


def test_skip_mode_reuses_already_translated_and_retranslates_rest():
    en = "First paragraph.\n\nSecond paragraph.\n"
    tr = "Первый абзац.\n\nSecond paragraph.\n"  # 2nd block never got translated
    out = translate_guide_markdown(en, tr, "skip", LANG, _shield_upper)
    lines = out.split("\n")
    assert lines[0] == "Первый абзац."
    assert lines[2] == "SECOND PARAGRAPH."


# ---------------------------------------------------------------------------
# count_guide_blocks / count_book_md_blocks (Analyze screen / cost estimate)
# ---------------------------------------------------------------------------


def test_count_book_md_blocks_counts_heading_and_paragraph():
    text = "# Heading\n\nSome prose paragraph here.\n"
    assert count_book_md_blocks(text) == 2


def test_count_guide_blocks_reports_already_translated():
    en = "# Title\n\nFirst paragraph.\n\nSecond paragraph.\n"
    tr = "# ЗАГОЛОВОК\n\nПервый абзац.\n\nSecond paragraph.\n"
    total, done = count_guide_blocks(en, tr, LANG["regex"])
    assert total == 3
    assert done == 2
