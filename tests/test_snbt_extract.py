"""Tests for mc_translator.processors.snbt_extract — extracting and writing
back translatable strings in FTB Quests .snbt files, including the escape/
unescape round-trip for values containing quotes or backslashes."""
from mc_translator.processors.snbt_extract import apply_snbt_translations, extract_snbt_strings


def test_extract_simple_field():
    content = 'title: "Hello World"\n'
    assert extract_snbt_strings(content) == ["Hello World"]


def test_extract_unescapes_quotes_and_backslashes():
    content = 'text: "He said \\"hi\\" to me"\ndesc: "Path is C:\\\\mods\\\\file"\n'
    strings = extract_snbt_strings(content)
    assert 'He said "hi" to me' in strings
    assert "Path is C:\\mods\\file" in strings


def test_extract_skips_translation_keys():
    content = 'name: "quest.category.intro"\n'
    assert extract_snbt_strings(content) == []


def test_extract_array_field():
    content = 'pages: ["First page text", "Second page text"]\n'
    strings = extract_snbt_strings(content)
    assert "First page text" in strings
    assert "Second page text" in strings


def test_apply_translates_matching_strings():
    content = 'title: "Hello World"\n'
    result = apply_snbt_translations(content, {"Hello World": "Привет мир"})
    assert 'title: "Привет мир"' in result


def test_apply_round_trips_quote_and_backslash_content():
    """Regression test: apply_snbt_translations used to look up the mapping
    using the still-ESCAPED raw text (never matching the unescaped keys
    extract_snbt_strings produces), silently leaving these strings
    untranslated -- and re-escaping the already-escaped fallback on top,
    corrupting the file's escape sequences even when nothing was translated."""
    content = 'text: "He said \\"hi\\" to me"\ndesc: "Path is C:\\\\mods\\\\file"\n'
    strings = extract_snbt_strings(content)
    mapping = {s: f"TR[{s}]" for s in strings}
    result = apply_snbt_translations(content, mapping)

    # Re-extracting the translated output must yield exactly the translated
    # (unescaped) semantic values -- proving the escape sequences round-trip
    # correctly rather than being double-escaped or left untranslated.
    round_tripped = extract_snbt_strings(result)
    assert 'TR[He said "hi" to me]' in round_tripped
    assert "TR[Path is C:\\mods\\file]" in round_tripped


def test_apply_leaves_untranslated_strings_unchanged():
    """A string with no entry in `mapping` (filtered out during extraction,
    e.g. by is_translation_key or an already-target-language check) must
    round-trip through apply_snbt_translations byte-for-byte -- not get
    corrupted by an escape/unescape mismatch."""
    content = 'text: "He said \\"hi\\" to me"\nuntouched: "Nothing to translate"\n'
    result = apply_snbt_translations(content, {})
    assert content == result


def test_apply_skips_missing_keys_keeping_original():
    content = 'title: "Untranslated Title"\n'
    result = apply_snbt_translations(content, {"Some Other String": "Другое"})
    assert 'title: "Untranslated Title"' in result


def test_apply_leaves_untranslated_field_with_bare_backslash_byte_for_byte():
    """Regression test: _escape/_unescape only recognize \\" and \\\\ as
    escapes. A field containing some OTHER backslash sequence (e.g. the
    "\\n" line-break marker FTB Quests text commonly uses, which is a
    literal backslash followed by 'n' -- not an actual newline character)
    survives _unescape unchanged, but _escape then blindly doubles that
    lone backslash on write-back even though the field was never actually
    translated (mapping miss -> falls back to `original`). This corrupts
    the live quest file's line-break markers on every run. A field not in
    `mapping` must round-trip completely untouched."""
    content = 'desc: "Line1\\nLine2"\n'
    result = apply_snbt_translations(content, {})
    assert result == content


def test_extract_and_apply_do_not_treat_displayname_as_the_name_field():
    """Regression test: the unanchored field-name alternation matches "name"
    as a SUFFIX of "displayName" (no word boundary), so an unrelated key
    like displayName/nickname/surname gets swept into extraction and
    rewritten as if it were the dedicated `name` field. Field names must be
    matched as whole tokens."""
    content = 'displayName: "Not a quest name field"\n'
    assert extract_snbt_strings(content) == []
    result = apply_snbt_translations(content, {"Not a quest name field": "Другое"})
    assert result == content
