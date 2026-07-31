"""Tests for mc_translator.text_processing — the string-safety layer every
translation engine relies on (mask/unmask in particular: it gates every
string sent to Google/DeepL/Kobold/OpenRouter)."""
import pytest

from mc_translator import text_processing
from mc_translator.text_processing import (
    already_translated,
    apply_smart_glue,
    has_foreign_script_contamination,
    has_unresolved_placeholders,
    is_only_placeholders,
    is_technical_term,
    looks_like_source_language,
    mask_protected_fragments,
    polish_translation,
    unmask_translation,
    unmask_translation_strict,
)

# ---------------------------------------------------------------------------
# mask_protected_fragments / unmask_translation
# ---------------------------------------------------------------------------
# These two are always used as a pair around a translation engine call:
# masked, mapping = mask_protected_fragments(text)  -> sent to the engine
# unmask_translation(engine_output, mapping)        -> shown to the user
# If the engine echoes the masked text back verbatim (the strictest possible
# baseline — no actual translation happening), unmask must reproduce the
# protected fragments byte-for-byte. That's the real contract these tests
# protect: an engine can only be trusted to preserve [#n#] tokens, never
# original formatting codes/placeholders, so masking must be lossless for them.

PROTECTED_FRAGMENT_SAMPLES = [
    "Place the &6Ancient Debris&r inside the furnace.",
    "Use $(item) to open the menu.",
    "Insert <tag> here please.",
    "Player {player} has joined the game.",
    "Click ![here](https://example.com) for info.",
    "See [modid:blockname] for details.",
    "Refer to (modid:blockname) documentation.",
    "Check (guide.md#section) for more.",
    "You need %s more items (%d total).",
    "Damage: %1$s and %.2f multiplier.",
    "The GUI shows RF/t values clearly.",
    "Plain text with no protected fragments at all.",
]


@pytest.mark.parametrize("text", PROTECTED_FRAGMENT_SAMPLES)
def test_mask_unmask_round_trip_when_engine_echoes_masked_text(text):
    masked, mapping = mask_protected_fragments(text)
    restored = unmask_translation(masked, mapping)
    # No newlines/extra whitespace in these samples, so masking is a true
    # identity when nothing was actually translated.
    assert restored == text


def test_mask_does_not_swallow_prose_inside_angle_brackets():
    """Regression test (H1): the old <[^>]+> pattern greedily masked ANY
    <...> span, including plain prose comparisons like "< 10 mana >" that
    have nothing to do with a real tag -- leaving them untranslated. Real
    tags (Patchouli/AE2 item tags, XML-ish markup) still start with a
    letter (or '/') right after '<' and must still be protected (see the
    test below)."""
    masked, mapping = mask_protected_fragments("Requires < 10 mana > to cast.")
    assert "10 mana" in masked
    assert not mapping


def test_mask_still_protects_real_angle_bracket_tags():
    masked, mapping = mask_protected_fragments("Insert <item:minecraft:dirt> here.")
    assert "<item:minecraft:dirt>" not in masked
    assert "<item:minecraft:dirt>" in mapping.values()


def test_mask_does_not_swallow_prose_inside_curly_braces():
    """Regression test (H1): the old \\{[^\\}]+\\} pattern masked ANY {...}
    span, including free prose like "{one or more}" -- leaving it
    untranslated. Only placeholder-shaped content (word chars/dots, no
    spaces) is a real protected fragment."""
    masked, mapping = mask_protected_fragments("Choose {one or more} options.")
    assert "one or more" in masked
    assert not mapping


def test_mask_still_protects_placeholder_shaped_curly_braces():
    masked, mapping = mask_protected_fragments("Player {player_name} joined.")
    assert "{player_name}" not in masked
    assert "{player_name}" in mapping.values()


def test_mask_replaces_format_codes_and_ignore_terms_with_placeholders():
    masked, mapping = mask_protected_fragments("The &6GUI&r shows RF/t clearly.")
    assert "&6" not in masked
    assert "GUI" not in masked
    assert "RF/t" not in masked
    # &6/GUI/&r sit back-to-back with nothing between them, so they coalesce
    # into one combined placeholder (see mask_protected_fragments' docstring)
    # -- RF/t is separated from anything else by real prose ("shows"/
    # "clearly") and stays its own token.
    assert set(mapping.values()) == {"&6GUI&r", "RF/t"}


def test_mask_coalesces_a_run_of_stacked_format_codes_into_one_token():
    """Regression test: a real production incident found long Minecraft
    strings with several stacked format codes (e.g. "&6&l&n...") failing
    to translate identically across OpenRouter/NVIDIA/local KoboldCPP --
    traced to mask_protected_fragments emitting one [#n#] placeholder PER
    individual code, so "&6&l&n" became 3 back-to-back tokens with nothing
    real between them. Any LLM tends to "tidy up" what looks like bracket
    noise (merge/reorder/drop one), and the strict all-or-nothing restore
    then discarded the whole otherwise-fine translation. Coalescing the
    run into a single token removes the compounding failure at its root."""
    masked, mapping = mask_protected_fragments("&6&l&nTitle&r")
    assert masked == "[#0#]Title[#1#]"
    assert mapping == {"[#0#]": "&6&l&n", "[#1#]": "&r"}


def test_mask_does_not_coalesce_format_codes_separated_by_real_prose():
    """Regression guard against over-coalescing: two format codes with
    actual translatable text between them must stay two separate tokens,
    not get merged just because they're both format codes."""
    masked, mapping = mask_protected_fragments("Place the &6Ancient Debris&r here.")
    assert masked == "Place the [#0#]Ancient Debris[#1#] here."
    assert mapping == {"[#0#]": "&6", "[#1#]": "&r"}


def test_is_only_placeholders_true_when_source_had_no_real_prose():
    """Regression test: "§f%s" is a bare color code glued to a sprintf
    placeholder with nothing else -- both format codes are adjacent, so
    coalescing collapses them into one token and nothing is left over."""
    masked, _mapping = mask_protected_fragments("§f%s")
    assert masked == "[#0#]"
    assert is_only_placeholders(masked) is True


def test_is_only_placeholders_false_when_real_prose_remains():
    masked, _mapping = mask_protected_fragments("Place the &6Ancient Debris&r here.")
    assert is_only_placeholders(masked) is False


def test_is_only_placeholders_false_for_plain_text_with_no_tokens():
    assert is_only_placeholders("hello world") is False


@pytest.mark.parametrize(
    "text",
    [
        "Two lines\nglued together.",
        "Three\n\nlines\n\nhere.",
        "Trailing newline at the end.\n",
    ],
)
def test_mask_unmask_preserves_newlines_exactly(text):
    """Each \\n is matched by FORMAT_PATTERN and protected individually, so
    (unlike other whitespace) newlines survive the round trip unchanged."""
    masked, mapping = mask_protected_fragments(text)
    restored = unmask_translation(masked, mapping)
    assert restored == text


@pytest.mark.parametrize(
    "text,collapsed",
    [
        ("Multiple   spaces   here.", "Multiple spaces here."),
        ("  Leading and trailing spaces.  ", "Leading and trailing spaces."),
        ("Tab\there.", "Tab here."),
    ],
)
def test_mask_unmask_collapses_non_newline_whitespace(text, collapsed):
    """Runs of spaces/tabs (not newlines) are intentionally normalized to a
    single space and stripped — this is the one case round-tripping is NOT
    a full identity, by design."""
    masked, mapping = mask_protected_fragments(text)
    restored = unmask_translation(masked, mapping)
    assert restored == collapsed
    assert restored != text


def test_unmask_translation_restores_fullwidth_bracket_variants():
    """Regression test: weak/CJK-target translators commonly rewrite ASCII
    [ ] around a shield placeholder into full-width ［］ or 【】 (a common
    typographic substitution next to CJK text). unmask_translation used to
    match ONLY ASCII brackets, so the format code the placeholder protected
    (e.g. %s) was silently lost and a literal ［#0#］ shipped into the mod."""
    masked, mapping = mask_protected_fragments("Use %s here")
    assert masked == "Use [#0#] here"
    for opening, closing in (("［", "］"), ("【", "】")):
        mangled = masked.replace("[", opening).replace("]", closing)
        restored = unmask_translation(mangled, mapping)
        assert restored == "Use %s here"
        assert not has_unresolved_placeholders(restored)


def test_has_unresolved_placeholders_detects_surviving_token():
    _masked, mapping = mask_protected_fragments("Use %s here")
    # Simulate a translator mangling the bracket into something unmask
    # cannot recognize at all -- the placeholder core "#0#" survives.
    broken = unmask_translation("Use <<#0#>> here", mapping)
    assert has_unresolved_placeholders(broken)


def test_has_unresolved_placeholders_false_for_clean_text():
    masked, mapping = mask_protected_fragments("Use %s here")
    restored = unmask_translation(masked, mapping)
    assert not has_unresolved_placeholders(restored)
    assert not has_unresolved_placeholders("Plain text with a # hashtag and a number 42.")


def test_unmask_translation_strict_restores_when_all_tokens_present():
    masked, mapping = mask_protected_fragments("Use %s here")
    assert unmask_translation_strict(masked, mapping) == "Use %s here"


def test_unmask_translation_strict_restores_fullwidth_bracket_variants():
    masked, mapping = mask_protected_fragments("Use %s here")
    mangled = masked.replace("[", "［").replace("]", "］")
    assert unmask_translation_strict(mangled, mapping) == "Use %s here"


def test_unmask_translation_strict_returns_none_when_token_fully_deleted():
    """Regression test (C1): a translator that DROPS a shield placeholder
    entirely (no bracket, no "#n#" trace at all -- e.g. an LLM that
    "cleans up" what looks like a stray artifact) is a real failure mode
    has_unresolved_placeholders alone cannot detect from the output text,
    since nothing of the token survives to search for. unmask_translation_
    strict must catch this by verifying every mapping token was actually
    located and substituted."""
    masked, mapping = mask_protected_fragments("Use %s here")
    fully_deleted = masked.replace("[#0#]", "")
    assert unmask_translation_strict(fully_deleted, mapping) is None


def test_unmask_translation_strict_returns_none_when_bracket_unrecognizable():
    masked, mapping = mask_protected_fragments("Use %s here")
    mangled = masked.replace("[", "<<").replace("]", ">>")
    assert unmask_translation_strict(mangled, mapping) is None


def test_unmask_translation_applies_actual_translated_text():
    original = "Place the &6Ancient Debris&r here."
    masked, mapping = mask_protected_fragments(original)
    # Simulate a real translated response using the same placeholders.
    fake_translated = masked.replace("Place the", "Помести").replace("here", "тут")
    result = unmask_translation(fake_translated, mapping)
    assert "&6Ancient Debris&r" in result
    assert "Помести" in result and "тут" in result


# ---------------------------------------------------------------------------
# has_foreign_script_contamination
# ---------------------------------------------------------------------------
# Regression tests: user reported seeing the tool once output Chinese
# characters instead of a real translation. Root cause -- CJK/Kana/Hangul
# are deliberately excluded from FOREIGN_SCRIPT_RE because they're this
# tool's own zh_cn/ja_jp/ko_kr TARGET languages, but that exclusion applied
# unconditionally, so a free/low-quality model answering in Chinese for a
# NON-CJK target (e.g. Russian) passed the check silently.

RU_TARGET = {"name": "Russian", "regex": r"[А-Яа-яЁё]"}
ZH_TARGET = {"name": "Simplified Chinese", "regex": r"[一-鿿]"}


def test_foreign_script_contamination_flags_cjk_for_a_non_cjk_target():
    assert has_foreign_script_contamination("Некоторые предметы 手斧", RU_TARGET) is True


def test_foreign_script_contamination_allows_cjk_for_a_cjk_target():
    assert has_foreign_script_contamination("这是一把手斧", ZH_TARGET) is False


def test_foreign_script_contamination_still_flags_arabic_regardless_of_target():
    assert has_foreign_script_contamination("Некоторые لاунчеры", RU_TARGET) is True


def test_foreign_script_contamination_flags_cjk_when_no_target_given():
    """No target_lang means no confirmation any script is expected -- stay
    conservative rather than silently reproducing the reported bug."""
    assert has_foreign_script_contamination("手斧") is True


def test_foreign_script_contamination_false_for_clean_translation():
    assert has_foreign_script_contamination("Обычный текст", RU_TARGET) is False


# ---------------------------------------------------------------------------
# apply_smart_glue
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("This is a sentence\nthat continues.", "This is a sentence that continues."),
        ("End of paragraph.\nNext sentence starts fresh.", "End of paragraph.\nNext sentence starts fresh."),
        ("Header line\n# Next section", "Header line\n# Next section"),
        ("List item\n- another item", "List item\n- another item"),
        ("yaml_key:\nsome value", "yaml_key:\nsome value"),
        ("", ""),
    ],
)
def test_apply_smart_glue(text, expected):
    assert apply_smart_glue(text) == expected


def test_apply_smart_glue_preserves_sentence_ending_before_break():
    # Preceding '.', '!', '?', '>', ']', ':' blocks gluing (paragraph/quote boundary).
    text = "Sentence one.\nSentence two starts new paragraph."
    assert apply_smart_glue(text) == text


# ---------------------------------------------------------------------------
# polish_translation
# ---------------------------------------------------------------------------

def test_polish_translation_tightens_format_code_spacing():
    assert polish_translation("&6 Ancient Debris") == "&6Ancient Debris"
    assert polish_translation("[ %s ]") == "[%s]"
    assert polish_translation("Value: % d") == "Value: %d"


def test_polish_translation_percent_regex_respects_word_boundary():
    """Regression test: without \\b, "%\\s+([sd])" also matched the "d" that
    starts the next word in other languages (e.g. French "50 % de degats"),
    gluing the percent sign straight onto it ("%de"). \\b restricts the fix
    to real %s/%d format placeholders, where s/d is NOT followed by another
    word character."""
    assert polish_translation("Le taux est de 50 % de degats") == "Le taux est de 50 % de degats"
    assert polish_translation("Value: % d") == "Value: %d"


def test_polish_translation_passthrough_for_non_string_or_empty():
    assert polish_translation("") == ""
    assert polish_translation(None) is None


def test_polish_translation_applies_dictionary_fix_with_case_matching(monkeypatch):
    monkeypatch.setitem(text_processing.TERMINOLOGY_FIXES, "полуслой", "плита")
    assert "плита" in polish_translation("Возьмите полуслой из сундука")
    assert "Плита" in polish_translation("Полуслой лежит в сундуке")


def test_polish_translation_cjk_dictionary_fix_skips_word_boundary(monkeypatch):
    """\\b never matches inside unbroken CJK text — the dictionary pass must
    special-case CJK terms (regression test for the Phase-1 fix)."""
    monkeypatch.setitem(text_processing.TERMINOLOGY_FIXES, "苹果", "梨")
    result = polish_translation("我喜欢苹果和香蕉")
    assert "梨" in result
    assert "苹果" not in result


# ---------------------------------------------------------------------------
# is_technical_term / looks_like_source_language / already_translated
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("block.name", True),
        ("some_id", True),
        ("some-id", False),
        ("12345", True),
        ("glyph_fire", True),
        ("kubejs_test", True),
        ("Hello World", False),
        ("RF/t", False),
        ("НЕ технический", True),  # no ASCII letters at all -> treated as technical
        ("", True),
        ("item.sword.name", True),
        ("#000000B4", True),  # hex color (with alpha) -- must not be sent to the AI
        ("#fff", True),  # short hex color
        ("#0078FFDC", True),  # hex color, mixed case digits
        ("minecraft:air", True),  # namespaced resource location
        ("eternal_starlight:saltpeter_powder", True),
        ("quad-mstv-mtv:pale_oak_torch", True),
        ("minecraft:block/stone", True),  # resource location with a path segment
        (
            "minecraft:dirt;minecraft:grass_block;minecraft:coarse_dirt",
            True,
        ),  # ";"-delimited list of resource locations (a real modpack config value --
        # translating words inside it corrupts the block-ID list)
        ("minecraft:dirt,minecraft:grass_block", True),  # ","-delimited variant
        # Regression tests: real production strings live-verified to be
        # queued for AI translation and fail every single run (empty
        # response / mangled shield) despite being pure non-prose data.
        ("infoDisplay:time:1:15:-1", True),  # scoreboard/objective id, 3+ bare colons
        ("Note: this is important", False),  # a real colon in real prose must NOT match
        (
            "source = [source:local]/config/fancymenu/assets/gui/inventory/side_button/textures/button.png",
            True,
        ),  # FancyMenu-style "key = [namespace:value]" config line, not prose
        ("message = [Warning] Something happened", False),  # bracket without a namespace: colon stays translatable
    ],
)
def test_is_technical_term(text, expected):
    assert is_technical_term(text) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hello world", True),
        ("12345", False),
        ("Привет мир", False),
        ("mixed Привет", True),
        ("", False),
    ],
)
def test_looks_like_source_language(text, expected):
    assert looks_like_source_language(text) is expected


def test_already_translated_uses_target_regex():
    ru_regex = r"[А-Яа-яЁё]"
    assert already_translated("Привет", ru_regex) is True
    assert already_translated("Hello", ru_regex) is False
    assert already_translated("Mixed Привет", ru_regex) is True
