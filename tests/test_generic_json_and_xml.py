"""Tests for GenericJsonProcessor's append-mode value preservation and
XmlProcessor's positional-path round-trip -- both regressions found and
fixed in this project:

- GenericJsonProcessor used to only ever write freshly-translated
  (`pending`) values into the output; any already-translated key not in
  `pending` this run was never re-applied from the existing target file, so
  every append-mode run silently reverted previously-translated strings
  back to English.
- XmlProcessor's collect_translatable/apply_translated used to correlate
  findings via id(element), which is NOT stable for any non-root element
  across two separate tree traversals with lxml (a fresh proxy object, with
  a different id(), can be returned on each `for child in element`
  iteration) -- so attribute/text translations on nested elements were
  silently dropped even though the string was translated and cached.
"""
import json

from mc_translator.processors.generic_json import GenericJsonProcessor
from mc_translator.processors.xml import XmlProcessor, apply_translated, collect_translatable

try:
    from lxml import etree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def test_generic_json_append_mode_preserves_previously_translated_keys(
    tmp_path, fake_service, job_state, fake_callbacks
):
    src = tmp_path / "en_us.json"
    src.write_text(
        json.dumps({"already_done": "Already Done", "brand_new": "Brand New"}), encoding="utf-8"
    )
    # Existing target already has a real (different-from-source) translation
    # for "already_done", and nothing for "brand_new".
    target = tmp_path / "ru_ru.json"
    target.write_text(json.dumps({"already_done": "Уже готово"}), encoding="utf-8")

    proc = GenericJsonProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        str(src),
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="append",
        output_mode="inplace",
        mc_dir=str(tmp_path),
    )

    result = json.loads(target.read_text(encoding="utf-8"))
    # Previously-translated value must survive, NOT revert to English.
    assert result["already_done"] == "Уже готово"
    # Freshly-translated value must be present too.
    assert "TR[Brand New]" in result["brand_new"]
    # Only the brand-new string should have gone through translate_dict.
    translated_strings = {k: v for call, _ in fake_service.calls for k, v in call.items()}
    assert "Already Done" not in translated_strings.values()


def test_generic_json_skip_mode_does_not_retranslate_completed_strings(
    tmp_path, fake_service, job_state, fake_callbacks
):
    src = tmp_path / "en_us.json"
    src.write_text(json.dumps({"greeting": "Hello"}), encoding="utf-8")
    target = tmp_path / "ru_ru.json"
    target.write_text(json.dumps({"greeting": "Привет"}), encoding="utf-8")

    proc = GenericJsonProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        str(src),
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="skip",
        output_mode="inplace",
        mc_dir=str(tmp_path),
    )
    assert fake_service.calls == []
    result = json.loads(target.read_text(encoding="utf-8"))
    assert result["greeting"] == "Привет"


def test_xml_collect_and_apply_round_trips_nested_elements():
    """Regression test for the id()-instability bug: a nested (non-root)
    element's text/attribute must survive collect_translatable ->
    (simulated translation) -> apply_translated."""
    root = ET.fromstring('<entry title="Bar Entry"><page text="Some xml text"/></entry>')
    attrs = {"title", "name", "header", "label", "text"}
    to_translate = collect_translatable(root, attrs, "force", {"regex": r"[А-Яа-яЁё]"})

    assert len(to_translate) == 2  # root's title@, nested page's text@

    translated = {k: f"TR[{v}]" for k, v in to_translate.items()}
    apply_translated(root, translated)

    assert root.attrib["title"] == "TR[Bar Entry]"
    page = list(root)[0]
    assert page.attrib["text"] == "TR[Some xml text]"


def test_xml_collect_and_apply_round_trips_deeply_nested_text_and_tail():
    root = ET.fromstring(
        "<root>Intro text<a>Inner A</a>Tail after A<b><c>Deep C</c></b></root>"
    )
    to_translate = collect_translatable(root, set(), "force", {"regex": r"[А-Яа-яЁё]"})
    translated = {k: f"TR[{v}]" for k, v in to_translate.items()}
    apply_translated(root, translated)

    assert root.text == "TR[Intro text]"
    a = list(root)[0]
    assert a.text == "TR[Inner A]"
    assert a.tail == "TR[Tail after A]"
    b = list(root)[1]
    c = list(b)[0]
    assert c.text == "TR[Deep C]"


def test_xml_processor_append_mode_preserves_previously_translated_content(
    tmp_path, fake_service, job_state, fake_callbacks
):
    """Regression test for the missing merge-back in XmlProcessor.process():
    apply_translated used to only ever be called with this run's fresh
    translations, so a second append-mode run silently reverted already-
    translated text/attributes back to English (existing_map was used to
    EXCLUDE already-done paths from to_translate, but was never re-applied
    to `root` before the file was written back out)."""
    src = tmp_path / "en_us.xml"
    src.write_text(
        '<root><entry title="Hello">Hello body</entry></root>', encoding="utf-8"
    )
    lang = {"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"}

    proc = XmlProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        str(src), target_lang=lang, mode="append", output_mode="inplace", mc_dir=str(tmp_path)
    )

    # Source file grows a brand-new second entry (simulates a modpack update
    # adding new content alongside already-translated content).
    src.write_text(
        '<root><entry title="Hello">Hello body</entry>'
        '<entry title="World">World body</entry></root>',
        encoding="utf-8",
    )
    proc.process(
        str(src), target_lang=lang, mode="append", output_mode="inplace", mc_dir=str(tmp_path)
    )

    target = tmp_path / "ru_ru.xml"
    root = ET.parse(str(target)).getroot()
    entries = list(root)
    assert entries[0].attrib["title"] == "TR[Hello]"
    assert entries[0].text.strip() == "TR[Hello body]"
    # New entry must have been freshly translated.
    assert entries[1].attrib["title"] == "TR[World]"
    assert entries[1].text.strip() == "TR[World body]"

    # The already-translated strings must not have been sent through
    # translate_dict a second time.
    second_run_strings = set(fake_service.calls[1][0].values())
    assert "Hello" not in second_run_strings
    assert "Hello body" not in second_run_strings


def test_xml_processor_force_mode_retranslates_already_translated_content(
    tmp_path, fake_service, job_state, fake_callbacks
):
    """Regression test: collect_translatable used to run AFTER seeding root
    with the existing translation, so looks_like_source_language saw the
    already-Cyrillic text and excluded it regardless of mode -- silently
    defeating "force" mode's "retranslate everything" contract on any node
    that already had a prior translation."""
    src = tmp_path / "en_us.xml"
    src.write_text('<root><entry title="Hello">Hello body</entry></root>', encoding="utf-8")
    lang = {"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"}

    proc = XmlProcessor(fake_service, job_state, fake_callbacks)
    proc.process(str(src), target_lang=lang, mode="append", output_mode="inplace", mc_dir=str(tmp_path))

    fake_service.calls.clear()
    proc.process(str(src), target_lang=lang, mode="force", output_mode="inplace", mc_dir=str(tmp_path))

    target = tmp_path / "ru_ru.xml"
    root = ET.parse(str(target)).getroot()
    entry = root[0]
    assert entry.attrib["title"] == "TR[Hello]"
    assert entry.text.strip() == "TR[Hello body]"
    # Force mode must have actually re-sent both strings for translation,
    # not silently skipped them because they already looked translated.
    assert len(fake_service.calls) == 1
    assert set(fake_service.calls[0][0].values()) == {"Hello", "Hello body"}


def test_xml_apply_translated_preserves_whitespace_around_inline_tag():
    """Regression test: collect_translatable strips text/tail before sending
    to translate_dict, but apply_translated used to write the translated
    (already-stripped) string straight back, collapsing the whitespace
    between an inline tag and surrounding prose, e.g. "Place the <item>Altar
    </item> nearby now." losing the spaces around <item>.

    Note: the tail is deliberately multi-word ("nearby now.", not "nearby.")
    -- a single dotted word matches is_technical_term's "looks like a
    dotted identifier" heuristic and would never reach translate_dict at
    all, which would defeat the point of this test.
    """
    root = ET.fromstring("<p>Place the <item>Altar</item> nearby now.</p>")
    to_translate = collect_translatable(root, set(), "force", {"regex": r"[А-Яа-яЁё]"})
    translated = {k: f"TR[{v}]" for k, v in to_translate.items()}
    apply_translated(root, translated)

    item = list(root)[0]
    assert root.text == "TR[Place the] "
    assert item.text == "TR[Altar]"
    assert item.tail == " TR[nearby now.]"
    # Reassembled mixed content must keep single spaces around the tag.
    assert root.text.endswith(" ")
    assert item.tail.startswith(" ")


def test_generic_json_processor_handles_utf8_bom(tmp_path, fake_service, job_state, fake_callbacks):
    """Regression test: a UTF-8 BOM in the source JSON used to raise
    JSONDecodeError (encoding="utf-8" does not strip the BOM), silently
    skipping the whole file."""
    src = tmp_path / "en_us.json"
    src.write_bytes(b"\xef\xbb\xbf" + json.dumps({"greeting": "Hello"}).encode("utf-8"))

    proc = GenericJsonProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        str(src),
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="force",
        output_mode="inplace",
        mc_dir=str(tmp_path),
    )

    target = tmp_path / "ru_ru.json"
    result = json.loads(target.read_text(encoding="utf-8"))
    assert result["greeting"] == "TR[Hello]"


def test_generic_json_handles_jsonc_line_comments(tmp_path, fake_service, job_state, fake_callbacks):
    """Regression test: some mods (lithostitched.json, polylib.json in a
    real ATM10 run) ship JSON5/JSONC-style config with real "//" line
    comments, which bare json.loads() rejects outright with
    JSONDecodeError, silently skipping the whole file."""
    src = tmp_path / "en_us.json"
    src.write_text(
        '{\n  // This is a real line comment, not a "//" key.\n  "greeting": "Hello"\n}\n',
        encoding="utf-8",
    )

    proc = GenericJsonProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        str(src),
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="force",
        output_mode="inplace",
        mc_dir=str(tmp_path),
    )

    error_logs = [msg for msg, tag in fake_callbacks.logs if tag == "red"]
    assert not error_logs, f"unexpected errors logged: {error_logs}"
    target = tmp_path / "ru_ru.json"
    result = json.loads(target.read_text(encoding="utf-8"))
    assert result["greeting"] == "TR[Hello]"


def test_generic_json_handles_slash_containing_comment_keys(
    tmp_path, fake_service, job_state, fake_callbacks
):
    """Regression test: a real-world JSON shape from the "attributefix" mod
    -- {"modify_range": {"//": "...", "value": false}, "min": {"//": "...",
    "value": 0.0}} -- has three sibling top-level keys ("modify_range",
    "min", "max") that each nest a "//" comment key. The old "/".join(path)
    scheme joined ("modify_range", "//") into the string "modify_range///",
    which key.split("/") then mis-reconstructed as FOUR segments instead of
    two, silently failing to write the translated comment back (and
    silently failing the append-mode existing-value seed/lookup the same
    way). Path tuples must round-trip exactly for every sibling."""
    src = tmp_path / "en_us.json"
    src.write_text(
        json.dumps({
            "modify_range": {"//": "Determines if the range should be modified.", "value": False},
            "min": {"//": "The lowest possible value.", "value": 0.0},
            "max": {"//": "The highest possible value.", "value": 1024.0},
        }),
        encoding="utf-8",
    )

    proc = GenericJsonProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        str(src),
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="force",
        output_mode="inplace",
        mc_dir=str(tmp_path),
    )

    error_logs = [msg for msg, tag in fake_callbacks.logs if tag == "red"]
    assert not error_logs, f"unexpected errors logged: {error_logs}"

    target = tmp_path / "ru_ru.json"
    result = json.loads(target.read_text(encoding="utf-8"))
    assert result["modify_range"]["//"] == "TR[Determines if the range should be modified.]"
    assert result["min"]["//"] == "TR[The lowest possible value.]"
    assert result["max"]["//"] == "TR[The highest possible value.]"


def test_generic_json_slash_containing_keys_survive_append_mode(
    tmp_path, fake_service, job_state, fake_callbacks
):
    """The seed/lookup side of the same collision: a previously-translated
    "//" comment must be recognized as already-done (not re-sent to
    translate_dict) and must survive being re-applied to a fresh append run."""
    src = tmp_path / "en_us.json"
    src.write_text(
        json.dumps({
            "modify_range": {"//": "Determines if the range should be modified.", "value": False},
            "min": {"//": "The lowest possible value.", "value": 0.0},
        }),
        encoding="utf-8",
    )
    target = tmp_path / "ru_ru.json"
    target.write_text(
        json.dumps({
            "modify_range": {"//": "Уже переведено.", "value": False},
        }),
        encoding="utf-8",
    )

    proc = GenericJsonProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        str(src),
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="append",
        output_mode="inplace",
        mc_dir=str(tmp_path),
    )

    result = json.loads(target.read_text(encoding="utf-8"))
    # Already-translated comment survives untouched, not reverted/re-sent.
    assert result["modify_range"]["//"] == "Уже переведено."
    # The sibling "min"."//" (never translated before) gets freshly translated.
    assert result["min"]["//"] == "TR[The lowest possible value.]"
    translated_strings = {v for call, _ in fake_service.calls for v in call.values()}
    assert "The lowest possible value." in translated_strings
    assert "Determines if the range should be modified." not in translated_strings
