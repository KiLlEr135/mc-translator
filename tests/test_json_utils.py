"""Tests for mc_translator.json_utils — JSON parsing/path helpers shared by jar.py,
generic_json.py, analyzer.py and estimator.py."""

from mc_translator.json_utils import (
    apply_translations_by_path,
    get_at_path,
    iter_all_json_strings,
    iter_translatable_strings,
    key_to_path,
    load_lenient_json,
    path_to_key,
    set_at_path,
)

# ---------------------------------------------------------------------------
# load_lenient_json
# ---------------------------------------------------------------------------

def test_load_lenient_json_trailing_comma_in_object():
    assert load_lenient_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_load_lenient_json_trailing_comma_in_array():
    assert load_lenient_json('{"a": [1, 2, 3,]}') == {"a": [1, 2, 3]}


def test_load_lenient_json_line_comments():
    # Only whole-line // comments are stripped (the regex anchors on ^\s*),
    # not a trailing // after code on the same line.
    raw = '{\n  // this is a comment\n  "a": 1,\n  "b": 2\n}'
    assert load_lenient_json(raw) == {"a": 1, "b": 2}


def test_load_lenient_json_block_comments():
    raw = '{\n  /* block comment */\n  "a": 1\n}'
    assert load_lenient_json(raw) == {"a": 1}


def test_load_lenient_json_accepts_bytes_with_bom():
    raw = '{"a": 1}'
    assert load_lenient_json(raw.encode("utf-8-sig")) == {"a": 1}


def test_load_lenient_json_strict_input_still_works():
    assert load_lenient_json('{"clean": true}') == {"clean": True}


# ---------------------------------------------------------------------------
# path_to_key / key_to_path
# ---------------------------------------------------------------------------

def test_path_to_key_joins_with_slash():
    assert path_to_key(("pages", 0, "text")) == "pages/0/text"


def test_key_to_path_keeps_components_as_strings():
    """key_to_path deliberately does NOT guess int-vs-string from content —
    that decision is deferred to get_at_path/set_at_path based on the actual
    container type at each step (regression test for the digit-string-dict-key
    bug: a bare 'isdigit() -> int' conversion here broke real dict keys that
    happen to be numeric strings)."""
    assert key_to_path("pages/0/text") == ("pages", "0", "text")


# ---------------------------------------------------------------------------
# get_at_path / set_at_path
# ---------------------------------------------------------------------------

def test_get_set_at_path_through_list_index():
    data = {"pages": [{"text": "Page zero"}, {"text": "Page one"}]}
    path = key_to_path("pages/1/text")
    assert get_at_path(data, path) == "Page one"
    set_at_path(data, path, "Страница один")
    assert data["pages"][1]["text"] == "Страница один"
    assert data["pages"][0]["text"] == "Page zero"  # untouched


def test_get_set_at_path_with_digit_string_dict_key():
    """A dict keyed by a literal digit string ('1') must NOT be treated as a
    list index — this is the exact scenario that broke before the fix."""
    data = {"variants": {"1": {"text": "Hello"}}}
    path = key_to_path("variants/1/text")
    assert get_at_path(data, path) == "Hello"
    set_at_path(data, path, "Привет")
    assert data == {"variants": {"1": {"text": "Привет"}}}


# ---------------------------------------------------------------------------
# iter_translatable_strings (book/guide JSON, KEYS_TO_TRANSLATE allowlist)
# ---------------------------------------------------------------------------

def test_iter_translatable_strings_yields_plain_allowlisted_key():
    data = {"name": "Guide", "randomKey": "not picked up"}
    result = list(iter_translatable_strings(data))
    assert result == [(("name",), "Guide")]


def test_iter_translatable_strings_recurses_into_pages_array_of_dicts():
    """Regression test: a Patchouli-style {'pages': [{'text': '...'}]} array
    of page-objects used to be silently dropped entirely (the KEYS_TO_TRANSLATE
    'if' branch didn't recurse into non-string list items)."""
    book = {
        "name": "Guide",
        "pages": [
            {"type": "text", "text": "Some long paragraph."},
            {"type": "text", "text": "Another page."},
        ],
    }
    result = list(iter_translatable_strings(book))
    assert (("name",), "Guide") in result
    assert (("pages", 0, "text"), "Some long paragraph.") in result
    assert (("pages", 1, "text"), "Another page.") in result
    assert len(result) == 3


def test_iter_translatable_strings_recurses_into_non_allowlisted_dict():
    data = {"container": {"name": "Nested Name"}}
    result = list(iter_translatable_strings(data))
    assert result == [(("container", "name"), "Nested Name")]


def test_iter_translatable_strings_plain_string_list_under_allowlisted_key():
    data = {"text": ["line one", "line two"]}
    result = list(iter_translatable_strings(data))
    assert result == [(("text", 0), "line one"), (("text", 1), "line two")]


# ---------------------------------------------------------------------------
# iter_all_json_strings (arbitrary mod/config JSON, no key allowlist)
# ---------------------------------------------------------------------------

def test_iter_all_json_strings_yields_every_string_regardless_of_key():
    data = {
        "tooltip": "Right click to open",
        "nested": {"buttonLabel": "Confirm"},
        "list": ["a", "b"],
        "number": 42,
        "flag": True,
    }
    result = list(iter_all_json_strings(data))
    assert (("tooltip",), "Right click to open") in result
    assert (("nested", "buttonLabel"), "Confirm") in result
    assert (("list", 0), "a") in result
    assert (("list", 1), "b") in result
    assert len(result) == 4  # number/flag are not strings, correctly excluded


def test_iter_all_json_strings_handles_top_level_list():
    data = ["first", {"key": "second"}, ["third"]]
    result = list(iter_all_json_strings(data))
    assert (( 0,), "first") in result
    assert ((1, "key"), "second") in result
    assert ((2, 0), "third") in result


# ---------------------------------------------------------------------------
# apply_translations_by_path (end-to-end with both iterators)
# ---------------------------------------------------------------------------

def test_apply_translations_by_path_end_to_end_with_book_iterator():
    book = {"name": "Guide", "pages": [{"text": "Hello"}, {"text": "World"}]}
    strings = {path_to_key(p): s for p, s in iter_translatable_strings(book)}
    translated = {key: f"[TR]{value}" for key, value in strings.items()}
    apply_translations_by_path(book, translated)
    assert book == {
        "name": "[TR]Guide",
        "pages": [{"text": "[TR]Hello"}, {"text": "[TR]World"}],
    }


def test_apply_translations_by_path_end_to_end_with_generic_iterator():
    data = {"tooltip": "Open menu", "nested": {"label": "Confirm"}}
    strings = {path_to_key(p): s for p, s in iter_all_json_strings(data)}
    translated = {key: value.upper() for key, value in strings.items()}
    apply_translations_by_path(data, translated)
    assert data == {"tooltip": "OPEN MENU", "nested": {"label": "CONFIRM"}}
