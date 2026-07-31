import re

from mc_translator.text_processing import is_translation_key, looks_like_source_language

SNBT_STRING_RE = r'"((?:[^"\\]|\\.)*)"'
SINGLE_FIELDS = ("title", "subtitle", "text", "desc", "description", "name", "comment", "objective", "statement", "message")
ARRAY_FIELDS = ("description", "text", "desc", "pages")


def _unescape(value: str) -> str:
    return value.replace('\\"', '"').replace('\\\\', '\\')


def _escape(value: str) -> str:
    return value.replace('\\', '\\\\').replace('"', '\\"')


def extract_snbt_strings(
    content: str, *, skip_translated_regex: str | None = None, require_source_language: bool = True
) -> list[str]:
    """`require_source_language=False` bypasses the looks_like_source_language
    filter, returning every translatable-field value regardless of what
    language it's currently in -- used by callers that need a true total
    field count (e.g. a done-ratio check) rather than "still needs
    translation" candidates."""
    strings: list[str] = []
    # (?<![...]) / (?![...]) anchor the field name as a whole token, so e.g.
    # "displayName"/"nickname" (which merely END in "name") aren't swept up
    # by the bare "name" alternative -- must stay in sync with the same
    # anchoring in apply_snbt_translations() below, or apply would silently
    # skip fields extract found (stale English left in place).
    field_pattern = "|".join(SINGLE_FIELDS)
    for match in re.finditer(
        rf'(?:"|)(?<![A-Za-z0-9_])(?:{field_pattern})(?![A-Za-z0-9_])(?:"|)\s*:\s*{SNBT_STRING_RE}',
        content,
        re.IGNORECASE,
    ):
        value = match.group(1)
        # Unescape the extracted string: handle \" -> " and \\ -> \
        value = _unescape(value)
        if not value.strip() or is_translation_key(value):
            continue
        if require_source_language and not looks_like_source_language(value):
            continue
        if skip_translated_regex and re.search(skip_translated_regex, value):
            continue
        strings.append(value)

    array_fields = "|".join(ARRAY_FIELDS)
    for block in re.finditer(
        rf'(?:"|)(?<![A-Za-z0-9_])(?:{array_fields})(?![A-Za-z0-9_])(?:"|)\s*:\s*\[\s*(?:"(?:[^"\\]|\\.)*"\s*,?\s*)*\]',
        content,
        re.IGNORECASE,
    ):
        for sm in re.finditer(SNBT_STRING_RE, block.group(0)):
            value = sm.group(1)
            # Unescape the extracted string: handle \" -> " and \\ -> \
            value = _unescape(value)
            if not value.strip() or is_translation_key(value):
                continue
            if require_source_language and not looks_like_source_language(value):
                continue
            if skip_translated_regex and re.search(skip_translated_regex, value):
                continue
            strings.append(value)
    return list(dict.fromkeys(strings))


def apply_snbt_translations(content: str, mapping: dict[str, str]) -> str:
    """`mapping` is keyed by UNESCAPED source text (see extract_snbt_strings,
    which unescapes before storing). The regex below matches the raw,
    still-escaped file content, so the captured value must be unescaped
    before the lookup -- otherwise any string containing \\" or \\\\ never
    matches its mapping key (silently left untranslated) AND gets
    re-escaped on top of its already-escaped form on write-back (corrupting
    it) whenever it falls through to the `original` fallback.

    A field with no entry in `mapping` (not selected for translation this
    run, or a cache miss) must come back byte-for-byte unchanged -- so on a
    mapping miss we return the original matched text verbatim instead of
    round-tripping it through _unescape/_escape. _escape/_unescape only
    understand \\" and \\\\; any other backslash sequence already present in
    the file (e.g. FTB Quests' literal "\\n" line-break marker, or \\uXXXX)
    would otherwise get its backslash silently doubled on every run, even
    though nothing about that field was ever translated."""

    def replace_single(match: re.Match) -> str:
        original = _unescape(match.group(2))
        if original not in mapping:
            return match.group(0)
        escaped = _escape(mapping[original])
        return f'{match.group(1)}: "{escaped}"'

    # Anchoring must match extract_snbt_strings() exactly (see the comment
    # there) -- otherwise this would silently skip fields extract found (or
    # rewrite fields extract never offered for translation).
    field_pattern = "|".join(SINGLE_FIELDS)
    content = re.sub(
        rf'(?:"|)(?<![A-Za-z0-9_])({field_pattern})(?![A-Za-z0-9_])(?:"|)\s*:\s*{SNBT_STRING_RE}',
        replace_single,
        content,
        flags=re.IGNORECASE,
    )

    def replace_array(match: re.Match) -> str:
        key = match.group(1)
        array_body = match.group(2)

        def replace_item(sm: re.Match) -> str:
            original = _unescape(sm.group(1))
            if original not in mapping:
                return sm.group(0)
            escaped = _escape(mapping[original])
            return f'"{escaped}"'

        new_body = re.sub(SNBT_STRING_RE, replace_item, array_body)
        return f'{key}: {new_body}'

    array_fields = "|".join(ARRAY_FIELDS)
    content = re.sub(
        rf'(?:"|)(?<![A-Za-z0-9_])({array_fields})(?![A-Za-z0-9_])(?:"|)\s*:\s*(\[\s*(?:"(?:[^"\\]|\\.)*"\s*,?\s*)*\])',
        replace_array,
        content,
        flags=re.IGNORECASE,
    )
    return content
