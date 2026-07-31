"""
Structure-preserving translator for guide/manual markdown pages (GuideME's
AE2 guide, and other mods using the same `/ae2guide/`, `/guide/`, `/manual/`,
`/lexicon/` conventions -- see MD_PATH_MARKERS).

Split out of jar.py's old `_process_book_md`, which had two compounding bugs:

1. It ran `apply_smart_glue` (meant for reflowing a single hard-wrapped
   paragraph) over the ENTIRE file at once, then translated non-yaml/non-"<"
   lines one physical line at a time. Since apply_smart_glue's regex only
   refuses to glue across a preceding '.', '!', '?', '>', ']', ':' (see
   text_processing.py), it happily glued a lone "# Channels" heading straight
   into the prose paragraph that followed it, and glued right past blank
   lines when the line before/after didn't end in one of those characters --
   collapsing paragraph breaks and welding headings into body text.
2. With no "/en_us/" segment in the path (GuideME's actual layout -- pages
   live at the guide root, not under a per-language subfolder), it fell back
   to `tr_path = fl`, i.e. it wrote the Russian translation OVER the English
   source file itself instead of into a "_ru_ru/" subfolder. GuideME's own
   documented convention (https://guideme.appliedenergistics.org/translation)
   is a per-language subfolder next to the page's normal location --
   "_<langcode>/folder/page.md" -- with the root falling back to English.
   Writing over the root also meant any OTHER language's shipped guide
   translation living at "_es_es/...", "_pt_br/...", etc. -- which matches
   the exact same MD_PATH_MARKERS gate -- got treated as "the English source"
   and clobbered with a Russian translation the next time this ran.

This module fixes both: `guide_target_path` computes the correct "_<lang>/"
(or "/en_us/"-substituted) output path and refuses to touch anything that's
already under a language subfolder, and `translate_guide_markdown` splits the
document into small, purpose-classified blocks (frontmatter title / heading /
structural single line / prose paragraph) so each is translated as its own
unit without smart_glue ever seeing more than one paragraph's worth of text.
"""
from __future__ import annotations

import re

from mc_translator.text_processing import already_translated, is_technical_term, looks_like_source_language

# Directory segment names that mark the root of a guide's page tree -- see
# MD_PATH_MARKERS in constants.py (kept as slash-delimited substrings there
# for the is_book_md gate; here we need the bare segment name to know WHERE
# in the path to insert "_<lang>/").
GUIDE_DIR_MARKERS = ("ae2guide", "guide", "manual", "lexicon")

# Matches a GuideME-style language subfolder segment anywhere in a path, e.g.
# ".../ae2guide/_ru_ru/..." or ".../ae2guide/_es_mx/...". Used both to keep
# `_split_blocks`'s callers (jar.py) from treating an already-localized page
# as an English source, and to recognize this run's own target-locale output
# on a re-run.
_LOCALIZED_SEGMENT_RE = re.compile(r"(^|/)_[a-z]{2,3}_[a-z]{2,3}(/|$)", re.IGNORECASE)

_TITLE_RE = re.compile(r'^(\s*title\s*:\s*[\'"]?)(.*?)([\'"]?)$', re.IGNORECASE)
_HEADING_RE = re.compile(r"^(#{1,6}\s+)(\S.*)$")
_ORDERED_LIST_RE = re.compile(r"^\d+[.)]\s")
_TAG_RE = re.compile(r"<[^>]+>")

# Kinds of _split_blocks() output that carry translatable text.
_TRANSLATABLE_KINDS = ("title", "heading", "line", "para")


def is_localized_guide_path(path: str) -> bool:
    """True if `path` (a lowercased, '/'-separated zip/file path) already
    sits inside a GuideME language subfolder -- someone else's shipped
    translation (_es_es/, _pt_br/, ...) or our own prior output (_ru_ru/).
    Such a file must never be treated as an English translation source."""
    return bool(_LOCALIZED_SEGMENT_RE.search(path))


def is_target_locale_path(path: str, lang_file: str) -> bool:
    """True if `path` is (or would be) this run's target-language output --
    either the "/xx_xx/" folder convention used by Patchouli-style books and
    jar lang files, or GuideME's "_xx_xx/" convention (see
    guide_target_path). Callers use this to recognize entries that belong to
    the CURRENT target language specifically (as opposed to
    is_localized_guide_path, which matches ANY language)."""
    return f"/{lang_file}/" in path or f"_{lang_file}/" in path


def guide_target_path(path: str, lang_file: str) -> str | None:
    """Compute the translated-output path for a guide page at `path`
    (already lowercased, '/'-separated). Returns None if `path` is itself a
    localized variant -- it must be left untouched, never treated as a
    source.

    - If "/en_us/" is present (Patchouli-style per-language subtree), swap it
      for "/<lang_file>/" -- unchanged from the pre-existing behavior.
    - Otherwise (GuideME's actual layout: pages live at the guide root, no
      per-language subtree in the source), insert "_<lang_file>" as a new
      path segment right after the guide root directory, e.g.
      ".../ae2guide/x.md" -> ".../ae2guide/_ru_ru/x.md" -- matching GuideME's
      documented convention, with the English root left untouched for
      fallback.
    """
    if is_localized_guide_path(path):
        return None
    if "/en_us/" in path:
        return path.replace("/en_us/", f"/{lang_file}/")
    segments = path.split("/")
    for i, seg in enumerate(segments):
        if seg in GUIDE_DIR_MARKERS:
            return "/".join(segments[: i + 1] + [f"_{lang_file}"] + segments[i + 1 :])
    # No known guide-root marker segment found -- shouldn't happen given the
    # MD_PATH_MARKERS gate callers apply before reaching here, but fall back
    # to inserting right before the filename so we still never overwrite the
    # English source.
    segments.insert(len(segments) - 1, f"_{lang_file}")
    return "/".join(segments)


def _split_line(line: str) -> tuple[str, str]:
    """Split a physical line (from text.split('\\n')) into (content, cr) --
    cr is '\\r' if the line came from a CRLF source, else ''. Matching/
    classification runs on `content`; `cr` is re-appended verbatim on output
    so CRLF guide pages (e.g. this pack's AE2 guide .md files) don't grow a
    stray literal '\\r' baked into translated text or frontmatter."""
    if line.endswith("\r"):
        return line[:-1], "\r"
    return line, ""


def _is_structural(stripped: str) -> bool:
    """True for markdown lines whose leading syntax must stay pinned to that
    exact physical line: list items, blockquotes, tables, code fences, images,
    and raw component tags (<GameScene>, <ItemLink/>, etc). These are
    translated one physical line at a time (if they carry real text) instead
    of being merged into a paragraph block, so list/table layout survives.

    (Note: the original code's image check was `startswith("[!")`, which
    can never match real markdown image syntax "![alt](url)" -- fixed here
    to the correct "![" prefix.)
    """
    if stripped.startswith(("<", "-", "*", "+", ">", "|", "```", "![")):
        return True
    return bool(_ORDERED_LIST_RE.match(stripped))


def _has_real_text_outside_tags(content: str) -> bool:
    """True if `content` has visible prose outside of any <...> component
    tags -- distinguishes a pure tag line (e.g. '<GameScene zoom="7">' or
    '  <ImportStructure src="../assets/x.snbt" />', no prose at all) from a
    line that mixes a tag with real text (a one-line annotation, or a list
    item referencing an <ItemLink/>). Pure tag lines are left completely
    untouched (kind "verbatim") rather than sent through translate_batch --
    besides being unnecessary API calls, mask_protected_fragments collapses
    and trims whitespace around whatever it doesn't mask (see
    text_processing.py), which would otherwise silently eat a pure tag
    line's leading indentation on the round trip."""
    return bool(_TAG_RE.sub(" ", content).strip())


def _split_blocks(text: str, *, classify: bool = True) -> list[dict]:
    """Decompose guide markdown into an ordered list of blocks. Each block is
    a dict with a "kind":
      - "verbatim": passthrough line, never translated (blank lines,
        frontmatter delimiters/non-title keys, non-source-language or
        technical-term lines).
      - "title": a single frontmatter `title:` line.
      - "heading": a single `#`..`######` line.
      - "line": a single structural line that still carries real prose (list
        item, blockquote, or a component tag with inline text).
      - "para": a maximal run of consecutive plain-prose lines, merged into
        one translation unit -- this is what whole-file apply_smart_glue used
        to do, scoped here to just the paragraph itself so headings/blank
        lines/structural lines are never swallowed into it.

    "title"/"heading"/"line"/"para" blocks carry "text" (the string to
    translate) plus whatever's needed to reconstruct the original line(s)
    around it; see _render_block.

    `classify=False` skips the "looks like source language" / "is a
    technical term" content filters -- structure (heading/title/blank/
    frontmatter/tag-only-line) is still detected the same way, but every
    heading/structural/prose line is treated as translatable regardless of
    its language. Used to decompose a PRIOR TRANSLATION's text (already in
    the target language, e.g. Cyrillic for Russian) for block-position
    alignment in translate_guide_markdown/count_guide_blocks -- with
    classify=True, looks_like_source_language (which only recognizes
    a-zA-Z) would reject nearly everything already translated, making it
    look like nothing had ever been translated.
    """
    blocks: list[dict] = []
    lines = text.split("\n")
    in_yaml = False
    para_raw: list[str] = []
    para_crs: list[str] = []

    def eligible(s: str) -> bool:
        return (not classify) or (looks_like_source_language(s) and not is_technical_term(s))

    def flush_para() -> None:
        if para_raw:
            blocks.append({
                "kind": "para",
                "text": " ".join(para_raw),
                "raw": list(para_raw),
                "crs": list(para_crs),
            })
            para_raw.clear()
            para_crs.clear()

    for line in lines:
        content, cr = _split_line(line)
        stripped = content.strip()

        if stripped == "---":
            flush_para()
            in_yaml = not in_yaml
            blocks.append({"kind": "verbatim", "raw": [content], "crs": [cr]})
            continue

        if in_yaml:
            flush_para()
            if stripped.lower().startswith("title:"):
                m = _TITLE_RE.match(content)
                if m and eligible(m.group(2)):
                    blocks.append({
                        "kind": "title", "text": m.group(2),
                        "prefix": m.group(1), "suffix": m.group(3), "cr": cr,
                    })
                    continue
            blocks.append({"kind": "verbatim", "raw": [content], "crs": [cr]})
            continue

        if not stripped:
            flush_para()
            blocks.append({"kind": "verbatim", "raw": [content], "crs": [cr]})
            continue

        heading = _HEADING_RE.match(content)
        if heading:
            flush_para()
            prefix, htext = heading.groups()
            if eligible(htext):
                blocks.append({"kind": "heading", "text": htext, "prefix": prefix, "cr": cr})
            else:
                blocks.append({"kind": "verbatim", "raw": [content], "crs": [cr]})
            continue

        if _is_structural(stripped):
            flush_para()
            if _has_real_text_outside_tags(content) and eligible(content):
                blocks.append({"kind": "line", "text": content, "cr": cr})
            else:
                blocks.append({"kind": "verbatim", "raw": [content], "crs": [cr]})
            continue

        # Plain prose line -- accumulate into the current paragraph run.
        if eligible(content):
            para_raw.append(content)
            para_crs.append(cr)
        else:
            flush_para()
            blocks.append({"kind": "verbatim", "raw": [content], "crs": [cr]})

    flush_para()
    return blocks


def _render_block(block: dict, translated_text: str | None) -> str:
    kind = block["kind"]
    if kind == "verbatim":
        return "\n".join(r + c for r, c in zip(block["raw"], block["crs"]))
    text = translated_text if translated_text is not None else block["text"]
    if kind == "title":
        return block["prefix"] + text + block["suffix"] + block["cr"]
    if kind == "heading":
        return block["prefix"] + text + block["cr"]
    if kind == "line":
        return text + block["cr"]
    if kind == "para":
        if translated_text is None:
            return "\n".join(r + c for r, c in zip(block["raw"], block["crs"]))
        return text + (block["crs"][-1] if block["crs"] else "")
    raise AssertionError(f"unknown block kind: {kind}")  # pragma: no cover


def translate_guide_markdown(
    en_text: str,
    tr_text: str,
    mode: str,
    target_lang: dict,
    translate_batch,
) -> str:
    """Translate a guide markdown page while preserving its structure:
    headings stay on their own line, blank lines between paragraphs survive,
    inline component tags (<ItemLink/>, <GameScene>...) keep their
    surrounding whitespace, and CRLF line endings / multi-line YAML
    frontmatter lists are left untouched. Only "title"/"heading"/"line"/
    "para" blocks are ever sent for translation; everything else is copied
    through byte-for-byte.

    `translate_batch(pending: dict[str, str]) -> dict[str, str]` is expected
    to already run every value through the project's format-protection
    ("Titanium Shield") + cache + engine pipeline (see
    TranslationService.translate_dict) -- this function only decides WHAT to
    send and how to put the answer back in place, never how a single string
    gets translated.

    `mode` mirrors the rest of the app: "force" always retranslates,
    "append"/"skip" reuse a prior translation's block when available. Reuse
    is matched by POSITION AMONG TRANSLATABLE BLOCKS rather than by physical
    line index, because a "para" block reflows N source lines into a single
    translated line -- so a previously-translated file naturally has fewer
    physical lines than its English source, and line-index alignment would
    silently break on every single prose paragraph.
    """
    target_regex = target_lang["regex"]
    blocks = _split_blocks(en_text)
    tr_blocks = _split_blocks(tr_text, classify=False) if tr_text else []
    tr_translatable = [b for b in tr_blocks if b["kind"] in _TRANSLATABLE_KINDS]

    pending: dict[str, str] = {}
    final_text: dict[int, str] = {}
    translatable_idx = 0
    for i, block in enumerate(blocks):
        if block["kind"] not in _TRANSLATABLE_KINDS:
            continue
        source = block["text"]
        existing = tr_translatable[translatable_idx] if translatable_idx < len(tr_translatable) else None
        existing_text = existing["text"] if existing else ""
        translatable_idx += 1

        if mode == "force":
            pending[str(i)] = source
        elif mode == "append":
            if existing_text.strip() and existing_text != source:
                final_text[i] = existing_text
            else:
                pending[str(i)] = source
        else:  # "skip"
            if existing_text.strip() and already_translated(existing_text, target_regex):
                final_text[i] = existing_text
            else:
                pending[str(i)] = source

    if pending:
        translated = translate_batch(pending)
        for key, source in pending.items():
            final_text[int(key)] = translated.get(key, source)

    return "\n".join(
        _render_block(block, final_text.get(i))
        for i, block in enumerate(blocks)
    )


def count_guide_blocks(en_text: str, tr_text: str, target_regex: str) -> tuple[int, int]:
    """Return (translatable_count, already_translated_count) for a guide
    page, for the Analyze screen and the pre-flight cost estimate -- must
    classify lines exactly like translate_guide_markdown so those screens
    agree with what a real translation run would actually do."""
    blocks = _split_blocks(en_text)
    tr_blocks = _split_blocks(tr_text, classify=False) if tr_text else []
    tr_translatable = [b for b in tr_blocks if b["kind"] in _TRANSLATABLE_KINDS]
    translatable = [b for b in blocks if b["kind"] in _TRANSLATABLE_KINDS]

    tr_count = 0
    for i in range(len(translatable)):
        if i < len(tr_translatable):
            existing = tr_translatable[i]["text"]
            if existing.strip() and already_translated(existing, target_regex):
                tr_count += 1
    return len(translatable), tr_count


def count_book_md_blocks(text: str) -> int:
    """Total translatable blocks in a guide page -- used by the cost
    estimator, which (matching _count_book_json's existing behavior) counts
    everything in the file rather than subtracting already-translated
    content."""
    return sum(1 for b in _split_blocks(text) if b["kind"] in _TRANSLATABLE_KINDS)
