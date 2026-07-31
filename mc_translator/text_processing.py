import json
import re
from functools import lru_cache

from mc_translator.constants import DICT_FILE, IGNORE_TERMS

FORMAT_PATTERN = re.compile(
    r"("
    r"\$\([^)]+\)|"
    r"[&§][0-9a-fk-orlmn]|"
    r"</?[a-zA-Z][^>]*>|"
    r"\{[a-zA-Z0-9_.]+\}|"
    r"\]\([^)]+\)|"
    r"!\[[^\]]*\]|"
    r"\[[a-z0-9_.-]+:[a-z0-9_./-]+\]|"
    r"\([a-z0-9_.-]+:[a-z0-9_./-]+\)|"
    r"\([A-Za-z0-9_./-]+\.md[#a-zA-Z0-9_-]*\)|"
    r"\n|"
    r"%[0-9.,]*\$?[a-zA-Z%]"
    r")",
    flags=re.IGNORECASE,
)

IGNORE_PATTERN = re.compile(
    r"(?<![a-zA-Z])(" + "|".join(re.escape(t) for t in IGNORE_TERMS) + r")(?![a-zA-Z])"
)

CJK_PATTERN = re.compile(r"[一-鿿぀-ヿ가-힯]")


def apply_smart_glue(text: str) -> str:
    if not text:
        return text
    return re.sub(
        r"(?<![.!?>\]:])\s*(?:\\n|\r?\n)\s*(?!(?:[\r\n\-*#<]|$|---|[\w\s]+:))",
        " ",
        text,
    )


def load_dictionary() -> dict[str, str]:
    if not __import__("os").path.exists(DICT_FILE):
        default = {"полуслой": "плита", "сыромятная медь": "сырая медь"}
        with open(DICT_FILE, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=4)
        return default
    try:
        with open(DICT_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


TERMINOLOGY_FIXES = load_dictionary()


def polish_translation(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    text = re.sub(r"([&§][0-9a-fk-or])\s+", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([&§][r])", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s+(%\d*\$?[sd])\s+\]", r"[\1]", text)
    text = re.sub(r"\(\s+(%\d*\$?[sd])\s+\)", r"(\1)", text)
    text = re.sub(r'\"\s+(%\d*\$?[sd])\s+\"', r'"\1"', text)
    text = re.sub(r"%\s+([sd])\b", r"%\1", text)
    text = re.sub(r"%\s+(\d+)\s*\$\s*([sd])", r"%\1$\2", text)
    text = re.sub(r"%\s*\.\s*(\d+)\s*([fd])", r"%.\1\2", text)
    text = re.sub(r"\]\s+\(", "](", text)
    text = re.sub(r"!\s+\[", "![", text)
    text = re.sub(r"\[\s+", "[", text)
    text = re.sub(r"\s+\]", "]", text)
    text = re.sub(r" {2,}", " ", text)

    for wrong, right in TERMINOLOGY_FIXES.items():
        # Idempotency guard: a rule whose replacement CONTAINS the search
        # term (e.g. "медь" -> "сырая медь") is not safe to apply blindly --
        # cache.py's load_and_polish re-runs polish_translation over every
        # cached value on EVERY program start, so a correct cached value
        # would otherwise re-match its own prior output and grow forever:
        # "сырая медь" -> "сырая сырая медь" -> "сырая сырая сырая медь" -> ...
        # Precompute where `right` already occurs and skip any `wrong` match
        # that falls inside one of those spans -- it's already correct.
        right_spans = [m.span() for m in re.finditer(re.escape(right), text, flags=re.IGNORECASE)]

        def repl(match, r=right, spans=right_spans):
            if any(start < match.end() and end > match.start() for start, end in spans):
                return match.group(0)
            word = match.group(0)
            if word.istitle():
                return r.capitalize()
            if word.isupper():
                return r.upper()
            return r

        # \b word-boundary anchors never match inside unbroken CJK text
        # (no whitespace between "words"), so skip them for CJK terms.
        pattern = re.escape(wrong) if CJK_PATTERN.search(wrong) else r"\b" + re.escape(wrong) + r"\b"
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def mask_protected_fragments(text: str) -> tuple[str, dict[str, str]]:
    """Replace format codes and protected terms with [#n#] placeholders.

    Adjacent matches (nothing but whitespace between them) are coalesced
    into a single combined placeholder before masking. Without this, a run
    of several stacked Minecraft format codes (e.g. "&6&l&n") became that
    many separate back-to-back tokens -- "[#0#][#1#][#2#]" -- with no real
    prose between them; an LLM asked to translate the surrounding text
    routinely "tidies up" what looks like redundant bracket noise (merging,
    reordering, or dropping one), and unmask_translation_strict's
    all-or-nothing restore then discards the WHOLE otherwise-correct
    translation over a single lost token. Live-verified as a real,
    provider-independent cause of specific strings staying untranslated
    forever (the same content failed identically across OpenRouter, a
    custom API, and local KoboldCPP). Coalescing cuts a run of N shields
    down to 1, removing the compounding all-or-nothing failure at its root
    without touching the strict restore contract itself."""
    spans = [m.span() for m in FORMAT_PATTERN.finditer(text)]
    for m in IGNORE_PATTERN.finditer(text):
        # Skip a match that overlaps one FORMAT_PATTERN already claimed --
        # format codes and ignore-terms don't overlap by construction, but
        # this guards against it rather than assuming it.
        if not any(s < m.end() and e > m.start() for s, e in spans):
            spans.append(m.span())
    spans.sort()

    coalesced: list[list[int]] = []
    for start, end in spans:
        if coalesced and not text[coalesced[-1][1] : start].strip():
            coalesced[-1][1] = end
        else:
            coalesced.append([start, end])

    mapping: dict[str, str] = {}
    parts = []
    last_end = 0
    for start, end in coalesced:
        parts.append(text[last_end:start])
        token = f"[#{len(mapping)}#]"
        mapping[token] = text[start:end]
        parts.append(token)
        last_end = end
    parts.append(text[last_end:])

    masked = re.sub(r"\s+", " ", "".join(parts)).strip()
    return masked, mapping


_PLACEHOLDER_TOKEN_RE = re.compile(r"\[#\d+#\]")


def is_only_placeholders(masked: str) -> bool:
    """True if `masked` (mask_protected_fragments' output) has nothing left
    but shield placeholder tokens and whitespace -- i.e. the original string
    had NO real translatable prose at all once its format codes/protected
    terms were removed (e.g. "§f%s", a bare color code immediately
    followed by a placeholder, masks down to just "[#0#]"). Sending such a
    string to an AI engine anyway is pure wasted work AND a real failure
    risk: live-verified for real that a masked "[#0#]" with nothing else
    came back from a local model as a bare "#0#" with the brackets
    stripped -- a small/quantized model has no real sentence to anchor
    "don't touch this" formatting around when the entire prompt IS the
    placeholder, and is more likely to "simplify" it away. Callers should
    treat this the same as mask_protected_fragments returning an empty
    string: skip translation, use the original text as-is."""
    return not _PLACEHOLDER_TOKEN_RE.sub("", masked).strip()


# Weak/CJK-target translators commonly rewrite ASCII [ ] around a shield
# placeholder into a full-width lookalike bracket (a routine typographic
# substitution next to CJK text). Tolerate the common variants so a
# perfectly fine CJK translation isn't needlessly discarded by the
# has_unresolved_placeholders() check below.
_OPEN_BRACKETS = r"\[［【〔"
_CLOSE_BRACKETS = r"\]］】〕"

# The placeholder "core" (#<digits>#) essentially never occurs in real
# prose, so its bare presence -- regardless of which bracket variant (if
# any) survives around it -- reliably means unmask_translation could not
# restore that token.
_UNRESOLVED_PLACEHOLDER_RE = re.compile(r"#\s*\d+\s*#")


def _unmask(text: str, mapping: dict[str, str]) -> tuple[str, bool]:
    """Shared core for unmask_translation/unmask_translation_strict below --
    keeping the bracket-tolerant substitution in one place so the two
    public functions can't drift apart. Returns (restored_text, all_found),
    where all_found is False if any mapping token could not be located
    anywhere in `text` (in ASCII or a tolerated full-width bracket variant)."""
    all_found = True
    for token, original in mapping.items():
        idx = token.strip("#[]")
        pattern = rf"[{_OPEN_BRACKETS}]\s*#\s*{re.escape(idx)}\s*#\s*[{_CLOSE_BRACKETS}]"
        text, count = re.subn(pattern, lambda _m, o=original: o, text)
        if count == 0:
            all_found = False
    return text, all_found


def unmask_translation(text: str, mapping: dict[str, str]) -> str:
    restored, _all_found = _unmask(text, mapping)
    return restored


def has_unresolved_placeholders(text: str) -> bool:
    """True if a [#n#] shield placeholder (or a recognizable mangled
    remnant of one) survived unmask_translation -- meaning the translator
    damaged the token badly enough that the restore couldn't find/replace
    it. See unmask_translation_strict, which also catches the case where a
    placeholder was dropped WITHOUT leaving any "#n#"-shaped trace (this
    function alone can't detect that from the output text)."""
    return bool(_UNRESOLVED_PLACEHOLDER_RE.search(text))


def unmask_translation_strict(text: str, mapping: dict[str, str]) -> str | None:
    """Like unmask_translation, but returns None instead of a partially
    restored string if ANY placeholder from `mapping` could not be located
    in `text` at all (in ASCII or a tolerated full-width bracket variant).
    Covers two real translator failure modes has_unresolved_placeholders
    alone can't: the token being mangled beyond recognition (e.g. into a
    bracket style we don't tolerate), and the token being dropped entirely
    with no trace left in the output. Either way a protected format code/
    tag was silently lost -- callers must treat this as a failed
    translation (drop to the English original, don't cache it) rather than
    ship a corrupted or incomplete result into the mod."""
    restored, all_found = _unmask(text, mapping)
    if not all_found or has_unresolved_placeholders(restored):
        return None
    return restored


@lru_cache(maxsize=10000)
def is_technical_term(text: str) -> bool:
    if not text:
        return True
    lower = text.lower()
    if not re.search(r"[a-z]", lower):
        return True
    if re.match(r"^#[0-9a-f]{3,8}$", lower):
        return True
    if re.match(r"^[a-z0-9_.-]+:[a-z0-9_./-]+(?:[;,|]\s*[a-z0-9_.-]+:[a-z0-9_./-]+)*$", lower):
        return True
    # A chain of 3+ bare-colon-separated identifier segments with no
    # whitespace anywhere (e.g. a scoreboard/objective id like
    # "infoDisplay:time:1:15:-1") -- distinct from the single namespace:path
    # (or ;/,/|-repeated) shape above, which requires exactly one colon per
    # segment. Real prose never has 3+ colons with zero spaces between them,
    # so this is safe: live-verified this exact string was queued for AI
    # translation and came back empty, wasting a call on a pure identifier.
    if re.match(r"^[a-z0-9_.-]+(?::[a-z0-9_.-]+){2,}$", lower):
        return True
    # FancyMenu-style config line "<key> = [namespace:value]..." -- the
    # right-hand side of "=" is a resource/asset reference (often followed
    # by a "/"-separated path), never real prose. Real example live-verified
    # in a run's log: "source = [source:local]/config/fancymenu/assets/gui/
    # inventory/side_button/textures/..." was queued for translation, the AI
    # mangled the shielded reference, and the whole line was silently
    # discarded as "untranslated" every single run.
    if re.match(r"^\s*[a-z][a-z0-9_]*\s*=\s*\[[a-z0-9_.-]+:", lower):
        return True
    if re.match(r"^[a-z0-9_.-]+$", lower) and any(c in lower for c in "._"):
        return True
    prefixes = (
        "glyph_", "ritual_", "familiar_", "source_", "mana_", "spell_",
        "effect_", "rune_", "altar_", "botania_", "create_", "kubejs_",
    )
    return any(lower.startswith(p) for p in prefixes)


def is_translation_key(text: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9_-]+[.:][a-zA-Z0-9_.-]+$", text.strip()))


# Unicode script ranges that never legitimately appear in any of this tool's
# supported source/target languages (English source; Cyrillic, Latin-with-
# diacritics, CJK, Kana, and Hangul targets in constants.LANGUAGES). A free/
# low-quality OpenRouter model occasionally answers in a mixed or entirely
# wrong script (observed for real: a Russian-target translation that came
# back with a stray Arabic word mid-sentence) -- FOREIGN_SCRIPT_RE lets
# callers treat that as a failed translation instead of caching/shipping it.
FOREIGN_SCRIPT_RE = re.compile(
    r"["
    r"԰-֏"  # Armenian
    r"֐-׿"  # Hebrew
    r"؀-ۿ"  # Arabic
    r"܀-ݏ"  # Syriac
    r"ހ-޿"  # Thaana
    r"ऀ-෿"  # Devanagari..Sinhala (Indic scripts)
    r"฀-๿"  # Thai
    r"Ⴀ-ჿ"  # Georgian
    r"ሀ-፿"  # Ethiopic
    r"]"
)


# CJK ideographs, Kana, and Hangul are deliberately absent from
# FOREIGN_SCRIPT_RE above -- they're this tool's own zh_cn/ja_jp/ko_kr
# TARGET languages, not universally-foreign scripts. But that made them
# invisible contamination for every OTHER target: live-observed for real, a
# Russian-target translation came back partly in Chinese characters and
# passed this check silently (CJK bypassed FOREIGN_SCRIPT_RE, and nothing
# else validates a translation's script against the actual target). Only
# flag a CJK/Kana/Hangul character as contamination when the current
# target_lang's own regex doesn't also expect that script. Reuses
# CJK_PATTERN (defined above for polish_translation's word-boundary logic)
# rather than a second copy of the same ranges.
def has_foreign_script_contamination(text: str, target_lang: dict | None = None) -> bool:
    if FOREIGN_SCRIPT_RE.search(text):
        return True
    target_regex = target_lang.get("regex") if target_lang else None
    for char in CJK_PATTERN.findall(text):
        if not (target_regex and re.search(target_regex, char)):
            return True
    return False


def looks_like_source_language(text: str) -> bool:
    return bool(re.search(r"[a-zA-Z]", text))


def already_translated(text: str, target_regex: str) -> bool:
    return bool(re.search(target_regex, text))


def is_displayable_source(text: str) -> bool:
    """True for real translatable prose worth showing/editing in the GUI (the
    live log's per-string lines and the review screen) -- False for
    translation keys, technical terms, and non-text (numbers, bare symbols,
    IDs). Display-only filter: strings that fail this are still translated
    and cached exactly as before, they just aren't surfaced in the GUI."""
    if not text or not text.strip():
        return False
    if not looks_like_source_language(text):
        return False
    return not (is_translation_key(text) or is_technical_term(text))
