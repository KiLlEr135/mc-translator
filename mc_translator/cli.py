"""
Headless CLI entry point for MC Translator.

TranslationJob/TranslationOptions are GUI-agnostic (all output goes through
injected on_log/on_status/on_row callbacks), so this just supplies console
versions of those callbacks instead of Tkinter widgets. Used by
`python -m mc_translator <args>` and `python translator.py <args>` when arguments
are given; with no arguments both fall back to the GUI (see __main__.py).
"""
from __future__ import annotations

import argparse
import sys

from mc_translator.cache import load_both_caches
from mc_translator.config import settings
from mc_translator.constants import LANGUAGES, MC_VERSIONS
from mc_translator.runtime.job import TranslationJob, TranslationOptions
from mc_translator.runtime.state import JobState


def _console_log(message: str, tag: str = "white") -> None:
    print(message)


def _console_status(message: str, progress: float | None) -> None:
    if progress is not None:
        print(f"[{int(progress * 100):3d}%] {message}")
    else:
        print(message)


def _console_row(icon: str, name: str, kind: str, trans_c: int, en_c: int, pct: int) -> None:
    print(f"{icon} {name[:34]:<35}[{kind}]".ljust(60) + f"{trans_c}/{en_c} ({pct}%)")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mc_translator",
        description=(
            "MC Translator — headless CLI mode. "
            "Run with no arguments to launch the GUI instead."
        ),
    )
    parser.add_argument("--mc-dir", required=True, help="Path to the Minecraft instance directory")
    parser.add_argument("--lang", required=True, choices=list(LANGUAGES.keys()), help="Target language")
    parser.add_argument("--mc-version", default="1.20.1", choices=MC_VERSIONS, help="Minecraft version (pack format)")
    parser.add_argument("--engine", default="google", choices=["google", "deepl", "ai"], help="Translation engine")
    parser.add_argument("--mode", default="append", choices=["append", "skip", "force"], help="Processing mode")
    parser.add_argument("--output", default="resourcepack", choices=["resourcepack", "inplace"], help="Output mode")
    parser.add_argument("--pack-name", default="MCTranslator_Pack", help="Resourcepack/datapack zip base name")
    parser.add_argument("--analyze-only", action="store_true", help="Only run analysis, don't translate")
    parser.add_argument("--no-mods", action="store_false", dest="translate_mods", help="Skip mod interface strings")
    parser.add_argument("--no-books", action="store_false", dest="translate_books", help="Skip guide/quest books")
    parser.add_argument("--no-quests", action="store_false", dest="translate_quests", help="Skip FTB/KubeJS quests")
    parser.set_defaults(translate_mods=True, translate_books=True, translate_quests=True)
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    # Console output (help text, Russian log lines, language names with
    # CJK/Cyrillic characters) can otherwise crash with UnicodeEncodeError
    # on a Windows console using a legacy codepage (e.g. cp1251) instead of UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    settings.set("GENERAL", "mc_dir", args.mc_dir)

    state = JobState()
    state.is_running = True
    cache_std, cache_ai, polish_total = load_both_caches()
    if polish_total:
        print(f"✨ Кэш отполирован: исправлено {polish_total} строк.")

    job = TranslationJob(
        settings,
        cache_std,
        cache_ai,
        state,
        on_log=_console_log,
        on_status=_console_status,
        on_row=_console_row,
    )
    options = TranslationOptions(
        mc_dir=args.mc_dir,
        language_label=args.lang,
        mc_version=args.mc_version,
        output_mode=args.output,
        pack_name=args.pack_name,
        engine=args.engine,
        google_mode="single",
        ai_mode="safe",
        ai_provider=settings.get("AI", "ai_provider") or "local",
        global_context=settings.get("AI", "global_context"),
        process_mode=args.mode,
        translate_mods=args.translate_mods,
        translate_books=args.translate_books,
        translate_quests=args.translate_quests,
    )

    try:
        if args.analyze_only:
            job.run_analysis(options)
        else:
            job.run_translation(options)
    finally:
        state.is_running = False
    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
