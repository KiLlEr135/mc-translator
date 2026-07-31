"""Resolves the application's persistent-data directory.

Every persistent file (settings.ini, cache.json, ai_cache.json,
dictionary.json, logs/) used to be a bare relative filename/dirname, which
Python resolves against the process's current working directory. For a
`python -m mc_translator` invocation from the project root that happens to
work (CWD == project root) -- but a PyInstaller --onefile build launched
with a different working directory (a desktop shortcut with a custom
"Start in", launched via PATH, "Open with a file", etc.) would read/write
these files in that unrelated CWD instead of next to the .exe, silently
starting from an empty config and an empty cache every time, contradicting
build.bat's documented instruction to place settings.ini/dictionary.json/
cache.json next to the built .exe.

This mirrors gui/app.py's existing sys._MEIPASS handling for bundled
read-only assets (web/, icon.ico), but for the *writable* files that must
live next to the executable, not in PyInstaller's temp extraction dir.
"""
from __future__ import annotations

import os
import sys


def app_base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    # Running from source: mc_translator/utils/app_paths.py -> project root
    # is two directories up.
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def app_path(*parts: str) -> str:
    return os.path.join(app_base_dir(), *parts)
