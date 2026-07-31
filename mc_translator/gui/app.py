"""
Entry point for the pywebview-based GUI. All actual UI logic lives in
mc_translator/gui/web/ (HTML/CSS/JS) and mc_translator/gui/bridge.py (the
Python<->JS bridge) — this module just creates the window.
"""
from __future__ import annotations

import os
import sys

import webview

from mc_translator import __version__
from mc_translator.gui.bridge import Api


def _web_dir() -> str:
    """Resolve mc_translator/gui/web/ whether running from source or a
    PyInstaller --onefile build (where files are extracted to sys._MEIPASS)."""
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "web")


def _icon_path() -> str | None:
    """icon.ico lives at the project root when running from source, and at
    the sys._MEIPASS root when frozen (see build.bat's --add-data)."""
    base = getattr(sys, "_MEIPASS", None) or os.getcwd()
    path = os.path.join(base, "icon.ico")
    return path if os.path.exists(path) else None


def run() -> None:
    api = Api()
    index_path = os.path.join(_web_dir(), "index.html")

    window = webview.create_window(
        f"MC Translator v{__version__}",
        url=index_path,
        js_api=api,
        width=1150,
        height=850,
        min_size=(900, 650),
        background_color="#16151a",
    )
    api.set_window(window)
    webview.start(icon=_icon_path())
