#!/usr/bin/env python3
"""Manual smoke test (not a pytest test): opens a minimal pywebview window
to verify the pywebview install/runtime works on this machine, independent
of the full MC Translator GUI. Run manually: `python scripts/pywebview_smoketest.py`.
"""
import webview


class Api:
    def ping(self):
        return "pong"


api = Api()
window = webview.create_window("Smoke Test", html="<h1>Hello pywebview</h1>", js_api=api, width=400, height=300)
webview.start()
