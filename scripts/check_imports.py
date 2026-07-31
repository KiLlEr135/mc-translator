#!/usr/bin/env python3
"""Diagnostic script (not a pytest test): import every mc_translator module
and report any that fail. Useful after refactors/renames or before cutting
a build, to catch a broken import before it surfaces as a runtime crash.

Run from anywhere: `python scripts/check_imports.py`.
"""
import os
import sys
import traceback

# Not installed as a package (no pyproject.toml/setup.py), so mc_translator
# is only importable when the repo root is on sys.path -- add it explicitly
# so this script works regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def try_import(module_name):
    try:
        __import__(module_name)
        print(f"[+] {module_name}")
        return True
    except Exception as e:
        print(f"[-] {module_name}: {e}")
        traceback.print_exc()
        return False


modules = [
    "mc_translator.config",
    "mc_translator.constants",
    "mc_translator.cache",
    "mc_translator.usage_log",
    "mc_translator.text_processing",
    "mc_translator.json_utils",
    "mc_translator.mod_names",
    "mc_translator.engines.base",
    "mc_translator.engines.service",
    "mc_translator.engines.deepl",
    "mc_translator.engines.google",
    "mc_translator.engines.kobold",
    "mc_translator.engines.llm_common",
    "mc_translator.engines.openrouter",
    "mc_translator.processors.analyzer",
    "mc_translator.processors.archive",
    "mc_translator.processors.bat",
    "mc_translator.processors.cfg",
    "mc_translator.processors.discovery",
    "mc_translator.processors.estimator",
    "mc_translator.processors.generic_json",
    "mc_translator.processors.jar",
    "mc_translator.processors.lang",
    "mc_translator.processors.locale_keys",
    "mc_translator.processors.loose_json",
    "mc_translator.processors.mcfunction",
    "mc_translator.processors.nbt",
    "mc_translator.processors.snbt",
    "mc_translator.processors.snbt_extract",
    "mc_translator.processors.text",
    "mc_translator.processors.xml",
    "mc_translator.processors.yaml_toml",
    "mc_translator.output.pack_writer",
    "mc_translator.runtime.ai_launcher",
    "mc_translator.runtime.job",
    "mc_translator.runtime.state",
    "mc_translator.utils.version_detector",
    "mc_translator.gui.bridge",
    "mc_translator.gui.app",
]

all_ok = True
for m in modules:
    if not try_import(m):
        all_ok = False

if all_ok:
    print("\nAll imports succeeded.")
else:
    print("\nSome imports failed.")
    sys.exit(1)
