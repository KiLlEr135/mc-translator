# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all

# ruamel.yaml is a namespace package with lazily-imported submodules
# (constructor/representer/resolver/comments) that PyInstaller's static
# analysis routinely misses. Without this, the frozen exe doesn't crash --
# yaml_toml.py catches the ImportError and silently disables YAML
# translation instead, which is much harder to notice than a startup error.
_ruamel_datas, _ruamel_binaries, _ruamel_hiddenimports = collect_all('ruamel.yaml')

a = Analysis(
    ['mc_translator\\__main__.py'],
    pathex=[],
    binaries=_ruamel_binaries,
    datas=[('mc_translator/gui/web', 'web'), ('icon.ico', '.')] + _ruamel_datas,
    hiddenimports=_ruamel_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6', 'PySide2', 'PySide6'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MC_Translator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico'],
)
