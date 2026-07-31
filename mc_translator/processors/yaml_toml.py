"""
Processor for YAML (.yaml, .yml) and TOML (.toml) files.
Extracts translatable strings, translates them, and writes back the same format.

Uses round-trip parsers (ruamel.yaml, tomlkit) rather than pyyaml/toml's
plain load+dump, which used to silently corrupt every untouched value in the
file: PyYAML's safe_load applies YAML 1.1 type coercion on *load* (so
"1.10" -> 1.1, "on"/"NO" -> bool) and safe_dump re-serializes the whole
document from that coerced Python data, discarding all comments and original
formatting in the process -- a mod config with a `# valid range: 0-100`
comment above an option came back with no comment at all, and a version
string like "1.10" became "1.1". ruamel.yaml's YAML(typ="rt") and tomlkit
preserve scalar representation, key order, and comments, and only the
specific string values this processor actually translates are changed.
"""
from __future__ import annotations

import io
import os
from typing import Any

try:
    from ruamel.yaml import YAML
    from ruamel.yaml.scalarstring import ScalarString

    _yaml = YAML()
    _yaml.preserve_quotes = True
    _yaml.allow_unicode = True
    _yaml.width = 4096  # avoid re-wrapping long translated lines
except Exception:  # pragma: no cover
    YAML = None  # type: ignore
    ScalarString = None  # type: ignore
    _yaml = None  # type: ignore

# YAML 1.1's core schema resolves these *unquoted* plain scalars to bool/null
# (yaml_implicit_resolvers in PyYAML's Resolver -- the same vocabulary that
# used to be silently coerced to True/False/None by yaml.safe_load). ruamel's
# round-trip mode deliberately keeps them as plain `str` instead (so it can
# write back the exact original spelling -- "on" vs "On" vs "true" -- rather
# than guessing), which means they're otherwise indistinguishable from
# genuine translatable text. A *quoted* "on"/"NO" (ScalarString subclass) is
# unambiguously a real string and is never skipped by this check.
_YAML_AMBIGUOUS_SCALAR_TOKENS = frozenset({
    "y", "n", "yes", "no", "true", "false", "on", "off",
    "null", "~",
})

try:
    import tomlkit
except Exception:  # pragma: no cover
    tomlkit = None  # type: ignore

from mc_translator.engines.base import EngineCallbacks
from mc_translator.engines.service import TranslationService
from mc_translator.processors.resourcepack_path import compute_target_paths
from mc_translator.text_processing import (
    already_translated,
    is_technical_term,
    looks_like_source_language,
)


def _walk_and_collect(
    data: Any,
    path: tuple[Any, ...],
    target_lang: dict,
    mode: str,
    existing_map: dict,
) -> list[tuple[tuple[Any, ...], str]]:
    """Recursively walk data, returning a flat list of (path_tuple, string)
    pairs for strings that need translation.

    Returns a LIST of real path tuples rather than a dict keyed by a
    "/"-joined path string: a joined string is ambiguous whenever a real
    dict key contains "/" (two distinct paths can stringify to the same
    key), and the old code used that ambiguous string as the *write-back*
    key too -- a collision there meant set_at_path either wrote to the wrong
    node or raised (silently swallowed by a bare except), permanently
    dropping a translation that was still translated and cached. The caller
    assigns its own collision-free keys (sequential indices) for the
    translate_dict() call and applies results back using these exact path
    tuples.
    """
    results: list[tuple[tuple[Any, ...], str]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            results.extend(_walk_and_collect(value, path + (key,), target_lang, mode, existing_map))
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            results.extend(_walk_and_collect(item, path + (idx,), target_lang, mode, existing_map))
    elif isinstance(data, str):
        s = str(data)
        if not s or not s.strip():
            return results
        is_quoted = ScalarString is not None and isinstance(data, ScalarString)
        if not is_quoted and s.strip().lower() in _YAML_AMBIGUOUS_SCALAR_TOKENS:
            return results
        if is_technical_term(s):
            return results
        if not looks_like_source_language(s):
            return results
        key_str = "/".join(map(str, path))
        existing = existing_map.get(key_str)
        if mode == "append" and existing and existing.strip() and existing != s:
            return results
        if mode == "append" and existing and existing.strip() == s:
            return results
        if mode == "skip":
            if existing and already_translated(existing, target_lang["regex"]):
                return results
        results.append((path, s))
    # Other types (int, float, bool, None) are ignored
    return results


def _get_at_path(data: Any, path_tuple: tuple[Any, ...]) -> Any:
    """Read-only traversal mirroring _set_at_path -- used to verify a stale
    existing_map path still resolves to a plain string before seeding it
    back. Without this, a field the mod restructured into a nested dict/list
    between runs got silently clobbered with the old flat string (no
    exception raised), and the real translation for whatever now lives under
    that path was lost right after."""
    cur = data
    for part in path_tuple:
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                idx = int(str(part))
            except ValueError:
                return None
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
        else:
            return None
    return cur


def _set_at_path(data: Any, path_tuple: tuple[Any, ...], value: Any) -> None:
    cur = data
    for part in path_tuple[:-1]:
        if isinstance(cur, dict):
            cur = cur[part]
        elif isinstance(cur, list):
            idx = int(str(part))
            if idx < 0 or idx >= len(cur):
                raise IndexError(f"List index {idx} out of range")
            cur = cur[idx]
        else:
            raise TypeError(f"Cannot traverse {type(cur)}")
    last = path_tuple[-1]
    if isinstance(cur, dict):
        cur[last] = value
    elif isinstance(cur, list):
        idx = int(str(last))
        if idx < 0 or idx >= len(cur):
            raise IndexError(f"List index {idx} out of range")
        cur[idx] = value
    else:
        raise TypeError(f"Cannot set on {type(cur)}")


class YamlTomlProcessor:
    def __init__(
        self,
        service: TranslationService,
        state: "JobState",
        callbacks: EngineCallbacks,
    ) -> None:
        self.service = service
        self.state = state
        self.callbacks = callbacks
        if _yaml is None:
            self.callbacks.on_log(
                "⚠️ Модуль ruamel.yaml не установлен. YAML файлы будут пропускаться.",
                "yellow",
            )
        if tomlkit is None:
            self.callbacks.on_log(
                "⚠️ Модуль tomlkit не установлен. TOML файлы будут пропускаться.",
                "yellow",
            )

    def _load(self, raw: str, ext: str) -> Any:
        if ext in {".yaml", ".yml"}:
            return _yaml.load(raw)
        return tomlkit.parse(raw)

    def _dump(self, data: Any, ext: str) -> str:
        if ext in {".yaml", ".yml"}:
            buf = io.StringIO()
            _yaml.dump(data, buf)
            return buf.getvalue()
        return tomlkit.dumps(data)

    def process(
        self,
        file_path: str,
        *,
        target_lang: dict,
        mode: str,
        output_mode: str,
        pack_writer: object | None = None,
        mc_dir: str | None = None,
    ) -> None:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in {".yaml", ".yml", ".toml"}:
            self.callbacks.on_log(
                f"⚠️ Неподдерживаемое расширение {ext} для YamlTomlProcessor",
                "red",
            )
            return

        # Load data
        try:
            with open(file_path, encoding="utf-8") as f:
                raw = f.read()
        except OSError as exc:
            self.callbacks.on_log(f"❌ Ошибка чтения {file_path}: {exc}", "red")
            return

        if ext in {".yaml", ".yml"}:
            if _yaml is None:
                self.callbacks.on_log(
                    f"❌ Невозможно прочитать YAML {file_path}: ruamel.yaml не установлен",
                    "red",
                )
                return
        elif tomlkit is None:
            self.callbacks.on_log(
                f"❌ Невозможно прочитать TOML {file_path}: tomlkit не установлен",
                "red",
            )
            return

        try:
            data = self._load(raw, ext)
        except Exception as exc:
            self.callbacks.on_log(f"❌ Ошибка парсинга {ext.upper()} {file_path}: {exc}", "red")
            return

        # Determine target path for output
        target_name, target_path, internal_path = compute_target_paths(file_path, target_lang["file"], mc_dir)

        # Load existing translation if any (for append/skip modes)
        existing_data: Any = None
        if os.path.exists(target_path):
            try:
                with open(target_path, encoding="utf-8") as f:
                    existing_raw = f.read()
                existing_data = self._load(existing_raw, ext)
            except Exception:
                existing_data = None

        # Build existing map (flattened "/"-joined path -> string, used both
        # to look up whether a given path was already translated AND to
        # reseed `data` below with prior translations. A collision here (two
        # distinct paths stringifying to the same key) at worst causes an
        # occasional redundant translation or a wrong seed value -- it can no
        # longer drop a translation outright, since write-back for freshly
        # translated strings below uses real path tuples, not this string.
        existing_map: dict = {}
        if existing_data is not None:
            def _flatten(obj: Any, cur_path: tuple[Any, ...]) -> None:
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        _flatten(v, cur_path + (str(k),))
                elif isinstance(obj, list):
                    for idx, v in enumerate(obj):
                        _flatten(v, cur_path + (idx,))
                elif isinstance(obj, str):
                    key_str = "/".join(map(str, cur_path))
                    existing_map[key_str] = str(obj)
            _flatten(existing_data, ())

        # Collect strings to translate
        pairs = _walk_and_collect(data, (), target_lang, mode, existing_map)

        # Seed data with already-translated values for paths that aren't being
        # freshly translated this run, so they survive into the output instead of
        # reverting to the English source (mirrors generic_json.py's identical fix).
        # pairs_paths is normalized to all-string tuples: _walk_and_collect's
        # real path tuples mix str (dict keys) and int (list indices), while
        # a seed path reconstructed from existing_map's "/"-joined key is
        # always str -- comparing them unnormalized meant this membership
        # check silently never matched for any list-indexed path (or an
        # unquoted-integer YAML mapping key), so a value could be queued for
        # both seeding and fresh translation at once (only order-of-
        # application happened to keep the output correct).
        pairs_paths = {tuple(str(p) for p in path) for path, _ in pairs}
        for key, value in existing_map.items():
            if not value.strip():
                continue
            path_tuple = tuple(key.split("/")) if key else ()
            if path_tuple in pairs_paths:
                continue
            # Skip if this path no longer resolves to a plain string (the mod
            # restructured a flat field into a nested dict/list between
            # runs) -- otherwise this blindly overwrites the new structure
            # with the old flat string, with no exception raised.
            if not isinstance(_get_at_path(data, path_tuple), str):
                continue
            try:
                _set_at_path(data, path_tuple, value)
            except Exception:
                pass  # existing_map's structure has diverged from data's; skip

        if not pairs:
            # Nothing to translate; just copy if needed. Prefer an existing
            # translated file over the English source, so a prior 'inplace'
            # run isn't silently overwritten with untranslated text.
            if output_mode == "resourcepack" and pack_writer:
                source_path = target_path if os.path.exists(target_path) else file_path
                try:
                    with open(source_path, "rb") as f:
                        pack_writer.write(internal_path, f.read())
                except Exception as exc:
                    self.callbacks.on_log(
                        f"❌ Ошибка записи в ресурспак {internal_path}: {exc}",
                        "red",
                    )
            elif output_mode == "inplace":
                # No change needed
                pass
            return

        to_translate = {str(i): s for i, (_path, s) in enumerate(pairs)}

        # Translate
        self.callbacks.on_log(
            f"⚡ Перевод {os.path.basename(file_path)} ({ext.upper()}) — {len(to_translate)} строк",
            "cyan",
        )
        translated = self.service.translate_dict(
            to_translate,
            target_lang,
            self.callbacks,
            context=f"{ext.upper()} файл",
            usage_label=(os.path.basename(file_path), ext.upper()),
        )
        # Apply translations back using the real path tuples collected above
        # (not by re-splitting a "/"-joined key, which is what made this
        # unsafe for keys containing "/").
        for i_str, new_val in translated.items():
            idx = int(i_str)
            path_tuple, _source = pairs[idx]
            try:
                _set_at_path(data, path_tuple, new_val)
            except Exception as exc:
                self.callbacks.on_log(
                    f"❌ Ошибка применения перевода в {file_path} (путь {path_tuple}): {exc}",
                    "red",
                )

        # Write output
        try:
            out_text = self._dump(data, ext)
        except Exception as exc:
            self.callbacks.on_log(f"❌ Ошибка сериализации {ext.upper()} {file_path}: {exc}", "red")
            return

        if output_mode == "resourcepack" and pack_writer:
            try:
                pack_writer.write(internal_path, out_text.encode("utf-8"))
            except Exception as exc:
                self.callbacks.on_log(
                    f"❌ Ошибка записи в ресурспак {internal_path}: {exc}",
                    "red",
                )
        elif output_mode == "inplace":
            try:
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(out_text)
            except OSError as exc:
                self.callbacks.on_log(
                    f"❌ Ошибка записи {target_path}: {exc}", "red"
                )
