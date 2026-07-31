"""
Processor for NBT files (.nbt, .dat).
Uses nbtlib to load, translate string tags, and save back.
"""
from __future__ import annotations

import os
from typing import Any

try:
    import nbtlib
    from nbtlib import tag
except Exception:  # pragma: no cover
    nbtlib = None  # type: ignore
    tag = None  # type: ignore

from mc_translator.engines.base import EngineCallbacks
from mc_translator.engines.service import TranslationService
from mc_translator.processors.resourcepack_path import compute_target_paths
from mc_translator.text_processing import (
    already_translated,
    is_technical_term,
    looks_like_source_language,
)


def _is_under_saves(file_path: str, mc_dir: str) -> bool:
    """True if `file_path` lives under mc_dir's "saves" root -- the world
    save tree (discover_mod_content_files' "saves" bucket), as opposed to
    "config"/"datapacks" (also NBT sources, but not provably dead in
    resourcepack mode)."""
    try:
        rel = os.path.relpath(file_path, mc_dir).replace("\\", "/")
    except ValueError:
        return False
    return rel.split("/", 1)[0].lower() == "saves"


def _walk_nbt(
    nbt_obj: Any,
    path: tuple[Any, ...],
    target_lang: dict,
    mode: str,
    existing_map: dict[str, str],
) -> list[tuple[tuple[Any, ...], str]]:
    """Recursively walk NBT structure, returning a flat list of (path_tuple,
    string) pairs to translate.

    Returns real path tuples rather than a dict keyed by a "/"-joined path
    string: an NBT compound tag's key is an arbitrary string and can itself
    contain "/", which would make the joined string ambiguous. The old code
    used that ambiguous string as the *write-back* key too, so a collision
    could silently drop an already-translated (and already cached/paid-for)
    string. The caller assigns its own collision-free keys for the
    translate_dict() call and applies results back using these exact tuples.
    """
    results: list[tuple[tuple[Any, ...], str]] = []
    if nbtlib is None or tag is None:
        return results
    # NOTE: nbtlib's tag classes are named Base/Compound/List/String (no
    # "Tag" suffix, no separate "Tag" base) -- tag.Tag/CompoundTag/ListTag/
    # StringTag never existed on the installed nbtlib>=2.0.4 (see
    # requirements.txt), so every isinstance() check below used to raise
    # AttributeError on the very first NBT file processed. tag.String(...)
    # below (the write-back side) already used the correct name, which is
    # what gave this away.
    if isinstance(nbt_obj, tag.Base):
        if isinstance(nbt_obj, tag.Compound):
            for key, val in nbt_obj.items():
                results.extend(_walk_nbt(val, path + (key,), target_lang, mode, existing_map))
        elif isinstance(nbt_obj, tag.List):
            for idx, val in enumerate(nbt_obj):
                results.extend(_walk_nbt(val, path + (idx,), target_lang, mode, existing_map))
        elif isinstance(nbt_obj, tag.String):
            s = str(nbt_obj)
            if not s or not s.strip():
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
        # Other tag types (numeric, byte arrays, etc.) are ignored.
    return results


def _get_nbt_value(nbt_obj: Any, path_tuple: tuple[Any, ...]) -> Any:
    """Read-only traversal mirroring _apply_nbt_translations._set -- used to
    verify a stale existing_map path still resolves to a String leaf before
    seeding it back. Without this, a field the mod turned into a nested
    Compound/List between runs got silently clobbered with the old flat
    string (no exception raised), and the real translation for whatever now
    lives under that path was lost right after."""
    cur = nbt_obj
    for part in path_tuple:
        if isinstance(cur, tag.Compound):
            if part not in cur:
                return None
            cur = cur[part]
        elif isinstance(cur, tag.List):
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


def _apply_nbt_translations(nbt_obj: Any, translations: list[tuple[tuple[Any, ...], str]]) -> Any:
    """Apply translations to NBT structure based on real path tuples."""
    if not translations:
        return nbt_obj

    def _set(obj: Any, path_tuple: tuple[Any, ...], value: Any) -> Any:
        if not path_tuple:
            return tag.String(value)
        cur = obj
        for part in path_tuple[:-1]:
            if isinstance(cur, tag.Compound):
                cur = cur[part]
            elif isinstance(cur, tag.List):
                idx = int(str(part))
                if idx < 0 or idx >= len(cur):
                    raise IndexError(f"List index {idx} out of range")
                cur = cur[idx]
            else:
                raise TypeError(f"Cannot traverse {type(cur)}")
        last = path_tuple[-1]
        if isinstance(cur, tag.Compound):
            cur[last] = tag.String(value)
        elif isinstance(cur, tag.List):
            idx = int(str(last))
            if idx < 0 or idx >= len(cur):
                raise IndexError(f"List index {idx} out of range")
            cur[idx] = tag.String(value)
        else:
            raise TypeError(f"Cannot set on {type(cur)}")
        return obj

    for path_tuple, value in translations:
        if not path_tuple:
            continue
        try:
            _set(nbt_obj, path_tuple, value)
        except Exception:
            # Skip if cannot set
            pass
    return nbt_obj


class NbtProcessor:
    def __init__(
        self,
        service: TranslationService,
        state: "JobState",
        callbacks: EngineCallbacks,
    ) -> None:
        self.service = service
        self.state = state
        self.callbacks = callbacks
        if nbtlib is None:
            self.callbacks.on_log(
                "⚠️ Модуль nbtlib не установлен. NBT файлы будут пропускаться.",
                "yellow",
            )

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
        """
        Process a single NBT file.

        Args:
            file_path: path to source .nbt or .dat file.
            target_lang: language dict with 'file' (lang code) and 'regex' for detection.
            mode: translation mode ('append', 'skip', 'force').
            output_mode: output mode ('resourcepack' or 'inplace').
            pack_writer: PackWriter if output_mode is 'resourcepack'.
        """
        if nbtlib is None:
            self.callbacks.on_log(
                f"❌ Невозможно обработать NBT {file_path}: nbtlib не установлен",
                "red",
            )
            return

        if output_mode == "resourcepack" and mc_dir and _is_under_saves(file_path, mc_dir):
            # Resource packs only ever deliver content under "assets/" --
            # anything written for a "saves/<world>/..." NBT file (world
            # save data: level.dat, per-save mod persistent data, etc.) is
            # never read by the game from inside a resourcepack zip, so
            # translating it here would just be wasted API calls/cache
            # space/zip bloat with no way for it to ever be seen. NBT under
            # "config/" or "datapacks/" is unaffected -- datapack NBT is
            # routed to the real datapack zip by PackWriter, and this only
            # needs to special-case the one root that's provably dead in
            # this output mode.
            self.callbacks.on_log(
                f"⚠️ NBT в saves/ ({os.path.basename(file_path)}) пропущен в режиме ресурспака "
                f"-- сохранения недоступны ресурспаку, перевод не будет виден в игре.",
                "yellow",
            )
            return

        try:
            # nbtlib.load() (not File.load(fileobj), which requires an
            # explicit `gzipped` argument and errors out otherwise -- see
            # nbtlib's own File.load signature) opens the path itself and
            # auto-detects gzip compression from the magic number, which
            # matters here since real Minecraft NBT files are commonly
            # gzip-compressed.
            nbt_data = nbtlib.load(file_path)
        except Exception as exc:
            self.callbacks.on_log(
                f"❌ Ошибка чтения NBT {file_path}: {exc}",
                "red",
            )
            return

        # Determine target path for output
        target_name, target_path, internal_path = compute_target_paths(file_path, target_lang["file"], mc_dir)

        # Load existing translation if any (for append/skip modes)
        existing_data = None
        if os.path.exists(target_path):
            try:
                existing_data = nbtlib.load(target_path)
            except Exception:
                existing_data = None

        existing_map: dict[str, str] = {}
        if existing_data is not None:
            # Flatten existing data to map paths to strings
            def _flatten(obj: Any, cur_path: tuple[Any, ...]) -> None:
                if isinstance(obj, tag.Base):
                    if isinstance(obj, tag.Compound):
                        for k, v in obj.items():
                            _flatten(v, cur_path + (k,))
                    elif isinstance(obj, tag.List):
                        for idx, v in enumerate(obj):
                            _flatten(v, cur_path + (idx,))
                    elif isinstance(obj, tag.String):
                        key_str = "/".join(map(str, cur_path))
                        existing_map[key_str] = str(obj)
                # else ignore other tag types
            try:
                _flatten(existing_data, ())
            except RecursionError:
                existing_map = {}

        # Collect strings to translate
        try:
            pairs = _walk_nbt(nbt_data, (), target_lang, mode, existing_map)
        except RecursionError:
            self.callbacks.on_log(
                f"❌ Ошибка чтения NBT {file_path}: структура слишком глубоко вложена",
                "red",
            )
            return

        if not pairs:
            # Nothing to translate; copy if needed. Prefer an existing
            # translated file over the English source, so a prior 'inplace'
            # run isn't silently overwritten with untranslated text.
            if output_mode == "resourcepack" and pack_writer:
                source_path = target_path if os.path.exists(target_path) else file_path
                with open(source_path, "rb") as f:
                    pack_writer.write(internal_path, f.read())
            elif output_mode == "inplace":
                # No change needed
                pass
            return

        to_translate = {str(i): s for i, (_path, s) in enumerate(pairs)}

        # Translate
        self.callbacks.on_log(
            f"⚡ Перевод NBT {os.path.basename(file_path)} — {len(to_translate)} строк",
            "cyan",
        )
        translated = self.service.translate_dict(
            to_translate,
            target_lang,
            self.callbacks,
            context="NBT файл",
            usage_label=(os.path.basename(file_path), "NBT"),
        )
        # Apply translations back using the real path tuples collected above.
        resolved = [(pairs[int(i)][0], value) for i, value in translated.items()]

        # Seed with already-translated values for paths that aren't being
        # freshly translated this run -- mirrors generic_json.py's identical
        # fix (see its comment at line 72): without this, append mode
        # silently reverted already-translated NBT strings back to English
        # on every subsequent run, since only this run's `resolved` values
        # ever got applied to `nbt_data` (the English source read from disk
        # above), and existing_map was otherwise only used inside _walk_nbt
        # to decide what to skip -- never merged back in.
        # pairs_paths is normalized to all-string tuples: _walk_nbt's real
        # path tuples mix str (Compound keys) and int (List indices), while a
        # seed path reconstructed from existing_map's "/"-joined key is
        # always str -- comparing them unnormalized meant this membership
        # check silently never matched for any List-indexed path, so a value
        # could be queued into both `seed` and `resolved` at once (only
        # order-of-application happened to keep the output correct).
        pairs_paths = {tuple(str(p) for p in path) for path, _ in pairs}
        seed = []
        for key_str, value in existing_map.items():
            if not value.strip():
                continue
            path_tuple = tuple(key_str.split("/")) if key_str else ()
            if path_tuple in pairs_paths:
                continue
            # Skip if this path no longer resolves to a String leaf (the mod
            # restructured a flat field into a Compound/List between runs) --
            # otherwise this blindly overwrites the new nested structure with
            # the old flat string, with no exception raised.
            if not isinstance(_get_nbt_value(nbt_data, path_tuple), tag.String):
                continue
            seed.append((path_tuple, value))
        updated_nbt = _apply_nbt_translations(nbt_data, seed + resolved)

        # Write output
        if output_mode == "resourcepack" and pack_writer:
            try:
                # File.save(filename) expects a path (it does its own
                # open()); it can't take a BytesIO. File.write(fileobj) is
                # the fileobj-based counterpart, but it always writes raw
                # (uncompressed) NBT -- gzip it ourselves first if the
                # source was gzipped, so the in-memory bytes match what
                # File.save(target_path) below would have produced on disk.
                import gzip
                from io import BytesIO
                buffer = BytesIO()
                if updated_nbt.gzipped:
                    with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
                        updated_nbt.write(gz)
                else:
                    updated_nbt.write(buffer)
                pack_writer.write(internal_path, buffer.getvalue())
            except Exception as exc:
                self.callbacks.on_log(
                    f"❌ Ошибка записи NBT в ресурспак {target_name}: {exc}",
                    "red",
                )
        elif output_mode == "inplace":
            try:
                # File.save(filename, ...) takes a path, not an open file
                # object -- it opens (and gzips, per self.gzipped) the file
                # itself. The old "with open(target_path, 'wb') as f:
                # updated_nbt.save(f)" passed a BufferedWriter where a path
                # was expected and crashed on every write.
                updated_nbt.save(target_path)
            except Exception as exc:
                self.callbacks.on_log(
                    f"❌ Ошибка записи NBT {target_path}: {exc}",
                    "red",
                )