"""Shared helper for computing a collision-safe resourcepack internal path.

PackWriter.write(internal_path, data) is a flat map keyed by exact
internal_path -- writing the same path twice silently drops the second write
(see PackWriter.write's `if internal_path not in self.written` guard). Several
loose-file processors used to pass a bare basename (e.g. "ru_ru.json") as that
internal_path, so any two source files that happen to share a filename in
different directories -- extremely common: many mods each ship their own
"config.json"/"README.md" -- collapsed onto the same pack entry, and every
translation but the first was silently discarded with no error.

This mirrors loose_json.py's existing approach (keep the path from "assets/"
onward when present) so every processor resolves to the same kind of
modpack-relative path instead of a collision-prone bare filename.
"""
from __future__ import annotations

import os


def compute_target_paths(file_path: str, lang_code: str, mc_dir: str | None) -> tuple[str, str, str]:
    """Compute (target_name, target_path, internal_path) for translating
    `file_path` into `lang_code` -- the "what do I name the output, where
    does the inplace/sidecar copy go, and what's its collision-safe
    resourcepack path" logic that used to be duplicated nearly verbatim
    across generic_json.py, nbt.py, yaml_toml.py, xml.py and
    line_processor.py (and, with an extra legacy-case branch, lang.py).

    target_name: the output filename -- "en_us.json" -> "<lang_code>.json"
    (case-insensitive match on "en_us", but the replacement is done on the
    ORIGINAL-case filename so a rare mixed-case source like "en_US.lang"
    keeps the rest of its name's casing instead of being forced fully
    lowercase -- this generalizes lang.py's dedicated "en_US" branch to all
    callers, and is a no-op behavior change for the overwhelming common case
    of an already-lowercase "en_us.*" source). If the source name doesn't
    contain "en_us" at all, falls back to "<stem>_<lang_code><ext>".
    target_path: target_name joined with file_path's own directory (for
    inplace/sidecar output).
    internal_path: target_name resolved to a collision-safe, mc_dir-relative
    resourcepack path (see resourcepack_internal_path above).
    """
    base_name = os.path.basename(file_path)
    lower_name = base_name.lower()
    idx = lower_name.find("en_us")
    if idx != -1:
        target_name = base_name[:idx] + lang_code + base_name[idx + len("en_us") :]
    else:
        name, ext = os.path.splitext(base_name)
        target_name = f"{name}_{lang_code}{ext}"
    target_path = os.path.join(os.path.dirname(file_path), target_name)
    internal_path = resourcepack_internal_path(file_path, mc_dir, target_name)
    return target_name, target_path, internal_path


def resourcepack_internal_path(file_path: str, mc_dir: str | None, target_name: str) -> str:
    """Return an internal zip path for `target_name` that preserves the
    source file's directory (relative to mc_dir), so distinct source files
    with the same target filename don't collide in PackWriter's flat
    namespace. Falls back to the bare `target_name` if `mc_dir` isn't known
    (e.g. a processor invoked by ArchiveProcessor against an extracted temp
    dir, which never uses resourcepack output mode anyway) or file_path isn't
    actually under mc_dir.
    """
    if not mc_dir:
        return target_name
    try:
        rel = os.path.relpath(file_path, mc_dir).replace("\\", "/")
    except ValueError:
        # Different drive on Windows -- no relative path is possible.
        return target_name
    if rel.startswith("../") or rel.startswith("..\\"):
        return target_name
    if "assets/" in rel:
        rel = rel[rel.find("assets/") :]
    rel_dir = rel.rsplit("/", 1)[0] if "/" in rel else ""
    return f"{rel_dir}/{target_name}" if rel_dir else target_name
