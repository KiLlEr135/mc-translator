import json
import re
from collections.abc import Iterator
from typing import Any

from mc_translator.constants import KEYS_TO_TRANSLATE


def load_lenient_json(raw: bytes | str) -> Any:
    if isinstance(raw, bytes):
        # "replace" (not "ignore"): a genuinely non-UTF-8 file is rare, but
        # silently DROPPING its bad bytes would delete characters from the
        # text with no visible sign of damage; a U+FFFD placeholder at
        # least surfaces that this file wasn't clean UTF-8.
        text = raw.decode("utf-8-sig", errors="replace")
    else:
        text = raw
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^\s*//.*$", "", text)
    text = re.sub(r",\s*([\]}])", r"\1", text)
    return json.loads(text)


def path_to_key(path: tuple) -> str:
    return "/".join(str(p) for p in path)


def key_to_path(key: str) -> tuple:
    return tuple(key.split("/"))


def get_at_path(data: Any, path: tuple) -> Any:
    cur = data
    for part in path:
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def set_at_path(data: Any, path: tuple, value: Any) -> None:
    cur = data
    for part in path[:-1]:
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    last = path[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def iter_translatable_strings(data: Any, path: tuple = ()) -> Iterator[tuple[tuple, str]]:
    """Yield (json_path, string) for book/guide JSON structures."""
    if isinstance(data, dict):
        for key, value in data.items():
            child_path = path + (key,)
            if key in KEYS_TO_TRANSLATE:
                if isinstance(value, str):
                    yield child_path, value
                elif isinstance(value, list):
                    for idx, item in enumerate(value):
                        item_path = child_path + (idx,)
                        if isinstance(item, str):
                            yield item_path, item
                        elif isinstance(item, (dict, list)):
                            yield from iter_translatable_strings(item, item_path)
                elif isinstance(value, dict):
                    yield from iter_translatable_strings(value, child_path)
            elif isinstance(value, (dict, list)):
                yield from iter_translatable_strings(value, child_path)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            yield from iter_translatable_strings(item, path + (idx,))


def iter_all_json_strings(data: Any, path: tuple = ()) -> Iterator[tuple[tuple, str]]:
    """Yield (json_path, string) for every string value anywhere in a JSON
    structure, regardless of key name. Unlike iter_translatable_strings
    (book/guide JSON, gated by the KEYS_TO_TRANSLATE allowlist), this is for
    arbitrary mod/resourcepack/config JSON where translatable text can live
    under any key."""
    if isinstance(data, dict):
        for key, value in data.items():
            child_path = path + (key,)
            if isinstance(value, str):
                yield child_path, value
            elif isinstance(value, (dict, list)):
                yield from iter_all_json_strings(value, child_path)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            item_path = path + (idx,)
            if isinstance(item, str):
                yield item_path, item
            elif isinstance(item, (dict, list)):
                yield from iter_all_json_strings(item, item_path)


def apply_translations_by_path(data: Any, translations: dict[str, str]) -> None:
    for key, translated in translations.items():
        set_at_path(data, key_to_path(key), translated)
