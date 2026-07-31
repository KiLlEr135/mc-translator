"""Streaming download orchestration for the "auto-setup local AI" feature.

Deliberately does NOT reuse utils/atomic_write.py's atomic_write(): that
helper buffers the entire payload in memory before writing, which is fine
for cache.json/settings.ini (kilobytes) but unacceptable for multi-gigabyte
model/engine downloads. This module streams chunks straight to disk instead,
using the same "write to a sibling temp file, os.replace() into place only
on full success" atomicity principle so a killed/crashed/cancelled download
never leaves a truncated file at the final path.
"""
from __future__ import annotations

import os
import shutil
from collections.abc import Callable

import requests

_CHUNK_SIZE = 1024 * 1024  # 1 MiB
_PART_SUFFIX = ".part"


class DownloadError(Exception):
    """Network/HTTP failure, or a release/asset that couldn't be resolved."""


class DownloadCancelled(Exception):
    """Raised when should_continue() returns False mid-download."""


def fetch_latest_release(repo: str) -> dict:
    """GET the GitHub "latest release" for `repo` ("owner/name"). No
    hardcoded koboldcpp version -- releases land every 1-2 weeks, so this
    always resolves whatever is current at call time."""
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise DownloadError(f"Не удалось получить информацию о релизе {repo}: {exc}") from exc
    return resp.json()


def pick_asset(release: dict, name: str) -> tuple[str, int]:
    """Exact-name match only (never substring) -- "koboldcpp.exe" must never
    accidentally match "koboldcpp-nocuda.exe" or vice versa."""
    for asset in release.get("assets", []):
        if asset.get("name") == name:
            return asset["browser_download_url"], int(asset.get("size", 0))
    raise LookupError(f"Asset '{name}' not found in release {release.get('tag_name', '?')}")


def has_enough_disk(directory: str, needed_bytes: int) -> bool:
    os.makedirs(directory, exist_ok=True)
    return shutil.disk_usage(directory).free >= needed_bytes


def ensure_downloaded(
    url: str,
    dest_path: str,
    expected_size: int,
    on_progress: Callable[[int, int], None],
    should_continue: Callable[[], bool],
) -> bool:
    """Skips the download entirely if `dest_path` already exists with
    exactly `expected_size` bytes (cheap idempotent re-run after a partial
    failure, or a repeat auto-setup click). Returns True if a download
    actually ran, False if it was skipped."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) == expected_size:
        return False
    download_stream(url, dest_path, expected_size, on_progress, should_continue)
    return True


def download_stream(
    url: str,
    dest_path: str,
    expected_size: int,
    on_progress: Callable[[int, int], None],
    should_continue: Callable[[], bool],
) -> None:
    """Streams `url` to `dest_path`, via a sibling `dest_path.part` file that
    is only os.replace()'d into place on full success. The partial file is
    removed on any failure or cancellation, so a killed/interrupted run never
    leaves a truncated file sitting at the final, "looks complete" path."""
    directory = os.path.dirname(os.path.abspath(dest_path)) or "."
    os.makedirs(directory, exist_ok=True)
    part_path = dest_path + _PART_SUFFIX

    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise DownloadError(f"Ошибка сети при загрузке {os.path.basename(dest_path)}: {exc}") from exc

    downloaded = 0
    try:
        with open(part_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                if not should_continue():
                    raise DownloadCancelled(f"Загрузка {os.path.basename(dest_path)} отменена")
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                on_progress(downloaded, expected_size)
    except DownloadCancelled:
        _remove_quietly(part_path)
        raise
    except (OSError, requests.RequestException) as exc:
        _remove_quietly(part_path)
        raise DownloadError(f"Ошибка сети при загрузке {os.path.basename(dest_path)}: {exc}") from exc
    finally:
        resp.close()

    os.replace(part_path, dest_path)


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
