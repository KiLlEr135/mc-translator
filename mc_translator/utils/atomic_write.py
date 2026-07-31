"""Crash-safe file writes.

Several files in this project are appended-to or rewritten many times over a
long translation run (cache.json, .mc_translator_manifest_*.json,
.mc_translator_usage_*.json, settings.ini, and every processor's in-place
output file). Writing directly to the destination with open(path, "w") means
a kill/crash/power-cut mid-write leaves a truncated or partial file behind --
for cache.json in particular, that silently discards every cached
translation (including paid AI/DeepL ones) the next time it fails to parse
and gets reset to {}.

atomic_write sidesteps this: write the full payload to a temp file in the
*same directory* (so the final os.replace is on the same filesystem, and
therefore atomic on both POSIX and Windows), then swap it into place. Readers
only ever see either the fully-old or the fully-new content, never a partial
write.
"""
from __future__ import annotations

import os
import tempfile


def atomic_write(path: str, data: bytes) -> None:
    """Write `data` to `path` atomically (temp file + os.replace)."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    """Text convenience wrapper around atomic_write."""
    atomic_write(path, text.encode(encoding))
