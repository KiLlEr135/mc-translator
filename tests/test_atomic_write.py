"""Tests for mc_translator.utils.atomic_write — crash-safe file writes used
by cache.py/manifest.py/usage_log.py/config.py to avoid leaving a truncated
file behind if the process dies mid-write."""
import os

import pytest

from mc_translator.utils.atomic_write import atomic_write, atomic_write_text


def test_atomic_write_creates_new_file(tmp_path):
    path = str(tmp_path / "out.bin")
    atomic_write(path, b"hello")
    with open(path, "rb") as f:
        assert f.read() == b"hello"


def test_atomic_write_overwrites_existing_file(tmp_path):
    path = str(tmp_path / "out.bin")
    with open(path, "wb") as f:
        f.write(b"old content that is longer than new")
    atomic_write(path, b"new")
    with open(path, "rb") as f:
        assert f.read() == b"new"


def test_atomic_write_leaves_no_temp_file_behind(tmp_path):
    path = str(tmp_path / "out.bin")
    atomic_write(path, b"data")
    remaining = os.listdir(str(tmp_path))
    assert remaining == ["out.bin"]


def test_atomic_write_failure_does_not_corrupt_existing_file(tmp_path, monkeypatch):
    path = str(tmp_path / "out.bin")
    with open(path, "wb") as f:
        f.write(b"original")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write(path, b"new data")

    # The original file must be untouched -- os.replace never ran, so the
    # temp file (now cleaned up) never became the destination.
    with open(path, "rb") as f:
        assert f.read() == b"original"
    # And the failed write's temp file must not be left behind.
    remaining = os.listdir(str(tmp_path))
    assert remaining == ["out.bin"]


def test_atomic_write_text_encodes_utf8(tmp_path):
    path = str(tmp_path / "out.txt")
    atomic_write_text(path, "Привет мир")
    with open(path, encoding="utf-8") as f:
        assert f.read() == "Привет мир"
