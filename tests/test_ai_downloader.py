"""Tests for mc_translator.runtime.ai_downloader -- streaming download +
atomic finalize logic behind the "auto-setup local AI" feature. All network
calls are monkeypatched onto requests.get; nothing here touches a real URL."""
import os

import pytest
import requests

from mc_translator.runtime import ai_downloader as dl


class _FakeResponse:
    def __init__(self, chunks=None, status_ok=True, raise_on_iter=None):
        self._chunks = chunks or []
        self._status_ok = status_ok
        self._raise_on_iter = raise_on_iter
        self.closed = False

    def raise_for_status(self):
        if not self._status_ok:
            raise requests.HTTPError("boom")

    def iter_content(self, chunk_size):
        for chunk in self._chunks:
            if self._raise_on_iter:
                raise self._raise_on_iter
            yield chunk

    def close(self):
        self.closed = True


def test_download_stream_writes_bytes_and_finalizes(tmp_path, monkeypatch):
    dest = str(tmp_path / "model.gguf")
    payload = [b"a" * 10, b"b" * 5]

    monkeypatch.setattr(dl.requests, "get", lambda url, stream, timeout: _FakeResponse(payload))

    progress = []
    dl.download_stream("http://example/model.gguf", dest, 15, lambda got, total: progress.append((got, total)), lambda: True)

    assert os.path.exists(dest)
    assert not os.path.exists(dest + ".part")
    with open(dest, "rb") as f:
        assert f.read() == b"a" * 10 + b"b" * 5
    assert progress == [(10, 15), (15, 15)]


def test_download_stream_removes_partial_file_on_cancellation(tmp_path, monkeypatch):
    dest = str(tmp_path / "model.gguf")
    monkeypatch.setattr(dl.requests, "get", lambda url, stream, timeout: _FakeResponse([b"a" * 10, b"b" * 10]))

    calls = {"n": 0}

    def should_continue():
        calls["n"] += 1
        return calls["n"] <= 1  # allow first chunk, cancel before the second

    with pytest.raises(dl.DownloadCancelled):
        dl.download_stream("http://example/model.gguf", dest, 20, lambda got, total: None, should_continue)

    assert not os.path.exists(dest)
    assert not os.path.exists(dest + ".part")


def test_download_stream_removes_partial_file_on_http_error(tmp_path, monkeypatch):
    dest = str(tmp_path / "model.gguf")
    monkeypatch.setattr(dl.requests, "get", lambda url, stream, timeout: _FakeResponse(status_ok=False))

    with pytest.raises(dl.DownloadError):
        dl.download_stream("http://example/model.gguf", dest, 20, lambda got, total: None, lambda: True)

    assert not os.path.exists(dest)
    assert not os.path.exists(dest + ".part")


def test_download_stream_removes_partial_file_on_mid_stream_network_error(tmp_path, monkeypatch):
    dest = str(tmp_path / "model.gguf")
    monkeypatch.setattr(
        dl.requests,
        "get",
        lambda url, stream, timeout: _FakeResponse([b"a"], raise_on_iter=requests.ConnectionError("dropped")),
    )

    with pytest.raises(dl.DownloadError):
        dl.download_stream("http://example/model.gguf", dest, 20, lambda got, total: None, lambda: True)

    assert not os.path.exists(dest)
    assert not os.path.exists(dest + ".part")


def test_ensure_downloaded_skips_when_correct_size_already_exists(tmp_path, monkeypatch):
    dest = tmp_path / "model.gguf"
    dest.write_bytes(b"x" * 20)

    def fail_get(*a, **kw):
        raise AssertionError("must not hit the network when size already matches")

    monkeypatch.setattr(dl.requests, "get", fail_get)

    ran = dl.ensure_downloaded("http://example/model.gguf", str(dest), 20, lambda got, total: None, lambda: True)

    assert ran is False


def test_ensure_downloaded_redownloads_when_size_mismatches(tmp_path, monkeypatch):
    dest = tmp_path / "model.gguf"
    dest.write_bytes(b"x" * 5)  # wrong size -- a stale/corrupt partial from an old run

    monkeypatch.setattr(dl.requests, "get", lambda url, stream, timeout: _FakeResponse([b"y" * 20]))

    ran = dl.ensure_downloaded("http://example/model.gguf", str(dest), 20, lambda got, total: None, lambda: True)

    assert ran is True
    assert dest.read_bytes() == b"y" * 20


def test_pick_asset_exact_match_only():
    release = {
        "assets": [
            {"name": "koboldcpp.exe", "browser_download_url": "http://x/koboldcpp.exe", "size": 100},
            {"name": "koboldcpp-nocuda.exe", "browser_download_url": "http://x/nocuda.exe", "size": 50},
        ]
    }
    url, size = dl.pick_asset(release, "koboldcpp.exe")
    assert url == "http://x/koboldcpp.exe"
    assert size == 100


def test_pick_asset_raises_lookup_error_when_absent():
    release = {"assets": [{"name": "koboldcpp-nocuda.exe", "browser_download_url": "http://x", "size": 1}]}
    with pytest.raises(LookupError):
        dl.pick_asset(release, "koboldcpp.exe")


def test_fetch_latest_release_wraps_network_error(monkeypatch):
    def fake_get(url, timeout):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(dl.requests, "get", fake_get)

    with pytest.raises(dl.DownloadError):
        dl.fetch_latest_release("LostRuins/koboldcpp")


def test_has_enough_disk_true_when_plenty_free(tmp_path):
    assert dl.has_enough_disk(str(tmp_path), 1) is True


def test_has_enough_disk_false_when_insufficient(tmp_path):
    assert dl.has_enough_disk(str(tmp_path), 10**18) is False
