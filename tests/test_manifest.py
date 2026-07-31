"""Tests for mc_translator.runtime.manifest — the incremental jar-skip manifest."""
import os

from mc_translator.runtime.manifest import (
    jar_fingerprint,
    load_estimate_cache,
    load_jar_manifest,
    save_estimate_cache,
    save_jar_manifest,
)


def test_load_jar_manifest_missing_file_returns_empty(tmp_path):
    assert load_jar_manifest(str(tmp_path), "ru_ru") == {}


def test_save_and_load_jar_manifest_round_trip(tmp_path):
    manifest = {"C:/mods/foo.jar": {"mtime": 123.0, "size": 456, "mods": True, "books": True, "output": "resourcepack"}}
    save_jar_manifest(str(tmp_path), "ru_ru", manifest)
    assert load_jar_manifest(str(tmp_path), "ru_ru") == manifest


def test_manifests_are_scoped_per_language(tmp_path):
    save_jar_manifest(str(tmp_path), "ru_ru", {"a": 1})
    save_jar_manifest(str(tmp_path), "de_de", {"b": 2})
    assert load_jar_manifest(str(tmp_path), "ru_ru") == {"a": 1}
    assert load_jar_manifest(str(tmp_path), "de_de") == {"b": 2}


def test_load_jar_manifest_corrupt_file_returns_empty(tmp_path):
    path = os.path.join(str(tmp_path), ".mc_translator_manifest_ru_ru.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not valid json")
    assert load_jar_manifest(str(tmp_path), "ru_ru") == {}


def test_jar_fingerprint_reflects_mtime_and_size(tmp_path):
    jar_path = tmp_path / "mod.jar"
    jar_path.write_bytes(b"fake jar content")
    fp1 = jar_fingerprint(str(jar_path), translate_mods=True, translate_books=True, output_mode="resourcepack")
    assert fp1["size"] == len(b"fake jar content")

    jar_path.write_bytes(b"different content, different size!")
    fp2 = jar_fingerprint(str(jar_path), translate_mods=True, translate_books=True, output_mode="resourcepack")
    assert fp2["size"] != fp1["size"]
    assert fp1 != fp2


def test_jar_fingerprint_missing_file_returns_none(tmp_path):
    assert jar_fingerprint(str(tmp_path / "missing.jar"), translate_mods=True, translate_books=True, output_mode="resourcepack") is None


def test_jar_fingerprint_differs_when_options_differ(tmp_path):
    jar_path = tmp_path / "mod.jar"
    jar_path.write_bytes(b"content")
    fp_mods_only = jar_fingerprint(str(jar_path), translate_mods=True, translate_books=False, output_mode="resourcepack")
    fp_mods_and_books = jar_fingerprint(str(jar_path), translate_mods=True, translate_books=True, output_mode="resourcepack")
    assert fp_mods_only != fp_mods_and_books

    fp_inplace = jar_fingerprint(str(jar_path), translate_mods=True, translate_books=True, output_mode="inplace")
    assert fp_inplace != fp_mods_and_books


def test_load_estimate_cache_missing_file_returns_empty(tmp_path):
    assert load_estimate_cache(str(tmp_path), "ru_ru") == {}


def test_save_and_load_estimate_cache_round_trip(tmp_path):
    cache = {
        "C:/mods/foo.jar": {
            "mtime": 123.0, "size": 456, "mode": "append",
            "mods": True, "books": True, "count": 5,
        }
    }
    save_estimate_cache(str(tmp_path), "ru_ru", cache)
    assert load_estimate_cache(str(tmp_path), "ru_ru") == cache


def test_estimate_cache_scoped_per_language(tmp_path):
    save_estimate_cache(str(tmp_path), "ru_ru", {"a": 1})
    save_estimate_cache(str(tmp_path), "de_de", {"b": 2})
    assert load_estimate_cache(str(tmp_path), "ru_ru") == {"a": 1}
    assert load_estimate_cache(str(tmp_path), "de_de") == {"b": 2}


def test_load_estimate_cache_corrupt_file_returns_empty(tmp_path):
    path = os.path.join(str(tmp_path), ".mc_translator_estimates_ru_ru.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not valid json")
    assert load_estimate_cache(str(tmp_path), "ru_ru") == {}
