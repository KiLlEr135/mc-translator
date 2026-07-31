"""Tests for mc_translator.usage_log — the per-mc_dir, per-language record of
which mods use which translated string, backing the GUI's review screen."""
import os

from mc_translator.usage_log import load_usage_log, record_usage, save_usage_log


def test_load_usage_log_missing_file_returns_empty(tmp_path):
    assert load_usage_log(str(tmp_path), "ru_ru") == {}


def test_save_and_load_usage_log_round_trip(tmp_path):
    usage = {"Hello": {"mods": ["ModA"], "kind": "Интерфейс", "engine": "google"}}
    save_usage_log(str(tmp_path), "ru_ru", usage)
    assert load_usage_log(str(tmp_path), "ru_ru") == usage


def test_usage_logs_are_scoped_per_language(tmp_path):
    save_usage_log(str(tmp_path), "ru_ru", {"a": 1})
    save_usage_log(str(tmp_path), "de_de", {"b": 2})
    assert load_usage_log(str(tmp_path), "ru_ru") == {"a": 1}
    assert load_usage_log(str(tmp_path), "de_de") == {"b": 2}


def test_load_usage_log_corrupt_file_returns_empty(tmp_path):
    path = os.path.join(str(tmp_path), ".mc_translator_usage_ru_ru.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not valid json")
    assert load_usage_log(str(tmp_path), "ru_ru") == {}


def test_record_usage_creates_new_entry():
    usage = {}
    record_usage(usage, "Shift-click to move", "Applied Energistics 2", "Интерфейс", "google")
    assert usage == {
        "Shift-click to move": {
            "mods": ["Applied Energistics 2"],
            "kind": "Интерфейс",
            "engine": "google",
        }
    }


def test_record_usage_accumulates_mods_without_duplicates():
    usage = {}
    record_usage(usage, "Shift-click to move", "Applied Energistics 2", "Интерфейс", "google")
    record_usage(usage, "Shift-click to move", "Mekanism", "Интерфейс", "google")
    record_usage(usage, "Shift-click to move", "Applied Energistics 2", "Интерфейс", "google")
    assert usage["Shift-click to move"]["mods"] == ["Applied Energistics 2", "Mekanism"]


def test_record_usage_ignores_empty_source_text():
    usage = {}
    record_usage(usage, "", "Applied Energistics 2", "Интерфейс", "google")
    assert usage == {}


def test_record_usage_updates_kind_and_engine_on_later_calls():
    usage = {}
    record_usage(usage, "Confirm", "ModA", "JSON", "google")
    record_usage(usage, "Confirm", "ModB", "Книга(JSON)", "ai")
    assert usage["Confirm"]["kind"] == "Книга(JSON)"
    assert usage["Confirm"]["engine"] == "ai"
    assert usage["Confirm"]["mods"] == ["ModA", "ModB"]
