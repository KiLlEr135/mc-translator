"""Tests for mc_translator.output.pack_writer — resourcepack/datapack zip
construction, including legacy (1.12-) .json->.lang conversion and
duplicate-path dedup."""
import json
import zipfile

from mc_translator.output.pack_writer import PackWriter


def test_write_creates_entry_in_resourcepack(tmp_path):
    pw = PackWriter(str(tmp_path), "TestPack", "1.20.1", "Russian")
    pw.write("assets/mod/lang/ru_ru.json", b'{"a": "b"}')
    rp, dp = pw.close()
    with zipfile.ZipFile(rp) as z:
        assert "assets/mod/lang/ru_ru.json" in z.namelist()
        assert z.read("assets/mod/lang/ru_ru.json") == b'{"a": "b"}'


def test_data_path_routes_to_datapack(tmp_path):
    pw = PackWriter(str(tmp_path), "TestPack", "1.20.1", "Russian")
    pw.write("data/mymod/functions/foo.mcfunction", b"say hi")
    rp, dp = pw.close()
    with zipfile.ZipFile(dp) as z:
        assert "data/mymod/functions/foo.mcfunction" in z.namelist()
    with zipfile.ZipFile(rp) as z:
        assert "data/mymod/functions/foo.mcfunction" not in z.namelist()


def test_duplicate_path_is_dropped_silently(tmp_path):
    """PackWriter.write dedups by exact internal_path -- the second write to
    the same path is a silent no-op (documented, intentional: it's what
    made the resourcepack basename-collision bug possible before processors
    were fixed to pass collision-safe paths)."""
    pw = PackWriter(str(tmp_path), "TestPack", "1.20.1", "Russian")
    pw.write("assets/mod/lang/ru_ru.json", b'{"first": true}')
    pw.write("assets/mod/lang/ru_ru.json", b'{"second": true}')
    rp, dp = pw.close()
    with zipfile.ZipFile(rp) as z:
        names = z.namelist()
        assert names.count("assets/mod/lang/ru_ru.json") == 1
        assert z.read("assets/mod/lang/ru_ru.json") == b'{"first": true}'


def test_legacy_version_converts_json_lang_to_key_value_format(tmp_path):
    pw = PackWriter(str(tmp_path), "TestPack", "1.12.2", "Russian")
    payload = json.dumps(
        {"item.mod.thing": "Штука", "item.mod.other": "Другое"}, ensure_ascii=False
    ).encode("utf-8")
    pw.write("assets/mod/lang/ru_ru.json", payload)
    rp, dp = pw.close()
    with zipfile.ZipFile(rp) as z:
        names = z.namelist()
        # Renamed to .lang, with the legacy region-case convention applied.
        assert "assets/mod/lang/ru_RU.lang" in names
        assert "assets/mod/lang/ru_ru.json" not in names
        content = z.read("assets/mod/lang/ru_RU.lang").decode("utf-8")
        assert "item.mod.thing=Штука" in content
        assert "item.mod.other=Другое" in content
        assert "{" not in content  # payload is key=value lines, not JSON


def test_legacy_version_leaves_non_lang_json_untouched(tmp_path):
    pw = PackWriter(str(tmp_path), "TestPack", "1.12.2", "Russian")
    payload = b'{"not": "a lang file"}'
    pw.write("assets/mod/somefile.json", payload)
    rp, dp = pw.close()
    with zipfile.ZipFile(rp) as z:
        assert "assets/mod/somefile.json" in z.namelist()
        assert z.read("assets/mod/somefile.json") == payload


def test_json_lang_to_legacy_handles_unexpected_shapes_gracefully():
    assert PackWriter._json_lang_to_legacy(b'"just a string"') is None
    assert PackWriter._json_lang_to_legacy(b"not json at all") is None
    assert PackWriter._json_lang_to_legacy(b'{"a": "b"}') == b"a=b\n"


def test_json_lang_to_legacy_flattens_embedded_newlines():
    payload = json.dumps({"k": "line one\nline two"}).encode("utf-8")
    result = PackWriter._json_lang_to_legacy(payload)
    assert result == b"k=line one line two\n"


def test_new_run_seeds_from_previous_pack_and_keeps_untouched_entries(tmp_path):
    """The real bug this guards against: a prior complete run wrote books/
    guides/interface text into the pack; a later run that only reprocesses
    a handful of files (because it got interrupted, or Stop was clicked, or
    it simply covers fewer file types this time) must not silently wipe out
    everything the previous run wrote and never touched again."""
    pw1 = PackWriter(str(tmp_path), "TestPack", "1.20.1", "Russian")
    pw1.write("assets/bookmod/lang/ru_ru.json", b'{"guide.title": "Old Guide"}')
    pw1.write("assets/other/lang/ru_ru.json", b'{"item.x": "X"}')
    pw1.close()

    pw2 = PackWriter(str(tmp_path), "TestPack", "1.20.1", "Russian")
    pw2.write("assets/other/lang/ru_ru.json", b'{"item.x": "X updated"}')
    rp, dp = pw2.close()

    with zipfile.ZipFile(rp) as z:
        names = z.namelist()
        assert names.count("assets/bookmod/lang/ru_ru.json") == 1
        assert z.read("assets/bookmod/lang/ru_ru.json") == b'{"guide.title": "Old Guide"}'
        assert z.read("assets/other/lang/ru_ru.json") == b'{"item.x": "X updated"}'


def test_new_run_with_no_writes_at_all_leaves_previous_pack_intact(tmp_path):
    """Simulates a run that gets interrupted before any processor calls
    write() -- e.g. Stop clicked or a crash right after construction. The
    zip PackWriter produces on close() must still be the full previous pack,
    not an empty one."""
    pw1 = PackWriter(str(tmp_path), "TestPack", "1.20.1", "Russian")
    pw1.write("assets/bookmod/lang/ru_ru.json", b'{"guide.title": "Old Guide"}')
    pw1.close()

    pw2 = PackWriter(str(tmp_path), "TestPack", "1.20.1", "Russian")
    rp, dp = pw2.close()

    with zipfile.ZipFile(rp) as z:
        assert z.read("assets/bookmod/lang/ru_ru.json") == b'{"guide.title": "Old Guide"}'


def test_seeding_ignores_a_corrupted_previous_pack(tmp_path):
    """A hard-killed prior run can leave a truncated/invalid zip on disk
    (confirmed in this project's own logs: "File is not a zip file"). That
    must not crash the next run -- just start this pack fresh."""
    rp_dir = tmp_path / "resourcepacks"
    rp_dir.mkdir()
    (rp_dir / "TestPack.zip").write_bytes(b"not a real zip file")

    pw = PackWriter(str(tmp_path), "TestPack", "1.20.1", "Russian")
    pw.write("assets/mod/lang/ru_ru.json", b'{"a": "b"}')
    rp, dp = pw.close()

    with zipfile.ZipFile(rp) as z:
        assert z.read("assets/mod/lang/ru_ru.json") == b'{"a": "b"}'


def test_datapack_entries_are_also_seeded_from_previous_run(tmp_path):
    pw1 = PackWriter(str(tmp_path), "TestPack", "1.20.1", "Russian")
    pw1.write("data/mymod/functions/foo.mcfunction", b"say old")
    pw1.close()

    pw2 = PackWriter(str(tmp_path), "TestPack", "1.20.1", "Russian")
    rp, dp = pw2.close()

    with zipfile.ZipFile(dp) as z:
        assert z.read("data/mymod/functions/foo.mcfunction") == b"say old"
