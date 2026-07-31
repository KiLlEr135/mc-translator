"""Tests for LooseJsonProcessor's target-path resolution -- a regression
found and fixed in this project:

LooseJsonProcessor used to compute its own internal/disk paths inline
instead of using resourcepack_path.compute_target_paths: any source file
without "assets/" in its mc_dir-relative path (e.g.
config/ftbquests/lang/en_us.json and defaultconfigs/*/lang/en_us.json --
both are entries in constants.LOOSE_JSON_SEARCH_DIRS) fell back to the same
hardcoded "assets/kubejs/lang/<basename>" internal path, so two such files
from different mods/sources collapsed onto one resourcepack entry and
PackWriter (first-write-wins) silently dropped every one after the first.
The old code's `.replace("en_us.json", ...)` was also case-sensitive, so a
mixed-case source name like "En_Us.json" was never renamed and the
"translation" got written right back over the English source instead of
into a sidecar file.
"""
import json
import zipfile

from mc_translator.output.pack_writer import PackWriter
from mc_translator.processors.loose_json import LooseJsonProcessor


def test_same_basename_from_different_search_dirs_does_not_collide(
    tmp_path, fake_service, job_state, fake_callbacks, lang
):
    default_configs_file = tmp_path / "defaultconfigs" / "modA" / "lang" / "en_us.json"
    default_configs_file.parent.mkdir(parents=True)
    default_configs_file.write_text(json.dumps({"greeting": "Hello from mod A"}), encoding="utf-8")

    ftbquests_file = tmp_path / "config" / "ftbquests" / "lang" / "en_us.json"
    ftbquests_file.parent.mkdir(parents=True)
    ftbquests_file.write_text(json.dumps({"quest.title": "Hello from FTB Quests"}), encoding="utf-8")

    pw = PackWriter(str(tmp_path), "TestPack", "1.20.1", "Russian")
    proc = LooseJsonProcessor(fake_service, job_state, fake_callbacks)
    for f in (default_configs_file, ftbquests_file):
        proc.process(
            str(f),
            str(tmp_path),
            target_lang=lang,
            mode="force",
            output_mode="resourcepack",
            pack_writer=pw,
        )
    rp, _dp = pw.close()

    with zipfile.ZipFile(rp) as z:
        names = set(z.namelist())
        assert "defaultconfigs/modA/lang/ru_ru.json" in names
        assert "config/ftbquests/lang/ru_ru.json" in names
        # Both wrote to distinct entries -- neither was silently dropped.
        assert json.loads(z.read("defaultconfigs/modA/lang/ru_ru.json"))["greeting"] == "TR[Hello from mod A]"
        assert (
            json.loads(z.read("config/ftbquests/lang/ru_ru.json"))["quest.title"]
            == "TR[Hello from FTB Quests]"
        )


def test_mixed_case_source_name_writes_sidecar_not_overwrite(
    tmp_path, fake_service, job_state, fake_callbacks, lang
):
    src = tmp_path / "En_Us.json"
    original_text = json.dumps({"greeting": "Hello"})
    src.write_text(original_text, encoding="utf-8")

    proc = LooseJsonProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        str(src),
        str(tmp_path),
        target_lang=lang,
        mode="force",
        output_mode="inplace",
        pack_writer=None,
    )

    # Source file must be untouched -- translation goes to a sidecar file.
    assert src.read_text(encoding="utf-8") == original_text

    sidecar = tmp_path / "ru_ru.json"
    assert sidecar.exists()
    assert json.loads(sidecar.read_text(encoding="utf-8"))["greeting"] == "TR[Hello]"
