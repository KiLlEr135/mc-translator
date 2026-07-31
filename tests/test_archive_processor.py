"""Tests for ArchiveProcessor -- regressions found and fixed:

- SnbtProcessor.process has signature (file_path, *, target_lang, mode), with
  no output_mode/pack_writer params (unlike every other inner processor), but
  ArchiveProcessor used to call it uniformly with output_mode="inplace",
  pack_writer=None -- a TypeError on every .snbt file inside an archive,
  silently swallowed by the per-file try/except and logged as an error. .snbt
  is a declared-supported format that never actually worked.
- output_mode == "resourcepack" used to extract+translate the whole archive
  and then repack it as a single zip blob written via pack_writer.write(),
  landing at a path like "mods/foo.zip" the game never reads as a resource
  pack -- all translation work wasted, so ArchiveProcessor used to skip
  archives entirely (with a log message) in resourcepack mode instead. Now
  it merges every translated file INDIVIDUALLY into the combined output pack
  at its own path relative to the archive root -- a valid pack zip always
  has assets/ or data/ at its own root, so this is exactly the shape
  PackWriter expects, and PackWriter.handle_for_path already routes
  "data/..." to the datapack zip and everything else to the resourcepack
  zip. This mirrors how translated jar content already gets merged into the
  output pack (same data path as the original, so the game's normal
  pack-priority "last loaded wins" applies the translation).
"""
import json
import zipfile

from mc_translator.output.pack_writer import PackWriter
from mc_translator.processors.archive import ArchiveProcessor


def test_archive_translates_snbt_without_raising_inplace(
    tmp_path, fake_service, job_state, fake_callbacks, lang
):
    zip_path = tmp_path / "quests.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("assets/modid/lang/en_us.json", json.dumps({"key": "Hello"}))
        zf.writestr(
            "config/ftbquests/quests/chapter/foo.snbt",
            'title: "Hello World"\n',
        )

    proc = ArchiveProcessor(fake_service, job_state, fake_callbacks)
    # Must not raise -- previously a TypeError from the incompatible
    # SnbtProcessor.process() call, swallowed by the surrounding try/except
    # and logged as "Ошибка обработки ... в архиве", leaving .snbt untouched.
    proc.process(
        str(zip_path),
        target_lang=lang,
        mode="append",
        output_mode="inplace",
        pack_writer=None,
    )

    with zipfile.ZipFile(zip_path, "r") as zf:
        snbt_content = zf.read("config/ftbquests/quests/chapter/foo.snbt").decode("utf-8")
    assert "TR[Hello World]" in snbt_content

    # No error should have been logged for the .snbt file.
    assert not any("snbt" in msg.lower() and tag == "red" for msg, tag in fake_callbacks.logs)


def test_archive_resourcepack_mode_merges_translated_files(
    tmp_path, fake_service, job_state, fake_callbacks, lang
):
    zip_path = tmp_path / "extra.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("pack.mcmeta", json.dumps({"pack": {"pack_format": 15, "description": "Extra"}}))
        zf.writestr("assets/modid/lang/en_us.json", json.dumps({"key": "Hello"}))
        zf.writestr("data/modid/tags/foo.json", json.dumps({"values": ["a"]}))

    written: dict[str, bytes] = {}

    class FakePackWriter:
        def write(self, internal_path, data):
            written[internal_path] = data

    proc = ArchiveProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        str(zip_path),
        target_lang=lang,
        mode="append",
        output_mode="resourcepack",
        pack_writer=FakePackWriter(),
    )

    # The lang JSON was actually translated (proves the inner extract-and-
    # translate step still runs the same way it does for inplace mode).
    assert "assets/modid/lang/ru_ru.json" in written
    assert json.loads(written["assets/modid/lang/ru_ru.json"]) == {"key": "TR[Hello]"}
    # Non-translatable content still gets carried through unchanged (the
    # merged pack must remain functionally complete, not just the strings
    # that happened to be translatable).
    assert "data/modid/tags/foo.json" in written
    # The inner archive's OWN pack.mcmeta must never be merged -- PackWriter
    # already wrote the real one at construction time, bypassing write()'s
    # dedup tracking, so a second write here would not be deduped.
    assert "pack.mcmeta" not in written


def test_archive_resourcepack_routes_data_vs_assets_with_real_pack_writer(
    tmp_path, fake_service, job_state, fake_callbacks, lang
):
    """End-to-end with a real PackWriter (not a fake) -- confirms assets/
    lands in the resourcepack zip, data/ lands in the datapack zip, and the
    pack's own pack.mcmeta is untouched/not duplicated in either."""
    zip_path = tmp_path / "mc" / "datapacks" / "extra.zip"
    zip_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("pack.mcmeta", json.dumps({"pack": {"pack_format": 15, "description": "Extra"}}))
        zf.writestr("assets/modid/lang/en_us.json", json.dumps({"key": "Hello"}))
        zf.writestr("data/modid/tags/foo.json", json.dumps({"values": ["a"]}))

    mc_dir = tmp_path / "mc"
    pack_writer = PackWriter(str(mc_dir), "MCTranslator_Pack", "1.21.1", lang["name"])
    try:
        proc = ArchiveProcessor(fake_service, job_state, fake_callbacks)
        proc.process(
            str(zip_path),
            target_lang=lang,
            mode="append",
            output_mode="resourcepack",
            pack_writer=pack_writer,
            mc_dir=str(mc_dir),
        )
    finally:
        rp_path, dp_path = pack_writer.close()

    with zipfile.ZipFile(rp_path) as rp:
        rp_names = rp.namelist()
        assert "assets/modid/lang/ru_ru.json" in rp_names
        assert "data/modid/tags/foo.json" not in rp_names
        assert rp_names.count("pack.mcmeta") == 1

    with zipfile.ZipFile(dp_path) as dp:
        dp_names = dp.namelist()
        assert "data/modid/tags/foo.json" in dp_names
        assert "assets/modid/lang/ru_ru.json" not in dp_names
        assert dp_names.count("pack.mcmeta") == 1
