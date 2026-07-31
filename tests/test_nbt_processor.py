"""Regression test for NbtProcessor's append-mode value preservation -- the
same class of bug fixed in generic_json.py/yaml_toml.py (see
test_generic_json_and_xml.py / test_yaml_toml_processor.py):

NbtProcessor used to only ever write freshly-translated (`resolved`) values
into `nbt_data` before saving; any already-translated path that wasn't part
of this run's `pairs` was never re-applied from `existing_map`, so a second
run in append/skip mode silently reverted previously-translated NBT strings
back to English.
"""
import pytest

nbtlib = pytest.importorskip("nbtlib")
from nbtlib import tag  # noqa: E402

from mc_translator.processors.nbt import NbtProcessor  # noqa: E402


def _make_nbt(path, values: dict) -> None:
    compound = tag.Compound({k: tag.String(v) for k, v in values.items()})
    nbtlib.File(compound).save(str(path))


def test_nbt_second_run_preserves_prior_translations_and_translates_new_tag(
    tmp_path, fake_service, job_state, fake_callbacks
):
    src = tmp_path / "en_us.nbt"
    _make_nbt(src, {"greeting": "Hello there", "farewell": "Goodbye friend"})
    target = tmp_path / "ru_ru.nbt"

    proc = NbtProcessor(fake_service, job_state, fake_callbacks)
    target_lang = {"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"}

    # First run (force mode): both string tags get translated.
    proc.process(
        str(src),
        target_lang=target_lang,
        mode="force",
        output_mode="inplace",
        mc_dir=str(tmp_path),
    )
    assert target.exists()
    first_result = nbtlib.load(str(target))
    assert str(first_result["greeting"]) == "TR[Hello there]"
    assert str(first_result["farewell"]) == "TR[Goodbye friend]"

    # Source gets a new, third string tag before the second (append-mode) run.
    _make_nbt(
        src,
        {
            "greeting": "Hello there",
            "farewell": "Goodbye friend",
            "welcome": "Welcome home",
        },
    )
    fake_service.calls.clear()

    proc.process(
        str(src),
        target_lang=target_lang,
        mode="append",
        output_mode="inplace",
        mc_dir=str(tmp_path),
    )

    second_result = nbtlib.load(str(target))
    # Both previously-translated tags must survive, NOT revert to English.
    assert str(second_result["greeting"]) == "TR[Hello there]"
    assert str(second_result["farewell"]) == "TR[Goodbye friend]"
    # The brand-new tag must be freshly translated.
    assert str(second_result["welcome"]) == "TR[Welcome home]"
    # Only the brand-new string should have gone through translate_dict.
    translated_strings = {k: v for call, _ in fake_service.calls for k, v in call.items()}
    assert "Welcome home" in translated_strings.values()
    assert "Hello there" not in translated_strings.values()
    assert "Goodbye friend" not in translated_strings.values()


def test_nbt_skip_mode_does_not_retranslate_completed_strings(
    tmp_path, fake_service, job_state, fake_callbacks
):
    src = tmp_path / "en_us.nbt"
    _make_nbt(src, {"greeting": "Hello"})
    target = tmp_path / "ru_ru.nbt"
    _make_nbt(target, {"greeting": "Привет"})

    proc = NbtProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        str(src),
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="skip",
        output_mode="inplace",
        mc_dir=str(tmp_path),
    )
    assert fake_service.calls == []
    result = nbtlib.load(str(target))
    assert str(result["greeting"]) == "Привет"


def test_nbt_list_indexed_paths_are_seeded_and_deduped_correctly(
    tmp_path, fake_service, job_state, fake_callbacks
):
    """Regression test for the list-index path-comparison bug: pairs_paths
    used to compare _walk_nbt's real (str, int)-mixed path tuples against
    seed paths reconstructed as all-str from existing_map's "/"-joined key,
    so the membership check never matched for a List-indexed path -- a value
    could be queued into both `seed` and `resolved` at once for the same
    path, with correctness resting entirely on undocumented order-of-
    application. This drives the walk through a ListTag so at least one
    collected path contains an int index."""
    src = tmp_path / "en_us.nbt"
    items = tag.List[tag.Compound]([
        tag.Compound({"name": tag.String("First item")}),
        tag.Compound({"name": tag.String("Second item")}),
    ])
    nbtlib.File(tag.Compound({"items": items})).save(str(src))
    target = tmp_path / "ru_ru.nbt"

    proc = NbtProcessor(fake_service, job_state, fake_callbacks)
    target_lang = {"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"}

    proc.process(
        str(src), target_lang=target_lang, mode="force",
        output_mode="inplace", mc_dir=str(tmp_path),
    )
    first_result = nbtlib.load(str(target))
    assert str(first_result["items"][0]["name"]) == "TR[First item]"
    assert str(first_result["items"][1]["name"]) == "TR[Second item]"

    # Second run, source unchanged -- append mode must reuse both list
    # entries' translations without sending them through translate_dict again.
    fake_service.calls.clear()
    proc.process(
        str(src), target_lang=target_lang, mode="append",
        output_mode="inplace", mc_dir=str(tmp_path),
    )
    second_result = nbtlib.load(str(target))
    assert str(second_result["items"][0]["name"]) == "TR[First item]"
    assert str(second_result["items"][1]["name"]) == "TR[Second item]"
    assert fake_service.calls == []  # nothing new to translate -- no API calls


def test_nbt_resourcepack_mode_skips_saves_with_warning(
    tmp_path, fake_service, job_state, fake_callbacks
):
    """Regression test: a real ATM10 run showed NbtProcessor (previously
    100% non-functional, see the module fix above) dutifully translating
    every .nbt/.dat file under "saves/<world>/..." in resourcepack mode --
    140 entries, none of them ever loadable, since resource packs only
    deliver content under "assets/". World save NBT is now skipped early
    (with a warning) in resourcepack mode specifically."""
    saves_dir = tmp_path / "saves" / "MyWorld"
    saves_dir.mkdir(parents=True)
    src = saves_dir / "level.dat"
    _make_nbt(src, {"greeting": "Hello there"})

    class FakePackWriter:
        def write(self, internal_path, data):
            raise AssertionError("pack_writer.write should never be called for saves/ NBT")

    proc = NbtProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        str(src),
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="force",
        output_mode="resourcepack",
        pack_writer=FakePackWriter(),
        mc_dir=str(tmp_path),
    )

    assert fake_service.calls == []
    assert any(tag == "yellow" and "ресурспак" in msg.lower() for msg, tag in fake_callbacks.logs)


def test_nbt_resourcepack_mode_still_translates_config_nbt(
    tmp_path, fake_service, job_state, fake_callbacks
):
    """The saves/ skip must not over-broadly catch "config/" (or other)
    NBT sources -- only "saves/" is provably dead in resourcepack mode."""
    config_dir = tmp_path / "config" / "somemod"
    config_dir.mkdir(parents=True)
    src = config_dir / "en_us.dat"
    _make_nbt(src, {"greeting": "Hello there"})

    proc = NbtProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        str(src),
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="force",
        output_mode="resourcepack",
        pack_writer=None,
        mc_dir=str(tmp_path),
    )

    assert fake_service.calls != []


def test_nbt_inplace_mode_still_translates_saves(
    tmp_path, fake_service, job_state, fake_callbacks
):
    """The saves/ skip is specific to resourcepack mode -- inplace mode
    (which writes a sidecar directly next to the source, unaffected by the
    resourcepack asset-loading restriction) must still work as before."""
    saves_dir = tmp_path / "saves" / "MyWorld"
    saves_dir.mkdir(parents=True)
    src = saves_dir / "level.dat"
    _make_nbt(src, {"greeting": "Hello there"})

    proc = NbtProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        str(src),
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="force",
        output_mode="inplace",
        mc_dir=str(tmp_path),
    )

    target = saves_dir / "level_ru_ru.dat"
    assert target.exists()
    assert str(nbtlib.load(str(target))["greeting"]) == "TR[Hello there]"


def test_nbt_seed_skips_path_that_became_a_container(
    tmp_path, fake_service, job_state, fake_callbacks
):
    """Regression test for the type-change corruption bug: if a path that
    was a flat String in a prior translated sidecar is now a nested Compound
    in the current source (the mod restructured that field between
    versions), seeding must NOT blindly overwrite the new Compound with the
    stale flat string."""
    src = tmp_path / "en_us.nbt"
    target = tmp_path / "ru_ru.nbt"
    # Prior translated sidecar: "data" was a flat string.
    nbtlib.File(tag.Compound({"data": tag.String("Старое значение")})).save(str(target))
    # Current source: "data" is now a nested Compound with its own string leaf.
    nbtlib.File(tag.Compound({"data": tag.Compound({"nested": tag.String("Nested value")})})).save(str(src))

    proc = NbtProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        str(src),
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="append",
        output_mode="inplace",
        mc_dir=str(tmp_path),
    )

    result = nbtlib.load(str(target))
    # The new nested structure must survive intact -- NOT get clobbered by
    # the stale flat "Старое значение" string, and its own leaf must be
    # translated like any other fresh string.
    assert isinstance(result["data"], tag.Compound)
    assert str(result["data"]["nested"]) == "TR[Nested value]"
