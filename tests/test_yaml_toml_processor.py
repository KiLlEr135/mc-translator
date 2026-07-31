"""Regression test for YamlTomlProcessor's append-mode value preservation --
the same class of bug fixed in generic_json.py (see test_generic_json_and_xml.py):

YamlTomlProcessor used to only ever write freshly-translated (`pairs`) values
into `data` before dumping; any already-translated path that wasn't part of
this run's `pairs` was never re-applied from `existing_map`, so a second run
in append/skip mode silently reverted previously-translated YAML/TOML strings
back to English.
"""
from mc_translator.processors.yaml_toml import YamlTomlProcessor


def test_yaml_second_run_preserves_prior_translations_and_translates_new_line(
    tmp_path, fake_service, job_state, fake_callbacks
):
    src = tmp_path / "en_us.yaml"
    src.write_text(
        "greeting: Hello there\nfarewell: Goodbye friend\n",
        encoding="utf-8",
    )
    target = tmp_path / "ru_ru.yaml"

    proc = YamlTomlProcessor(fake_service, job_state, fake_callbacks)
    target_lang = {"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"}

    # First run (force mode): both lines get translated.
    proc.process(
        str(src),
        target_lang=target_lang,
        mode="force",
        output_mode="inplace",
        mc_dir=str(tmp_path),
    )
    assert target.exists()
    first_result = target.read_text(encoding="utf-8")
    assert "TR[Hello there]" in first_result
    assert "TR[Goodbye friend]" in first_result

    # Source gets a new, third line before the second (append-mode) run.
    src.write_text(
        "greeting: Hello there\nfarewell: Goodbye friend\nwelcome: Welcome home\n",
        encoding="utf-8",
    )
    fake_service.calls.clear()

    proc.process(
        str(src),
        target_lang=target_lang,
        mode="append",
        output_mode="inplace",
        mc_dir=str(tmp_path),
    )

    second_result = target.read_text(encoding="utf-8")
    # Both previously-translated lines must survive, NOT revert to English.
    assert "TR[Hello there]" in second_result
    assert "TR[Goodbye friend]" in second_result
    # The brand-new line must be freshly translated.
    assert "TR[Welcome home]" in second_result
    # Only the brand-new string should have gone through translate_dict.
    translated_strings = {k: v for call, _ in fake_service.calls for k, v in call.items()}
    assert "Welcome home" in translated_strings.values()
    assert "Hello there" not in translated_strings.values()
    assert "Goodbye friend" not in translated_strings.values()


def test_yaml_list_indexed_paths_are_seeded_and_deduped_correctly(
    tmp_path, fake_service, job_state, fake_callbacks
):
    """Regression test for the list-index path-comparison bug: pairs_paths
    used to compare _walk_and_collect's real (str, int)-mixed path tuples
    against seed paths reconstructed as all-str from existing_map's
    "/"-joined key, so the membership check never matched for a
    list-indexed path -- a value could be queued for both seeding and fresh
    translation at once, with correctness resting entirely on undocumented
    order-of-application."""
    src = tmp_path / "en_us.yaml"
    src.write_text("items:\n  - First item\n  - Second item\n", encoding="utf-8")
    target = tmp_path / "ru_ru.yaml"

    proc = YamlTomlProcessor(fake_service, job_state, fake_callbacks)
    target_lang = {"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"}

    proc.process(
        str(src), target_lang=target_lang, mode="force",
        output_mode="inplace", mc_dir=str(tmp_path),
    )
    first_result = target.read_text(encoding="utf-8")
    assert "TR[First item]" in first_result
    assert "TR[Second item]" in first_result

    fake_service.calls.clear()
    proc.process(
        str(src), target_lang=target_lang, mode="append",
        output_mode="inplace", mc_dir=str(tmp_path),
    )
    second_result = target.read_text(encoding="utf-8")
    assert "TR[First item]" in second_result
    assert "TR[Second item]" in second_result
    assert fake_service.calls == []  # nothing new to translate -- no API calls


def test_yaml_seed_skips_path_that_became_a_nested_mapping(
    tmp_path, fake_service, job_state, fake_callbacks
):
    """Regression test for the type-change corruption bug: if a path that
    was a flat scalar in a prior translated sidecar is now a nested mapping
    in the current source (the mod restructured that field between
    versions), seeding must NOT blindly overwrite the new mapping with the
    stale flat string."""
    src = tmp_path / "en_us.yaml"
    target = tmp_path / "ru_ru.yaml"
    target.write_text("data: Old value\n", encoding="utf-8")
    src.write_text("data:\n  nested: Nested value\n", encoding="utf-8")

    proc = YamlTomlProcessor(fake_service, job_state, fake_callbacks)
    proc.process(
        str(src),
        target_lang={"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"},
        mode="append",
        output_mode="inplace",
        mc_dir=str(tmp_path),
    )

    result = target.read_text(encoding="utf-8")
    assert "TR[Nested value]" in result
    assert "Old value" not in result
    assert "nested:" in result  # "data" became a real nested mapping


def test_toml_second_run_preserves_prior_translations_and_translates_new_line(
    tmp_path, fake_service, job_state, fake_callbacks
):
    src = tmp_path / "en_us.toml"
    src.write_text(
        'greeting = "Hello there"\nfarewell = "Goodbye friend"\n',
        encoding="utf-8",
    )
    target = tmp_path / "ru_ru.toml"

    proc = YamlTomlProcessor(fake_service, job_state, fake_callbacks)
    target_lang = {"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"}

    # First run (force mode): both lines get translated.
    proc.process(
        str(src),
        target_lang=target_lang,
        mode="force",
        output_mode="inplace",
        mc_dir=str(tmp_path),
    )
    assert target.exists()
    first_result = target.read_text(encoding="utf-8")
    assert "TR[Hello there]" in first_result
    assert "TR[Goodbye friend]" in first_result

    # Source gets a new, third line before the second (append-mode) run.
    src.write_text(
        'greeting = "Hello there"\nfarewell = "Goodbye friend"\nwelcome = "Welcome home"\n',
        encoding="utf-8",
    )
    fake_service.calls.clear()

    proc.process(
        str(src),
        target_lang=target_lang,
        mode="append",
        output_mode="inplace",
        mc_dir=str(tmp_path),
    )

    second_result = target.read_text(encoding="utf-8")
    # Both previously-translated lines must survive, NOT revert to English.
    assert "TR[Hello there]" in second_result
    assert "TR[Goodbye friend]" in second_result
    # The brand-new line must be freshly translated.
    assert "TR[Welcome home]" in second_result
    # Only the brand-new string should have gone through translate_dict.
    translated_strings = {k: v for call, _ in fake_service.calls for k, v in call.items()}
    assert "Welcome home" in translated_strings.values()
    assert "Hello there" not in translated_strings.values()
    assert "Goodbye friend" not in translated_strings.values()
