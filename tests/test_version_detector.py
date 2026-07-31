"""Tests for mc_translator.utils.version_detector.detect_mc_version.

Regression coverage for a real bug: Prism Launcher/MultiMC store
per-instance metadata (instance.cfg, mmc-pack.json) in the INSTANCE ROOT,
one directory level above the actual game directory
(".../instances/<name>/minecraft") that mc_dir is correctly set to for
translation purposes. Looking only inside mc_dir itself (the old behavior)
silently found nothing for every Prism/MultiMC instance. Modern Prism
instance.cfg files also no longer carry an intendedVersion= key at all --
the real version now only lives in mmc-pack.json's "net.minecraft"
component.
"""
import json
import zipfile

from mc_translator.utils.version_detector import detect_mc_version


def test_detects_version_from_parent_mmc_pack_json(tmp_path):
    """Modern Prism Launcher layout: mc_dir is the "minecraft" subfolder;
    mmc-pack.json (with no matching instance.cfg field) lives in the parent
    instance root."""
    instance_root = tmp_path / "All the Mods 10 - ATM10"
    mc_dir = instance_root / "minecraft"
    mc_dir.mkdir(parents=True)
    (instance_root / "mmc-pack.json").write_text(
        json.dumps(
            {
                "components": [
                    {"uid": "org.lwjgl3", "version": "3.3.3"},
                    {"uid": "net.minecraft", "version": "1.21.1", "cachedVersion": "1.21.1"},
                    {"uid": "net.neoforged", "version": "21.1.100"},
                ]
            }
        ),
        encoding="utf-8",
    )
    # Modern instance.cfg has no intendedVersion= key at all.
    (instance_root / "instance.cfg").write_text(
        "[General]\nManagedPackVersionName=7.1\nname=All the Mods 10 - ATM10\n", encoding="utf-8"
    )

    assert detect_mc_version(str(mc_dir)) == "1.21.1"


def test_detects_version_from_parent_legacy_instance_cfg(tmp_path):
    """Classic MultiMC-style instance.cfg with an intendedVersion= field,
    also one directory level above mc_dir, no mmc-pack.json present."""
    instance_root = tmp_path / "SomeOldPack"
    mc_dir = instance_root / "minecraft"
    mc_dir.mkdir(parents=True)
    (instance_root / "instance.cfg").write_text(
        "[General]\nintendedVersion=1.16.5\nname=SomeOldPack\n", encoding="utf-8"
    )

    assert detect_mc_version(str(mc_dir)) == "1.16.5"


def test_detects_version_from_curseforge_minecraftinstance_json_in_mc_dir(tmp_path):
    """CurseForge App instances keep minecraftinstance.json directly
    alongside mods/config in the same directory the user selects."""
    mc_dir = tmp_path / "curseforge_pack"
    mc_dir.mkdir()
    (mc_dir / "minecraftinstance.json").write_text(
        json.dumps({"gameVersion": "1.20.1"}), encoding="utf-8"
    )

    assert detect_mc_version(str(mc_dir)) == "1.20.1"


def test_returns_none_when_no_metadata_found(tmp_path):
    mc_dir = tmp_path / "empty_folder"
    mc_dir.mkdir()
    assert detect_mc_version(str(mc_dir)) is None


def test_returns_none_for_nonexistent_directory(tmp_path):
    assert detect_mc_version(str(tmp_path / "does_not_exist")) is None


def test_mmc_pack_json_without_minecraft_component_falls_through(tmp_path):
    """A malformed/unusual mmc-pack.json with no net.minecraft component
    must not crash -- just fall through to the next detection method."""
    instance_root = tmp_path / "WeirdPack"
    mc_dir = instance_root / "minecraft"
    mc_dir.mkdir(parents=True)
    (instance_root / "mmc-pack.json").write_text(
        json.dumps({"components": [{"uid": "org.lwjgl3", "version": "3.3.3"}]}), encoding="utf-8"
    )
    (instance_root / "instance.cfg").write_text(
        "[General]\nintendedVersion=1.18.2\n", encoding="utf-8"
    )

    assert detect_mc_version(str(mc_dir)) == "1.18.2"


def test_detects_version_from_atlauncher_instance_json(tmp_path):
    """ATLauncher has no root/minecraft split -- instance.json lives
    directly alongside mods/config. The top-level "id" field is the MC
    version (confirmed via Instance.getMinecraftVersion()); the nested
    launcher.version field is a trap -- it's the MODPACK's own version."""
    mc_dir = tmp_path / "atlauncher_pack"
    mc_dir.mkdir()
    (mc_dir / "instance.json").write_text(
        json.dumps(
            {
                "id": "1.20.1",
                "launcher": {"name": "My Modpack", "pack": "My Modpack", "version": "1.4.2"},
            }
        ),
        encoding="utf-8",
    )

    assert detect_mc_version(str(mc_dir)) == "1.20.1"


def test_detects_version_from_technic_bin_version_json_inherits_from(tmp_path):
    """Technic's bin/version.json prefers "inheritsFrom" over the composite
    "id" field (e.g. "1.12.2-forge1.12.2-14.23.5.2860")."""
    mc_dir = tmp_path / "technic_pack"
    bin_dir = mc_dir / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "version.json").write_text(
        json.dumps(
            {
                "id": "1.12.2-forge1.12.2-14.23.5.2860",
                "type": "release",
                "inheritsFrom": "1.12.2",
            }
        ),
        encoding="utf-8",
    )

    assert detect_mc_version(str(mc_dir)) == "1.12.2"


def test_detects_version_from_technic_bin_version_json_composite_id_fallback(tmp_path):
    """Without inheritsFrom, the leading version prefix is extracted from
    the composite "id" string (Technic's ids are consistently
    Forge-style/version-first)."""
    mc_dir = tmp_path / "technic_pack_no_inherits"
    bin_dir = mc_dir / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "version.json").write_text(
        json.dumps({"id": "1.16.5-forge-36.2.39", "type": "release"}), encoding="utf-8"
    )

    assert detect_mc_version(str(mc_dir)) == "1.16.5"


def test_detects_version_from_technic_modpack_jar_zip_fallback(tmp_path):
    """When bin/version.json hasn't been cached yet, read the internal
    version.json entry out of bin/modpack.jar (a zip), matching Technic's
    own ZipMinecraftVersionInfoRetriever."""
    mc_dir = tmp_path / "technic_pack_jar_only"
    bin_dir = mc_dir / "bin"
    bin_dir.mkdir(parents=True)
    with zipfile.ZipFile(bin_dir / "modpack.jar", "w") as zf:
        zf.writestr("version.json", json.dumps({"id": "1.7.10", "inheritsFrom": "1.7.10"}))

    assert detect_mc_version(str(mc_dir)) == "1.7.10"


def test_technic_ignores_bin_version_and_installed_packs_traps(tmp_path):
    """bin/version (no extension) and the central installedPacks file hold
    the modpack's own build id, never the Minecraft version -- must not be
    mistaken for a version source."""
    mc_dir = tmp_path / "technic_pack_traps"
    bin_dir = mc_dir / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "version").write_text(json.dumps({"version": "1.4.5", "legacy": False}), encoding="utf-8")

    assert detect_mc_version(str(mc_dir)) is None


def test_detects_version_from_vanilla_launcher_profiles_clean_version(tmp_path):
    """Plain vanilla launcher_profiles.json: picks the profile with the max
    lastUsed timestamp and its lastVersionId is already a bare version."""
    mc_dir = tmp_path / "dotminecraft"
    mc_dir.mkdir()
    (mc_dir / "launcher_profiles.json").write_text(
        json.dumps(
            {
                "profiles": {
                    "(Default)": {
                        "name": "(Default)",
                        "lastVersionId": "1.12.2",
                        "lastUsed": "2020-01-01T00:00:00.000Z",
                    },
                    "Saved": {
                        "name": "Saved",
                        "lastVersionId": "1.20.1",
                        "lastUsed": "2024-06-12T13:25:51.000Z",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    assert detect_mc_version(str(mc_dir)) == "1.20.1"


def test_detects_version_from_vanilla_launcher_profiles_resolves_forge_profile(tmp_path):
    """A modloader lastVersionId (e.g. Forge) is resolved via that version's
    own versions/<id>/<id>.json "inheritsFrom" field -- never guessed from
    the id string's shape."""
    mc_dir = tmp_path / "dotminecraft_forge"
    mc_dir.mkdir()
    (mc_dir / "launcher_profiles.json").write_text(
        json.dumps(
            {
                "profiles": {
                    "forge": {
                        "name": "forge",
                        "lastVersionId": "1.20.1-forge-47.2.0",
                        "lastUsed": "2024-06-12T13:25:51.000Z",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    version_dir = mc_dir / "versions" / "1.20.1-forge-47.2.0"
    version_dir.mkdir(parents=True)
    (version_dir / "1.20.1-forge-47.2.0.json").write_text(
        json.dumps({"id": "1.20.1-forge-47.2.0", "inheritsFrom": "1.20.1", "type": "release"}),
        encoding="utf-8",
    )

    assert detect_mc_version(str(mc_dir)) == "1.20.1"


def test_vanilla_launcher_profiles_unresolvable_modloader_id_returns_none(tmp_path):
    """If the versions/<id>/<id>.json file is missing entirely, the
    detector must not guess from the id string -- it returns None rather
    than risk a wrong pack_format."""
    mc_dir = tmp_path / "dotminecraft_unresolvable"
    mc_dir.mkdir()
    (mc_dir / "launcher_profiles.json").write_text(
        json.dumps(
            {
                "profiles": {
                    "custom": {
                        "name": "custom",
                        "lastVersionId": "fabric-loader-0.15.7-1.20.1",
                        "lastUsed": "2024-06-12T13:25:51.000Z",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert detect_mc_version(str(mc_dir)) is None


def test_detects_version_from_gdlauncher_carbon_instance_json_object_form(tmp_path):
    """GDLauncher Carbon splits instance root (holds instance.json) from
    the "instance" subfolder (the actual mods/config game dir) -- same
    root/game-dir split as Prism/MultiMC. version.release is the MC
    version; _version is a config SCHEMA version and must be ignored."""
    instance_root = tmp_path / "My Modpack"
    mc_dir = instance_root / "instance"
    mc_dir.mkdir(parents=True)
    (instance_root / "instance.json").write_text(
        json.dumps(
            {
                "_version": "1",
                "name": "My Modpack",
                "game_configuration": {
                    "version": {"release": "1.20.1", "modloaders": [{"type": "forge", "version": "47.2.0"}]}
                },
            }
        ),
        encoding="utf-8",
    )

    assert detect_mc_version(str(mc_dir)) == "1.20.1"


def test_detects_version_from_gdlauncher_carbon_instance_json_string_form(tmp_path):
    """The untagged serde enum's rare "Custom" variant stores the version
    as a bare string directly at game_configuration.version."""
    instance_root = tmp_path / "Custom Modpack"
    mc_dir = instance_root / "instance"
    mc_dir.mkdir(parents=True)
    (instance_root / "instance.json").write_text(
        json.dumps({"_version": "1", "game_configuration": {"version": "1.16.5"}}),
        encoding="utf-8",
    )

    assert detect_mc_version(str(mc_dir)) == "1.16.5"


def test_detects_version_from_gdlauncher_legacy_config_json(tmp_path):
    """Legacy (discontinued Electron) GDLauncher has no root/game-dir
    split -- config.json sits directly alongside mods/, with the MC
    version at loader.mcVersion."""
    mc_dir = tmp_path / "gdlauncher_legacy_pack"
    mc_dir.mkdir()
    (mc_dir / "config.json").write_text(
        json.dumps(
            {
                "loader": {
                    "loaderType": "forge",
                    "mcVersion": "1.12.2",
                    "loaderVersion": "14.23.5.2860",
                },
                "timePlayed": 3600,
            }
        ),
        encoding="utf-8",
    )

    assert detect_mc_version(str(mc_dir)) == "1.12.2"


def test_detects_version_from_microsoft_store_launcher_profiles(tmp_path):
    """Microsoft Store build uses launcher_profiles_microsoft_store.json
    instead, with the identical profiles/lastVersionId schema."""
    mc_dir = tmp_path / "dotminecraft_msstore"
    mc_dir.mkdir()
    (mc_dir / "launcher_profiles_microsoft_store.json").write_text(
        json.dumps(
            {
                "profiles": {
                    "(Default)": {
                        "name": "(Default)",
                        "lastVersionId": "1.19.2",
                        "lastUsed": "2023-01-01T00:00:00.000Z",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert detect_mc_version(str(mc_dir)) == "1.19.2"
