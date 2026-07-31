import json
import os
import re
import zipfile

_CLEAN_VERSION_RE = re.compile(r"^\d+\.\d+(?:\.\d+)?$")


def detect_mc_version(mc_dir: str) -> str | None:
    """
    Tries to detect the Minecraft version of a given instance directory.
    Returns version string (e.g., "1.20.1") or None if not found.

    Checks mc_dir itself AND its parent directory: Prism Launcher/MultiMC
    store per-instance metadata (instance.cfg, mmc-pack.json) in the
    INSTANCE ROOT (".../instances/<name>/"), one level above the actual
    game directory (".../instances/<name>/minecraft") that mc_dir is
    correctly set to for translation purposes -- mods/config/resourcepacks
    all live there, not in the instance root. Looking only inside mc_dir
    (the old behavior) silently found nothing for every Prism/MultiMC
    instance, since the metadata files were always one level up.

    Supported launchers, in check order: Prism Launcher/MultiMC
    (mmc-pack.json, instance.cfg), CurseForge App (minecraftinstance.json),
    ATLauncher (instance.json), Technic Launcher (bin/version.json or
    bin/modpack.jar), vanilla Minecraft Launcher (launcher_profiles.json),
    GDLauncher Carbon (instance.json) and GDLauncher Legacy (config.json).
    Every field used here was verified against real launcher source code or
    docs, not guessed -- see tests/test_version_detector.py for the exact
    field names and layouts each check depends on.
    """
    if not mc_dir or not os.path.isdir(mc_dir):
        return None

    candidates = [mc_dir]
    parent = os.path.dirname(os.path.normpath(mc_dir))
    if parent and parent != mc_dir:
        candidates.append(parent)

    for base in candidates:
        # 1. Prism Launcher/MultiMC mmc-pack.json -- the modern, structured
        # source: a "net.minecraft" component carries the real MC version.
        # Newer Prism instance.cfg files no longer carry an intendedVersion=
        # key at all (that's a legacy MultiMC-era field), so this must be
        # checked first/independently, not as a fallback from instance.cfg.
        pack_json = os.path.join(base, "mmc-pack.json")
        if os.path.exists(pack_json):
            try:
                with open(pack_json, encoding="utf-8") as f:
                    data = json.load(f)
                for component in data.get("components", []):
                    if component.get("uid") == "net.minecraft":
                        ver = component.get("version") or component.get("cachedVersion")
                        if ver:
                            return _normalize_version(ver)
            except Exception:
                pass

        # 2. Prism/MultiMC instance.cfg -- legacy field, still present on
        # older/classic instances.
        cfg_path = os.path.join(base, "instance.cfg")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, errors="ignore") as f:
                    content = f.read()
                    m = re.search(r"intendedVersion=(.*)", content)
                    if m and m.group(1).strip():
                        return _normalize_version(m.group(1).strip())
            except Exception:
                pass

        # 3. CurseForge App minecraftinstance.json (lives alongside
        # mods/config in the same directory the user selects, unlike
        # Prism/MultiMC's split root-vs-minecraft-subfolder layout).
        cf_json = os.path.join(base, "minecraftinstance.json")
        if os.path.exists(cf_json):
            try:
                with open(cf_json) as f:
                    data = json.load(f)
                    ver = data.get("gameVersion")
                    if ver:
                        return _normalize_version(ver)
            except Exception:
                pass

        # 4. ATLauncher instance.json -- lives directly alongside mods/config
        # (no root/minecraft split like Prism/MultiMC). The top-level "id"
        # field is the Minecraft version (confirmed via ATLauncher's own
        # Instance.getMinecraftVersion(), which returns exactly this field).
        # Do NOT use launcher.version -- that's the MODPACK's own version
        # string (e.g. "1.4.2"), a plausible-looking trap.
        at_json = os.path.join(base, "instance.json")
        if os.path.exists(at_json):
            try:
                with open(at_json, encoding="utf-8") as f:
                    data = json.load(f)
                ver = data.get("id")
                if ver:
                    return _normalize_version(ver)
            except Exception:
                pass

        # 5. Technic Launcher bin/version.json -- prefer "inheritsFrom" (the
        # clean base MC version written for Forge/NeoForge/Fabric profiles)
        # over "id", which is often a composite string like
        # "1.12.2-forge1.12.2-14.23.5.2860". If bin/version.json hasn't been
        # cached yet, fall back to reading the same version.json entry out
        # of bin/modpack.jar (a zip), exactly like Technic's own
        # ZipMinecraftVersionInfoRetriever does. Deliberately never reads
        # bin/version (no extension) or the central installedPacks file --
        # both are real files but hold the modpack's own build id, not the
        # Minecraft version.
        tech_data = None
        tech_json = os.path.join(base, "bin", "version.json")
        if os.path.exists(tech_json):
            try:
                with open(tech_json, encoding="utf-8") as f:
                    tech_data = json.load(f)
            except Exception:
                tech_data = None
        else:
            modpack_jar = os.path.join(base, "bin", "modpack.jar")
            if os.path.exists(modpack_jar):
                try:
                    with zipfile.ZipFile(modpack_jar) as zf:
                        with zf.open("version.json") as f:
                            tech_data = json.load(f)
                except Exception:
                    tech_data = None
        if tech_data:
            ver = tech_data.get("inheritsFrom") or tech_data.get("id")
            if ver:
                clean = ver if _CLEAN_VERSION_RE.match(ver) else _leading_version(ver)
                if clean:
                    return _normalize_version(clean)

        # 6. Vanilla Minecraft Launcher launcher_profiles.json (or the
        # Microsoft Store build's launcher_profiles_microsoft_store.json) --
        # the weakest signal here, since a plain .minecraft folder has one
        # shared mods/ folder not namespaced by version. Picks the profile
        # with the most recent "lastUsed" ISO-8601 timestamp (these compare
        # correctly as plain strings) as the best guess of "the version this
        # folder is about". If lastVersionId isn't already a bare "x.y.z"
        # string (e.g. a Forge/Fabric profile id), it's resolved via that
        # version's own versions/<id>/<id>.json "inheritsFrom" field --
        # never guessed from the id string's shape.
        for profiles_name in ("launcher_profiles.json", "launcher_profiles_microsoft_store.json"):
            prof_path = os.path.join(base, profiles_name)
            if not os.path.exists(prof_path):
                continue
            try:
                with open(prof_path, encoding="utf-8") as f:
                    data = json.load(f)
                best_id = None
                best_used = ""
                for profile in data.get("profiles", {}).values():
                    version_id = profile.get("lastVersionId")
                    if not version_id:
                        continue
                    used = profile.get("lastUsed") or profile.get("created") or ""
                    if best_id is None or used >= best_used:
                        best_used = used
                        best_id = version_id
                if best_id:
                    if _CLEAN_VERSION_RE.match(best_id):
                        return _normalize_version(best_id)
                    resolved = _resolve_vanilla_version_id(base, best_id)
                    if resolved:
                        return _normalize_version(resolved)
            except Exception:
                pass

        # 7. GDLauncher Carbon instance.json -- the CURRENT GDLauncher
        # generation (Rust core, actively maintained). Sits in the instance
        # ROOT, a sibling of the "instance" subfolder that actually holds
        # mods/saves/config -- the same root/game-dir split as Prism/MultiMC,
        # which is why checking `parent` (already done via `candidates`)
        # matters here too. The version field is a serde untagged enum:
        # either an object {"release": "...", "modloaders": [...]} or,
        # rarely, a bare version string -- both are handled. Do NOT use the
        # top-level "_version" field: that's a config SCHEMA version
        # (currently the literal "1"), not a Minecraft version. This file
        # name collides with ATLauncher's instance.json (step 4 above), but
        # the two schemas don't overlap ("id" vs "game_configuration"), so
        # checking both in sequence is safe.
        carbon_json = os.path.join(base, "instance.json")
        if os.path.exists(carbon_json):
            try:
                with open(carbon_json, encoding="utf-8") as f:
                    data = json.load(f)
                version_field = data.get("game_configuration", {}).get("version")
                ver = None
                if isinstance(version_field, dict):
                    ver = version_field.get("release")
                elif isinstance(version_field, str):
                    ver = version_field
                if ver:
                    return _normalize_version(ver)
            except Exception:
                pass

        # 8. GDLauncher Legacy (discontinued Electron app) config.json --
        # lives directly alongside mods/resources, no root/game-dir split.
        legacy_json = os.path.join(base, "config.json")
        if os.path.exists(legacy_json):
            try:
                with open(legacy_json, encoding="utf-8") as f:
                    data = json.load(f)
                ver = data.get("loader", {}).get("mcVersion")
                if ver:
                    return _normalize_version(ver)
            except Exception:
                pass

    return None


def _leading_version(raw: str) -> str | None:
    """Extracts a leading "x.y(.z)" version prefix from a composite id like
    Technic's "1.12.2-forge1.12.2-14.23.5.2860". Only matches at the START
    of the string -- Technic's composite ids are consistently Forge-style
    (version-first); this is not reused for vanilla/Fabric profile ids,
    which are resolved via the authoritative inheritsFrom field instead
    (see _resolve_vanilla_version_id) rather than guessed from id shape."""
    m = re.match(r"^(\d+\.\d+(?:\.\d+)?)", raw)
    return m.group(1) if m else None


def _resolve_vanilla_version_id(base: str, version_id: str) -> str | None:
    """Resolves a modloader-composite lastVersionId (e.g.
    "1.20.1-forge-47.2.0") to its underlying Minecraft version via that
    version's own versions/<id>/<id>.json, which Forge/NeoForge/OptiFine
    installers write with an "inheritsFrom" field pointing at the base MC
    version. Returns None (never guesses from the id string) if that file
    or a clean version field isn't present -- an unresolved profile should
    fall back to no auto-detection rather than risk a wrong pack_format."""
    version_json = os.path.join(base, "versions", version_id, f"{version_id}.json")
    if not os.path.exists(version_json):
        return None
    try:
        with open(version_json, encoding="utf-8") as f:
            data = json.load(f)
        inherits = data.get("inheritsFrom")
        if inherits and _CLEAN_VERSION_RE.match(inherits):
            return inherits
        own_id = data.get("id")
        if own_id and _CLEAN_VERSION_RE.match(own_id):
            return own_id
    except Exception:
        pass
    return None

def _normalize_version(ver: str) -> str:
    """Maps a detected version to one of MC Translator's supported
    MC_VERSIONS -- but only when the two share the exact same resource/data
    pack_format (verified against the Minecraft Wiki's pack format history
    table), never onto a merely "nearby" version. A wrong pack_format in
    pack.mcmeta makes the game flag the generated pack as "made for a
    different version"; sub-versions that don't share a format with any
    supported entry (e.g. 1.19.3, 1.20.2 previously fell through to
    1.19.2/1.20.1 respectively) are collapsed onto their own dedicated
    PACK_FORMATS entry instead of a wrong neighbor's.
    """
    if ver.startswith("1.7"): return "1.7.10"
    if ver.startswith("1.8"): return "1.8.9"
    if ver.startswith("1.10"): return "1.10.2"
    if ver.startswith("1.12"): return "1.12.2"
    if ver.startswith("1.16"): return "1.16.5"
    if ver.startswith("1.18"): return "1.18.2"
    if ver.startswith("1.19"):
        if ver == "1.19.4": return "1.19.4"
        if ver == "1.19.3": return "1.19.3"  # distinct format, not 1.19.2's
        return "1.19.2"  # covers 1.19, 1.19.1, 1.19.2 (identical format)
    if ver.startswith("1.20"):
        if ver == "1.20.2": return "1.20.2"  # distinct format
        if ver in ("1.20.3", "1.20.4"): return "1.20.4"  # share one format
        if ver in ("1.20.5", "1.20.6"): return "1.20.6"  # share one format
        return "1.20.1"  # covers 1.20, 1.20.1
    if ver.startswith("1.21"):
        parts = tuple(int(p) for p in ver.split(".") if p.isdigit())
        if parts >= (1, 21, 4): return "1.21.4"
        if parts >= (1, 21, 2): return "1.21.3"  # 1.21.2 shares 1.21.3's format
        return "1.21.1"  # covers 1.21, 1.21.1

    return ver
