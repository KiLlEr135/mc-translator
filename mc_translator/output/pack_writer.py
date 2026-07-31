import io
import json
import os
import re
import threading
import zipfile

from mc_translator.constants import PACK_FORMATS
from mc_translator.utils.atomic_write import atomic_write
from mc_translator.utils.legacy_lang import is_legacy_version, legacy_lang_filename


def expected_pack_paths(mc_dir: str, pack_name: str | None) -> tuple[str, str]:
    """Compute the (resourcepack_zip_path, datapack_zip_path) a PackWriter
    for `pack_name` would create/reuse -- same sanitization PackWriter.
    __init__ applies, factored out so ModpackAnalyzer can locate a PRIOR
    run's actual output (to check resourcepack-mode readiness) without
    constructing a real PackWriter. Doesn't chase any collision-suffix
    fallback -- if the plain path isn't there, the analyzer just finds no
    pack to check and falls back to its old (source-file-only) behavior."""
    safe_name = re.sub(r'[\\/*?:"<>|]', "", (pack_name or "").strip() or "MCTranslator_Pack")
    if not safe_name.lower().endswith(".zip"):
        safe_name += ".zip"
    rp_path = os.path.join(mc_dir, "resourcepacks", safe_name)
    dp_name = safe_name.replace(".zip", "_Datapack.zip")
    dp_path = os.path.join(mc_dir, "config", "openloader", "data", dp_name)
    return rp_path, dp_path


class PackWriter:
    """Creates resource pack and datapack zip handles for translated assets.

    Every entry gets buffered in memory (self._rp_entries/self._dp_entries)
    and the real zip files are only written once, atomically, in close() --
    seeded from whatever a PRIOR run already produced at the same path.
    Without this, a run that stops early (Stop clicked, a crash, the app
    closed, or even just OpenRouter timing out mid-run) would silently
    regress the pack back to only-what-this-run-reached, discarding
    everything a previous complete run had written: confirmed on a real
    pack where an interrupted run left only 6 files in the resourcepack,
    wiping out books/guides/interface text a prior 6h45m run had already
    produced (the actual translated strings survived independently in
    TranslationCache/ai_cache.json, but the pack itself had no memory of
    them). Entries this run's processors actually call write() for still
    get freshly overwritten -- only entries this run never touches at all
    survive untouched from the previous pack."""

    def __init__(
        self,
        mc_dir: str,
        pack_base_name: str,
        mc_version: str,
        lang_name: str,
    ) -> None:
        self.mc_dir = mc_dir
        # Paths actually written THIS run (as opposed to merely seeded from
        # a previous pack) -- write()'s dedup guard uses this, not whether
        # the entry dict already has the key, so a fresh write always
        # replaces a stale seeded value on its first call this run, while a
        # second call this run for the same path is still silently dropped
        # (documented, intentional -- see test_duplicate_path_is_dropped_silently).
        self.written: set[str] = set()
        # write() does an unlocked read-modify-write on self.written/the
        # entry dicts, which is not thread-safe on its own -- needed once
        # job.py can process multiple files concurrently (OpenRouter runs).
        self._lock = threading.Lock()
        fmt = PACK_FORMATS.get(mc_version, PACK_FORMATS["1.21.1"])

        rp_dir = os.path.join(mc_dir, "resourcepacks")
        dp_dir = os.path.join(mc_dir, "config", "openloader", "data")
        os.makedirs(rp_dir, exist_ok=True)
        os.makedirs(dp_dir, exist_ok=True)

        expected_rp_path, _ = expected_pack_paths(mc_dir, pack_base_name)
        safe_name = os.path.basename(expected_rp_path)

        # Version-specific pathing for legacy
        self.is_legacy = is_legacy_version(mc_version)

        self.rp_zip_path = os.path.join(rp_dir, safe_name)
        dp_name = safe_name.replace(".zip", "_Datapack.zip")
        self.dp_zip_path = os.path.join(dp_dir, dp_name)

        self._rp_entries: dict[str, bytes] = self._read_existing_entries(self.rp_zip_path)
        self._dp_entries: dict[str, bytes] = self._read_existing_entries(self.dp_zip_path)
        self._rp_meta = {"pack_format": fmt["rp"], "description": f"{safe_name} - MC Translator"}
        self._dp_meta = {"pack_format": fmt["dp"], "description": f"{dp_name} - MC Translator"}

    @staticmethod
    def _read_existing_entries(path: str) -> dict[str, bytes]:
        """Reads every non-metadata entry out of a pack a PRIOR run already
        wrote at `path`, so a fresh PackWriter can seed its output from it
        instead of starting empty. Tolerates a missing or corrupted zip
        (e.g. a past hard-kill mid-write, seen for real as "File is not a
        zip file" in this project's own logs) by returning {} -- same
        fail-safe philosophy as analyzer.py's _PackIndex.read_json."""
        if not os.path.exists(path):
            return {}
        try:
            with zipfile.ZipFile(path) as zf:
                entries: dict[str, bytes] = {}
                for info in zf.infolist():
                    if info.is_dir() or info.filename == "pack.mcmeta":
                        continue
                    try:
                        entries[info.filename] = zf.read(info.filename)
                    except (zipfile.BadZipFile, KeyError, OSError):
                        continue
                return entries
        except (zipfile.BadZipFile, OSError):
            return {}

    def handle_for_path(self, internal_path: str) -> dict[str, bytes] | None:
        if internal_path.lower().startswith("data/"):
            return self._dp_entries
        return self._rp_entries

    def write(self, internal_path: str, data: bytes) -> None:
        target = self.handle_for_path(internal_path)
        if target is None:
            return

        # Legacy version adjustment: Fix paths and names
        if self.is_legacy:
            # 1.12- often uses assets/modid/lang/en_US.lang
            # 1.13+ uses assets/modid/lang/en_us.json
            if internal_path.endswith(".json") and "/lang/" in internal_path:
                # Renaming the extension alone used to leave JSON *content*
                # under a .lang name -- 1.12- parses .lang as plain
                # key=value lines, not JSON, so the game failed to load it
                # and the translation silently never applied. Convert the
                # payload too; if it isn't a flat {key: value} JSON object
                # (unexpected shape), leave the bytes alone rather than
                # guessing.
                converted = self._json_lang_to_legacy(data)
                if converted is not None:
                    data = converted
                internal_path = internal_path[: -len(".json")] + ".lang"

            # Case sensitivity in legacy: ru_ru -> ru_RU
            if "/lang/" in internal_path:
                parts = internal_path.split("/")
                parts[-1] = legacy_lang_filename(parts[-1])
                internal_path = "/".join(parts)

        with self._lock:
            if internal_path not in self.written:
                target[internal_path] = data
                self.written.add(internal_path)

    @staticmethod
    def _json_lang_to_legacy(data: bytes) -> bytes | None:
        """Convert a flat {key: "value"} lang JSON payload to legacy
        key=value line format. Returns None (leave data untouched) if it
        isn't that shape."""
        try:
            obj = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(obj, dict):
            return None
        lines = []
        for key, value in obj.items():
            if not isinstance(value, str):
                continue
            # .lang has no escaping convention for embedded newlines; drop
            # them to keep each entry on its own line rather than producing
            # a malformed file.
            flat_value = value.replace("\r\n", " ").replace("\n", " ")
            lines.append(f"{key}={flat_value}")
        return ("\n".join(lines) + "\n").encode("utf-8")

    @staticmethod
    def _build_zip_bytes(meta: dict, entries: dict[str, bytes]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("pack.mcmeta", json.dumps({"pack": meta}, indent=2))
            for name, data in entries.items():
                zf.writestr(name, data)
        return buf.getvalue()

    def close(self) -> tuple[str | None, str | None]:
        atomic_write(self.rp_zip_path, self._build_zip_bytes(self._rp_meta, self._rp_entries))
        atomic_write(self.dp_zip_path, self._build_zip_bytes(self._dp_meta, self._dp_entries))
        return self.rp_zip_path, self.dp_zip_path
