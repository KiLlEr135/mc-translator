"""
Processor for archive files (.zip, .mcpack, .mcaddon, .litemod).
Treats them as zip archives, extracts, translates inner files, and repacks
(inplace mode) or merges the translated files into the combined output pack
(resourcepack mode).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import zipfile

from mc_translator.engines.base import EngineCallbacks
from mc_translator.engines.service import TranslationService
from mc_translator.processors.generic_json import GenericJsonProcessor
from mc_translator.processors.lang import LangProcessor
from mc_translator.processors.nbt import NbtProcessor
from mc_translator.processors.snbt import SnbtProcessor
from mc_translator.processors.text import TextProcessor
from mc_translator.processors.yaml_toml import YamlTomlProcessor


class ArchiveProcessor:
    def __init__(
        self,
        service: TranslationService,
        state: "JobState",
        callbacks: EngineCallbacks,
    ) -> None:
        self.service = service
        self.state = state
        self.callbacks = callbacks

    def _get_inner_processor(self, ext: str):
        """Return appropriate processor instance for inner file extension."""
        if ext == ".json":
            return GenericJsonProcessor(self.service, self.state, self.callbacks)
        if ext == ".lang":
            return LangProcessor(self.service, self.state, self.callbacks)
        if ext in {".nbt", ".dat"}:
            return NbtProcessor(self.service, self.state, self.callbacks)
        if ext in {".yaml", ".yml", ".toml"}:
            return YamlTomlProcessor(self.service, self.state, self.callbacks)
        if ext in {".txt", ".md"}:
            return TextProcessor(self.service, self.state, self.callbacks)
        if ext == ".snbt":
            return SnbtProcessor(self.service, self.state, self.callbacks)
        # Default: no processing
        return None

    def process(
        self,
        file_path: str,
        *,
        target_lang: dict,
        mode: str,
        output_mode: str,
        pack_writer: object | None = None,
        mc_dir: str | None = None,
    ) -> None:
        """
        Process a single archive file.

        Args:
            file_path: path to archive (.zip, .mcpack, .mcaddon, .litemod).
            target_lang: language dict.
            mode: translation mode.
            output_mode: output mode ('resourcepack' or 'inplace').
            pack_writer: PackWriter if output_mode is 'resourcepack'.
        """
        # Determine target archive path (keep same name) -- only used for the
        # inplace rebuild-and-replace path below.
        base_name = os.path.basename(file_path)
        target_path = os.path.join(os.path.dirname(file_path), base_name)

        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract archive
            MAX_UNCOMPRESSED_SIZE = 500 * 1024 * 1024  # 500 MB
            MAX_ENTRY_COUNT = 20000
            try:
                with zipfile.ZipFile(file_path, "r") as zf:
                    infos = zf.infolist()
                    total_size = sum(info.file_size for info in infos)
                    if len(infos) > MAX_ENTRY_COUNT or total_size > MAX_UNCOMPRESSED_SIZE:
                        self.callbacks.on_log(
                            f"❌ Архив {file_path} слишком большой после распаковки "
                            f"({total_size / (1024 * 1024):.0f} МБ, {len(infos)} файлов) — пропущен",
                            "red",
                        )
                        return
                    zf.extractall(tmpdir)
            except Exception as exc:
                self.callbacks.on_log(f"❌ Ошибка чтения архива {file_path}: {exc}", "red")
                return

            # Walk extracted files and process supported types
            for root, _, files in os.walk(tmpdir):
                if not self.state.should_run():
                    break
                for fname in files:
                    if not self.state.should_run():
                        break
                    self.state.wait_if_paused()
                    fpath = os.path.join(root, fname)
                    ext = os.path.splitext(fname)[1].lower()
                    processor = self._get_inner_processor(ext)
                    if processor is None:
                        continue
                    # Determine relative path within archive for logging
                    rel_path = os.path.relpath(fpath, tmpdir)
                    try:
                        # Process file in place (output_mode='inplace', no pack_writer).
                        # SnbtProcessor's signature has no output_mode/pack_writer params
                        # (it always writes in place), unlike every other inner processor.
                        if ext == ".snbt":
                            processor.process(fpath, target_lang=target_lang, mode=mode)
                        else:
                            processor.process(
                                fpath,
                                target_lang=target_lang,
                                mode=mode,
                                output_mode="inplace",
                                pack_writer=None,
                            )
                        self.callbacks.on_log(
                            f"📦 Архив: обработано {rel_path} ({ext})", "cyan"
                        )
                    except Exception as exc:
                        self.callbacks.on_log(
                            f"❌ Ошибка обработки {rel_path} в архиве: {exc}", "red"
                        )
                        continue

            if output_mode == "resourcepack" and pack_writer:
                # Merge every translated file INDIVIDUALLY into the combined
                # output pack, using each file's own path relative to the
                # archive root as its internal pack path -- NOT rebuilt into
                # one nested zip and written as a single blob. A valid
                # Minecraft resourcepack/datapack always has assets/ or
                # data/ at its OWN zip root, so the archive's internal
                # relative path is already exactly the shape PackWriter
                # expects, and PackWriter.handle_for_path already routes
                # "data/..." to the datapack zip and everything else to the
                # resourcepack zip -- the same overlay mechanism already
                # used for translated jar content (matching data paths so
                # Minecraft's normal pack-priority "last loaded wins" applies
                # the translation). A zip nested inside another zip at an
                # arbitrary internal path, by contrast, is never opened by
                # the game -- that was the bug this replaces.
                for root, _, files in os.walk(tmpdir):
                    if not self.state.should_run():
                        break
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        rel = os.path.relpath(fpath, tmpdir).replace(os.sep, "/")
                        low = rel.lower()
                        if low == "pack.mcmeta":
                            # PackWriter already wrote the OUTPUT pack's own
                            # pack.mcmeta directly at construction time, via
                            # a raw zipfile write that bypasses its own
                            # write()'s dedup tracking -- merging an inner
                            # archive's pack.mcmeta here would not be deduped
                            # and could duplicate/corrupt the pack metadata.
                            continue
                        if low.endswith(".bak"):
                            # One-time-snapshot artifact left behind by the
                            # inner SnbtProcessor while translating inside
                            # this tempdir -- never meant to ship.
                            continue
                        try:
                            with open(fpath, "rb") as f:
                                pack_writer.write(rel, f.read())
                        except OSError as exc:
                            self.callbacks.on_log(
                                f"❌ Ошибка записи {rel} из архива {base_name}: {exc}", "red"
                            )
            elif output_mode == "inplace":
                # Recreate archive. The rebuilt zip is written OUTSIDE tmpdir
                # (via tempfile.mkstemp) so the os.walk(tmpdir) below can't
                # see and embed the in-progress output file inside itself.
                try:
                    tmp_zip_fd, tmp_zip_path = tempfile.mkstemp(suffix=".zip")
                    os.close(tmp_zip_fd)
                    with zipfile.ZipFile(
                        tmp_zip_path, "w", compression=zipfile.ZIP_DEFLATED
                    ) as zip_out:
                        for root, _, files in os.walk(tmpdir):
                            for fname in files:
                                fpath = os.path.join(root, fname)
                                arcname = os.path.relpath(fpath, tmpdir)
                                zip_out.write(fpath, arcname)
                    shutil.move(tmp_zip_path, target_path)
                except Exception as exc:
                    self.callbacks.on_log(
                        f"❌ Ошибка создания архива {target_path}: {exc}", "red"
                    )