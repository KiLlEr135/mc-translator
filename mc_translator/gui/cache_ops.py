"""Cache-clear/scope JS-callable methods -- split out of gui/bridge.py
purely to keep that file under the project's 500-line guideline.
CacheOpsMixin is mixed into Api (bridge.py), which still owns
self.job_state/self.cache_std/self.cache_ai this mixin reads/writes; this
is pure code motion, not a behavior change."""
from __future__ import annotations

import os
import shutil
from datetime import datetime

from mc_translator.config import settings
from mc_translator.constants import LANGUAGES
from mc_translator.usage_log import load_usage_log, save_usage_log


class CacheOpsMixin:
    def get_cache_scopes(self, scope_all_langs: bool, language_label: str = "") -> dict:
        """Feeds the cache-clear modal's mod/type checklist. Reads the usage
        log(s) (never the cache itself, which has no metadata) for either the
        currently-selected language or every known language, and returns the
        set of distinct mod names and file 'kinds' seen -- exactly what a
        selective clear_cache(scope=...) call below can target.

        `language_label` is the language dropdown's CURRENT value, passed
        explicitly by app.js. SESSION.language is only written when a
        run actually starts (_save_session_settings), so falling back to it
        here (as this used to do unconditionally) served the wrong language's
        checklist whenever the user switched the dropdown without starting a
        run first."""
        mc_dir = settings.get("GENERAL", "mc_dir")
        if not mc_dir:
            return {"mods": [], "kinds": []}
        selected = LANGUAGES.get(language_label) or LANGUAGES.get(settings.get("SESSION", "language"))
        if scope_all_langs:
            langs = LANGUAGES.values()
        elif selected:
            langs = [selected]
        else:
            langs = []
        mods: set[str] = set()
        kinds: set[str] = set()
        for lang in langs:
            usage_log = load_usage_log(mc_dir, lang["file"])
            for entry in usage_log.values():
                mods.update(entry.get("mods", []))
                kind = entry.get("kind")
                if kind:
                    kinds.add(kind)
        return {"mods": sorted(mods), "kinds": sorted(kinds)}

    def clear_cache(self, scope: dict | None = None) -> dict:
        """scope=None (or {"mode":"all"}) wipes both cache files entirely --
        the original behavior, backup-then-clear. scope={"mode":"mods"|"kinds",
        "values":[...], "languages":[...]|"all"} instead removes only cache
        entries belonging to the given mods/file-kinds, for the given
        language(s) -- everything else in the cache is left untouched.

        Refused while a job is running: TranslationJob holds its own
        in-memory usage_log snapshot for the whole run and only saves it
        once at the end, so a concurrent clear here would get silently
        undone by the job's final save (and a full clear would wipe cache
        entries the running job already produced earlier in the same run)."""
        if self.job_state.is_running:
            return {"ok": False, "errors": ["Дождитесь завершения текущей операции перед очисткой кэша."], "removed": 0}
        mode = (scope or {}).get("mode", "all")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        errors: list[str] = []

        if mode == "all":
            # Only clear a cache whose backup actually succeeded (or that
            # had no file to begin with) -- clearing unconditionally used
            # to silently discard a cache's in-memory data with NO backup
            # whenever its own os.replace() happened to fail, permanently
            # losing every previously paid-for translation in it the next
            # time save() ran.
            for cache in (self.cache_std, self.cache_ai):
                path = cache.filepath
                if os.path.exists(path):
                    backup_path = f"{path}.bak-{timestamp}"
                    try:
                        os.replace(path, backup_path)
                    except OSError as exc:
                        errors.append(f"Не удалось создать резервную копию: {exc}")
                        continue
                cache.clear()
            return {"ok": not errors, "errors": errors, "removed": None}

        mc_dir = settings.get("GENERAL", "mc_dir")
        if not mc_dir:
            return {"ok": False, "errors": ["Выберите папку Minecraft!"], "removed": 0}

        values = set(scope.get("values") or [])
        lang_selector = scope.get("languages", "all")
        langs = list(LANGUAGES.values()) if lang_selector == "all" else [
            LANGUAGES[label] for label in lang_selector if label in LANGUAGES
        ]

        touched_caches: set[str] = set()
        removed = 0
        for lang in langs:
            usage_log = load_usage_log(mc_dir, lang["file"])
            if not usage_log:
                continue
            changed = False
            for source, entry in list(usage_log.items()):
                if mode == "mods":
                    match = bool(values.intersection(entry.get("mods", [])))
                else:  # mode == "kinds"
                    match = entry.get("kind") in values
                if not match:
                    continue
                engine = entry.get("engine", "google")
                cache = self.cache_ai if engine == "ai" else self.cache_std
                if cache.discard(lang["api"], source):
                    removed += 1
                    touched_caches.add(engine)
                del usage_log[source]
                changed = True
            if changed:
                save_usage_log(mc_dir, lang["file"], usage_log)

        for engine in touched_caches:
            cache = self.cache_ai if engine == "ai" else self.cache_std
            path = cache.filepath
            if os.path.exists(path):
                backup_path = f"{path}.bak-{timestamp}"
                try:
                    shutil.copy2(path, backup_path)
                except OSError as exc:
                    # Same rule as the mode=="all" branch above: never
                    # persist a destructive discard whose backup failed --
                    # the in-memory entries stay discarded for this session,
                    # but the on-disk cache (and every already-paid-for
                    # translation in it) is left untouched instead of being
                    # overwritten with no recovery copy.
                    errors.append(f"Не удалось создать резервную копию: {exc}")
                    continue
            cache.save()

        return {"ok": not errors, "errors": errors, "removed": removed}
