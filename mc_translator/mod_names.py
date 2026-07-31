import json
import os
import re
import zipfile
from functools import cache


def get_mod_name(filepath: str) -> str:
    """Cached by (path, mtime): a jar's display name is looked up via
    analyzer/estimator/jar processor multiple times in one session, and this
    avoids reopening the same zip just to read its manifest each time. Keying
    on mtime (not just path) means a jar that changes on disk still gets a
    fresh read."""
    try:
        mtime = os.path.getmtime(filepath)
    except OSError:
        mtime = 0.0
    return _get_mod_name_cached(filepath, mtime)


@cache
def _get_mod_name_cached(filepath: str, mtime: float) -> str:
    # Try to find real name in metadata first
    try:
        with zipfile.ZipFile(filepath, "r") as zin:
            # Fabric
            if "fabric.mod.json" in zin.namelist():
                try:
                    data = json.loads(zin.read("fabric.mod.json").decode("utf-8-sig"))
                    if "name" in data:
                        return data["name"]
                except Exception:
                    pass
            # Forge
            if "META-INF/mods.toml" in zin.namelist():
                try:
                    content = zin.read("META-INF/mods.toml").decode("utf-8-sig")
                    match = re.search(r'displayName\s*=\s*"(.*?)"', content)
                    if match:
                        return match.group(1)
                except Exception:
                    pass
    except Exception:
        pass

    # Fallback to filename
    base = os.path.basename(filepath).replace(".jar", "")
    # Remove versioning hints
    base = base.split("-0")[0].split("-1")[0].split("-2")[0].split("-3")[0]
    return base.replace("_", " ").replace("-", " ").title()
