"""Compute the mlir-aie wheel version from git history.

Standalone helper: stdlib-only imports so CI can invoke it directly
(``python utils/mlir_aie_wheels/_version_helper.py``) before any build
dependencies have been installed, while setup.py re-uses the same logic
to keep a single source of truth.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).parent


def _check_env(name: str, default: int = 0) -> bool:
    return os.getenv(name, str(default)) in {"1", "true", "True", "ON", "YES"}


def _git(*args: str) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", *args],
                cwd=str(_HERE),
                stderr=subprocess.PIPE,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        msg = getattr(exc, "stderr", b"") or b""
        if isinstance(msg, bytes):
            msg = msg.decode(errors="replace")
        print(
            f"warning: git {' '.join(args)} failed ({type(exc).__name__}): "
            f"{msg.strip() or exc}",
            file=sys.stderr,
        )
        return None


def get_version() -> str:
    override = os.environ.get("AIE_WHEEL_VERSION", "").lstrip("v")
    if override:
        return override

    described = _git(
        "describe",
        "--tags",
        "--long",
        "--abbrev=7",
        "--match",
        "v[0-9]*.[0-9]*.[0-9]*",
        "--exclude",
        "*-*",
    )
    if described:
        m = re.match(r"^v(\d+)\.(\d+)\.(\d+)-(\d+)-g([0-9a-f]+)$", described)
        if m:
            major, minor, patch, distance_s, sha = m.groups()
            distance = int(distance_s)
            if distance == 0:
                return f"{major}.{minor}.{patch}"
            base = f"{major}.{minor}.{int(patch) + 1}"
            version = f"{base}.dev{distance}"
            if _check_env("AIE_WHEEL_KEEP_LOCAL"):
                version += f"+g{sha}"
            return version
        print(
            f"warning: git describe returned {described!r} which does not match "
            "the expected vMAJOR.MINOR.PATCH-DISTANCE-gSHA shape",
            file=sys.stderr,
        )

    commit_count = _git("rev-list", "--count", "HEAD") or "0"
    version = f"0.0.0.dev{commit_count}"
    print(
        f"warning: get_version() falling back to {version!r}; "
        "no matching release tag was reachable from HEAD",
        file=sys.stderr,
    )
    if _check_env("AIE_WHEEL_KEEP_LOCAL"):
        sha = _git("rev-parse", "--short=7", "HEAD")
        if sha:
            version += f"+g{sha}"
    return version


if __name__ == "__main__":
    print(get_version())
