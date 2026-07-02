# Copyright (C) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Enforce a canonical format for first-party (AMD / Xilinx) copyright notices.

``reuse lint`` only checks that *some* notice is present; it never validates the
wording, holder name, or year format.

This script scans the **raw** text of every REUSE-tracked file. A notice that
names a first-party holder (AMD or Xilinx) must match one of the canonical
patterns in ``APPROVED`` below, otherwise the check fails. Third-party notices
(any other holder) are left untouched, so vendored upstream attribution is
preserved verbatim.

Usage:

    python utils/check_copyright_format.py            # check the whole repo
    python utils/check_copyright_format.py --list     # also print every notice

Note: the trigger tokens below are assembled from fragments on purpose, so that
this file's own pattern strings are not themselves picked up as notices.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parent.parent
# Only the first lines of each file are scanned; license headers live at the top.
HEADER_LINES = 80

# Assembled so the literal tokens never appear contiguously in this file.
_CR = "Copy" + "right"
_TAG = "SPDX-FileCopy" + "rightText:"

# A year is a single year or an inclusive range, e.g. 2024 or 2022-2026.
_YEARS = r"\d{4}(?:-\d{4})?"

# Canonical first-party formats. A first-party notice is valid only if it fully
# matches at least one of these (anchored) patterns. The SPDX-FileCopyrightText:
# prefix is intentionally NOT accepted; neither is a trailing comma after the
# year, "All rights reserved", or the "AMD Inc." shorthand.
APPROVED: list[tuple[str, str]] = [
    ("AMD", rf"{_CR} \(C\) {_YEARS} Advanced Micro Devices, Inc\."),
    ("Xilinx", rf"{_CR} \(C\) {_YEARS} Xilinx, Inc\."),
]

_COMPILED = [(label, re.compile(rf"^{pattern}$")) for label, pattern in APPROVED]

# A notice is "first-party" (and therefore format-enforced) if it names AMD or
# Xilinx. Every other holder is third-party: its notice is left as-is so vendored
# upstream attribution is preserved verbatim. The bare "AMD" shorthand is matched
# too, so non-canonical forms like "AMD Inc." are still caught.
_FIRST_PARTY = re.compile(r"Advanced Micro Devices|\bAMD\b|Xilinx", re.IGNORECASE)

# A line is treated as a copyright notice if (after stripping comment syntax) it
# starts with one of these tokens. Assembled from fragments, case-insensitive.
_NOTICE_START = re.compile(rf"^(?:{_TAG}|©|\(c\)\s*{_CR}|{_CR})", re.IGNORECASE)

# Leading comment markers to strip before matching, e.g. "# ", "// ", " * ",
# "<!-- ", ";; ", "%". Trailing comment closers ("-->", "*/") are stripped too.
_LEAD = re.compile(r"^\s*(?:#+|//+|/\*+|\*+|<!--+|;+|%+|--+|!)\s?")
_TRAIL = re.compile(r"\s*(?:-->|\*/)\s*$")


def is_first_party(notice: str) -> bool:
    return bool(_FIRST_PARTY.search(notice))


def is_approved(notice: str) -> bool:
    return any(rx.match(notice) for _, rx in _COMPILED)


def strip_comment(line: str) -> str:
    """Strip a single layer of comment syntax and surrounding whitespace."""
    s = line.rstrip("\n").lstrip("\ufeff").strip()
    s = _TRAIL.sub("", s)
    s = _LEAD.sub("", s)
    return s.strip()


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a REUSE path glob (supporting ``**``) to an anchored regex."""
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def load_reuse_toml() -> tuple[list[re.Pattern[str]], list[str]]:
    """Return (override path matchers, all declared notice strings)."""
    toml_path = REPO_ROOT / "REUSE.toml"
    overrides: list[re.Pattern[str]] = []
    declared: list[str] = []
    if not toml_path.exists():
        return overrides, declared
    data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    for ann in data.get("annotations", []):
        paths = ann.get("path", [])
        if isinstance(paths, str):
            paths = [paths]
        cps = ann.get("SPDX-FileCopyrightText", [])
        if isinstance(cps, str):
            cps = [cps]
        declared.extend(cps)
        if ann.get("precedence") == "override":
            overrides.extend(_glob_to_regex(p) for p in paths)
    return overrides, declared


def tracked_files() -> list[str]:
    """File paths REUSE tracks (respects .gitignore and skips submodules)."""
    reuse = shutil.which("reuse")
    if reuse is None:
        sys.exit(
            "error: 'reuse' is not on PATH "
            '(python -m pip install "reuse[charset-normalizer]==6.2.0")'
        )
    # `reuse lint --json` exits non-zero when the repo isn't fully REUSE
    # compliant, but still emits the full JSON report on stdout. We only need
    # the tracked-file list, so parse stdout regardless of the exit code and
    # fail gracefully if the output isn't valid JSON.
    proc = subprocess.run(
        [reuse, "lint", "--json"],
        text=True,
        capture_output=True,
    )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        msg = proc.stderr.strip() or proc.stdout.strip() or "no output"
        sys.exit(f"error: 'reuse lint --json' produced no parseable output:\n{msg}")
    return [rec.get("path", "").lstrip("./") for rec in data.get("files", [])]


def collect_notices() -> dict[str, list[str]]:
    """Return {raw notice line: [sources]} from inline headers + REUSE.toml."""
    overrides, declared = load_reuse_toml()
    notices: dict[str, list[str]] = defaultdict(list)

    # Notices declared inside REUSE.toml (binary files, third-party, proprietary).
    for value in declared:
        notices[value].append("REUSE.toml")

    self_rel = str(Path(__file__).resolve().relative_to(REPO_ROOT))
    skip_exact = {"REUSE.toml", self_rel}
    for rel in tracked_files():
        if rel in skip_exact or rel.startswith("LICENSES/"):
            continue
        if any(rx.match(rel) for rx in overrides):
            continue  # licensing comes from REUSE.toml; keep upstream text as-is
        path = REPO_ROOT / rel
        try:
            with path.open("r", encoding="utf-8", errors="strict") as fh:
                lines = []
                for _ in range(HEADER_LINES):
                    line = fh.readline()
                    if not line:
                        break
                    lines.append(line)
        except (OSError, UnicodeDecodeError):
            continue  # binary or unreadable: REUSE.toml handles these
        for line in lines:
            stripped = strip_comment(line)
            if _NOTICE_START.match(stripped):
                notices[stripped].append(rel)
    return notices


def main(argv: list[str]) -> int:
    show_all = "--list" in argv
    notices = collect_notices()

    first_party = {n: f for n, f in notices.items() if is_first_party(n)}
    violations = {n: f for n, f in first_party.items() if not is_approved(n)}

    if show_all:
        for notice in sorted(notices):
            if not is_first_party(notice):
                mark = "3p "  # third-party: allowed, not format-checked
            elif is_approved(notice):
                mark = "ok "
            else:
                mark = "BAD"
            print(f"[{mark}] ({len(notices[notice]):4d}) {notice}")
        print()

    if not violations:
        n_third = len(notices) - len(first_party)
        print(
            f"OK: all {len(first_party)} first-party (AMD/Xilinx) notices match "
            f"the canonical format ({n_third} third-party notice(s) allowed)."
        )
        return 0

    print("Non-conforming first-party copyright notices found:\n")
    for notice in sorted(violations):
        files = violations[notice]
        print(f"  {notice!r}  ({len(files)} source(s))")
        for f in sorted(files)[:3]:
            print(f"      e.g. {f}")
        if len(files) > 3:
            print(f"      ... and {len(files) - 3} more")
    print(
        f"\n{len(violations)} distinct non-conforming first-party notice(s). "
        f"Fix each to the canonical '{_CR} (C) <year> Advanced Micro Devices, "
        "Inc.' / Xilinx form."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
