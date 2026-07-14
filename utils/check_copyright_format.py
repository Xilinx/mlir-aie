# Copyright (C) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Enforce a canonical format for copyright notices and license headers.

``reuse lint`` only checks that *some* notice is present; it never validates the
wording, holder name, year format, or the ordering of the header lines.

This script scans the **raw** text of every REUSE-tracked file and enforces:

1. Holder whitelist. Every copyright notice's holder must appear verbatim in
   ``utils/copyright_holders.txt``. This is what catches misspelled or otherwise
   unapproved company names; a holder that is not on the list fails the check.
2. First-party format. A notice naming a first-party holder (AMD or Xilinx) must
   additionally match one of the canonical patterns in ``APPROVED`` below.
3. Copyright before license. Within a file, every copyright notice must precede
   the ``SPDX-License-Identifier`` line.
4. No forbidden header strings. Any substring listed in
   ``utils/copyright_forbidden_strings.txt`` (e.g. the old LLVM boilerplate
   "This file is licensed under ...") fails the check.

Files whose licensing is declared via a ``precedence = "override"`` block in
``REUSE.toml`` (vendored / upstream text) are left untouched.

Usage:

    python utils/check_copyright_format.py            # check the whole repo
    python utils/check_copyright_format.py FILE...     # check only the given files
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

# Data files: the holder whitelist and the forbidden-string blacklist. Both are
# plain text, one entry per line, '#'-comments and blank lines ignored.
WHITELIST_PATH = REPO_ROOT / "utils" / "copyright_holders.txt"
FORBIDDEN_PATH = REPO_ROOT / "utils" / "copyright_forbidden_strings.txt"

# Assembled so the literal tokens never appear contiguously in this file.
_CR = "Copy" + "right"
_TAG = "SPDX-FileCopy" + "rightText:"
_LIC = "SPDX-License-" + "Identifier:"

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

# A line is treated as a license identifier if it starts with the SPDX tag.
_LICENSE_START = re.compile(rf"^{_LIC}", re.IGNORECASE)

# Parse the holder out of a notice: strip any leading prefix (SPDX tag, ©, (c)),
# the word "Copyright", an optional (C)/(c)/© symbol, and the year(s); whatever
# remains is the holder. Multiple comma-separated years/ranges are tolerated.
_HOLDER_RE = re.compile(
    rf"^(?:{_TAG}\s*)?"
    r"(?:©\s*|\(c\)\s*)?"
    rf"{_CR}\s*"
    r"(?:\(c\)|\(C\)|©)?\s*"
    rf"{_YEARS}(?:\s*,\s*{_YEARS})*"
    r"\s+(.+?)\s*$",
    re.IGNORECASE,
)

# Leading comment markers to strip before matching, e.g. "# ", "// ", " * ",
# "<!-- ", ";; ", "%". Trailing comment closers ("-->", "*/") are stripped too.
_LEAD = re.compile(r"^\s*(?:#+|//+|/\*+|\*+|<!--+|;+|%+|--+|!)\s?")
_TRAIL = re.compile(r"\s*(?:-->|\*/)\s*$")


def _read_lines(path: str) -> list[str]:
    """Return each non-empty entry from a one-per-line data file.

    Blank lines and lines starting with '#' are ignored.
    """
    entries: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            entries.append(s)
    return entries


def load_whitelist() -> set[str]:
    """Allowed copyright holders (exact match)."""
    return set(_read_lines(WHITELIST_PATH))


def load_forbidden() -> list[str]:
    """Substrings that must never appear in a license header."""
    return _read_lines(FORBIDDEN_PATH)


def extract_holder(notice: str) -> str | None:
    """Return the holder named in a notice, or None if it can't be parsed."""
    m = _HOLDER_RE.match(notice)
    return m.group(1).strip() if m else None


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


def _to_repo_relative(paths: list[str]) -> list[str]:
    """Return the in-repo paths as repo-relative POSIX strings.

    Paths outside the repository are dropped, so file-scoped scanning can never
    open arbitrary files off disk (e.g. when invoked with an absolute path);
    only files under REPO_ROOT are ever scanned. ``as_posix()`` keeps the
    separators consistent with ``tracked_files()`` on every platform.
    """
    rels = []
    for p in paths:
        try:
            rels.append(Path(p).resolve().relative_to(REPO_ROOT).as_posix())
        except ValueError:
            continue  # outside the repo: never scan
    return rels


def collect_notices(
    files: list[str] | None = None,
    forbidden: list[str] | None = None,
) -> tuple[dict[str, list[str]], list[str], list[tuple[str, str]]]:
    """Scan headers and return (notices, ordering_violations, forbidden_hits).

    - ``notices`` maps each raw notice line to the files it came from (inline
      headers + REUSE.toml declarations).
    - ``ordering_violations`` lists files where a copyright notice appears *after*
      the ``SPDX-License-Identifier`` line (every copyright notice must come
      before it).
    - ``forbidden_hits`` lists ``(file, matched_substring)`` for every forbidden
      string found in a header.

    When ``files`` is given, only those files are scanned (used by the file-scoped
    pre-push hook). When it is empty/None, the whole repo is scanned via
    ``tracked_files()`` (used by CI and ``--list``). REUSE.toml-declared notices
    are only folded in for the whole-repo scan, since they are not attributable to
    any single pushed file.
    """
    overrides, declared = load_reuse_toml()
    forbidden = forbidden if forbidden is not None else load_forbidden()
    forbidden_lower = [(s, s.lower()) for s in forbidden]
    notices: dict[str, list[str]] = defaultdict(list)
    ordering_violations: list[str] = []
    forbidden_hits: list[tuple[str, str]] = []

    scoped = bool(files)
    if not scoped:
        # Notices declared inside REUSE.toml (binary files, third-party, proprietary).
        for value in declared:
            notices[value].append("REUSE.toml")

    self_rel = str(Path(__file__).resolve().relative_to(REPO_ROOT))
    # Skip the data files too: they contain holder names / forbidden strings that
    # would otherwise be flagged as violations of themselves. mkdocs.yml carries a
    # real SPDX header at the top (validated by reuse lint), but its Material
    # `copyright:` footer setting is a config key the notice regex cannot tell
    # apart from a file license header, so the whole file is exempt here.
    skip_exact = {
        "REUSE.toml",
        self_rel,
        "mkdocs.yml",
        str(WHITELIST_PATH.relative_to(REPO_ROOT)),
        str(FORBIDDEN_PATH.relative_to(REPO_ROOT)),
    }
    candidates = _to_repo_relative(files) if scoped else tracked_files()
    for rel in candidates:
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
        copyright_idxs: list[int] = []
        license_idx: int | None = None
        seen_forbidden: set[str] = set()
        for i, line in enumerate(lines):
            low = line.lower()
            for original, needle in forbidden_lower:
                if needle in low and original not in seen_forbidden:
                    seen_forbidden.add(original)
                    forbidden_hits.append((rel, original))
            stripped = strip_comment(line)
            if _NOTICE_START.match(stripped):
                notices[stripped].append(rel)
                copyright_idxs.append(i)
            elif license_idx is None and _LICENSE_START.match(stripped):
                license_idx = i
        # Copyright must precede the license line: *every* copyright notice has to
        # come before the first license identifier. A file may carry several
        # notices (e.g. AMD and Xilinx) at the top, but none of them may appear
        # after the SPDX-License-Identifier line.
        if (
            license_idx is not None
            and copyright_idxs
            and max(copyright_idxs) > license_idx
        ):
            ordering_violations.append(rel)
    return notices, ordering_violations, forbidden_hits


def main(argv: list[str]) -> int:
    show_all = "--list" in argv
    files = [a for a in argv if not a.startswith("-")]
    whitelist = load_whitelist()
    notices, ordering_violations, forbidden_hits = collect_notices(files)

    # Classify every notice. A notice is a violation if its holder is not on the
    # whitelist, if it cannot be parsed, or if it is first-party but does not
    # match the canonical format. reason -> human-readable label.
    def classify(notice: str) -> str | None:
        holder = extract_holder(notice)
        if holder is None:
            return "unparseable notice"
        if holder not in whitelist:
            return f"holder not in whitelist: {holder!r}"
        if is_first_party(notice) and not is_approved(notice):
            return "non-canonical first-party format"
        return None

    violations = {n: classify(n) for n in notices}
    violations = {n: r for n, r in violations.items() if r is not None}

    if show_all:
        for notice in sorted(notices):
            reason = classify(notice)
            if reason is not None:
                mark = "BAD"
            elif is_first_party(notice):
                mark = "ok "
            else:
                mark = "3p "  # third-party, whitelisted holder
            print(f"[{mark}] ({len(notices[notice]):4d}) {notice}")
        print()

    if not violations and not ordering_violations and not forbidden_hits:
        n_first = sum(1 for n in notices if is_first_party(n))
        n_third = len(notices) - n_first
        print(
            f"OK: all {len(notices)} distinct copyright notice(s) use a "
            f"whitelisted holder ({n_first} first-party, {n_third} third-party); "
            "copyright precedes license and no forbidden header strings found."
        )
        return 0

    if violations:
        print("Non-conforming copyright notices found:\n")
        for notice in sorted(violations):
            srcs = notices[notice]
            print(f"  {notice!r}  [{violations[notice]}]  ({len(srcs)} source(s))")
            for f in sorted(srcs)[:3]:
                print(f"      e.g. {f}")
            if len(srcs) > 3:
                print(f"      ... and {len(srcs) - 3} more")
        print(
            f"\n{len(violations)} distinct non-conforming notice(s). Every holder "
            f"must be listed in {WHITELIST_PATH.relative_to(REPO_ROOT)}; first-party "
            f"notices must use the canonical '{_CR} (C) <year> Advanced Micro "
            "Devices, Inc.' / Xilinx form."
        )

    if ordering_violations:
        print(
            f"\nCopyright notice must precede the license identifier in "
            f"{len(ordering_violations)} file(s):"
        )
        for f in sorted(ordering_violations):
            print(f"      {f}")

    if forbidden_hits:
        print(
            f"\nForbidden header string(s) found (see "
            f"{FORBIDDEN_PATH.relative_to(REPO_ROOT)}):"
        )
        for f, needle in sorted(set(forbidden_hits)):
            print(f"      {f}: {needle!r}")

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
