#!/usr/bin/env python3

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run programming_examples/ Makefile-driven examples against a pip-installed
mlir_aie wheel + Peano, without a CMake/lit build tree.

test/ and programming_guide/ drive their lit suites through a CMake-templated
lit.site.cfg.py and are out of scope here; programming_examples/*/*.lit files
are simple enough (REQUIRES + a handful of RUN: lines) to replay directly.

Scope, matching what a `pip install mlir_aie` + Peano-only user gets:
  - only run_makefile*.lit / run_strix_makefile*.lit files tagged
    `makefile_examples` and `peano` (or with no compiler tag at all)
  - only the NPU family this runner targets (ryzen_ai_npu1 / ryzen_ai_npu2)
  - `chess`-only variants, and anything requiring `opencv` or `torch` extras,
    are reported as skipped rather than silently dropped
  - a test can also fail purely because it reaches for a tool/package the
    wheel-only environment doesn't ship (FileCheck, jupyter, an undeclared
    torch import, ...) without that being spelled out in REQUIRES; those are
    detected from the failure output and reported as skipped too
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

NPU_FEATURE = {"aie2-4col": "ryzen_ai_npu1", "aie2p-8col": "ryzen_ai_npu2"}
NPU_KIND = {"aie2-4col": "npu1", "aie2p-8col": "npu2"}
EXTRA_FEATURES_SKIPPED = {"opencv", "torch"}

REQUIRES_RE = re.compile(r"^//\s*REQUIRES:\s*(.*)$")
XFAIL_RE = re.compile(r"^//\s*XFAIL:")
RUN_RE = re.compile(r"^//\s*RUN:\s*(.*)$")

MISSING_MODULE_RE = re.compile(r"ModuleNotFoundError: No module named '([\w.]+)'")
COMMAND_NOT_FOUND_RE = re.compile(
    r"(?:/bin/sh: \d+: |bash: line \d+: )(\S+): (?:not found|command not found)"
)


def missing_optional_dependency(log: str) -> str | None:
    """Recognize a failure caused by a tool/package the wheel-only
    environment doesn't ship, even when the .lit file's REQUIRES didn't
    declare it (e.g. mobilenet imports torch without REQUIRES: torch, and
    notebook/FileCheck-based examples don't declare those tools at all)."""
    if m := MISSING_MODULE_RE.search(log):
        return f"missing Python package '{m.group(1)}'"
    if m := COMMAND_NOT_FOUND_RE.search(log):
        return f"missing tool '{m.group(1)}'"
    return None


def parse_lit(path: Path):
    requires: list[str] = []
    xfail = False
    run_lines: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if m := REQUIRES_RE.match(line):
            requires.extend(x.strip() for x in m.group(1).split(","))
        elif XFAIL_RE.match(line):
            xfail = True
        elif m := RUN_RE.match(line):
            run_lines.append(m.group(1))
    return requires, xfail, run_lines


def substitute(
    cmd: str, lit_dir: Path, repo_root: Path, npu_kind: str, tmp: Path, stem: str
) -> str:
    cmd = cmd.replace("%S", str(lit_dir))
    # Match lit's %t/%T: %T is the per-test scratch directory, %t is a unique
    # path stem within it (tests commonly append a suffix, e.g. "%t.work").
    cmd = cmd.replace("%T", str(tmp))
    cmd = cmd.replace("%t", str(tmp / stem))
    wrapper = (
        f'"{sys.executable}" "{repo_root / "utils" / "run_on_npu.py"}" "{npu_kind}"'
    )
    cmd = re.sub(r"%run_on_npu[12]%", wrapper, cmd)
    return cmd


def run_one(
    lit_path: Path,
    run_lines: list[str],
    lit_dir: Path,
    repo_root: Path,
    npu_kind: str,
    timeout: int,
):
    with tempfile.TemporaryDirectory(prefix=lit_path.stem + "_") as tmp_str:
        tmp = Path(tmp_str)
        script = "set -euo pipefail\n" + "\n".join(
            substitute(line, lit_dir, repo_root, npu_kind, tmp, lit_path.stem)
            for line in run_lines
        )
        try:
            proc = subprocess.run(
                ["bash", "-c", script],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            return (
                False,
                f"TIMEOUT after {timeout}s\n{e.stdout or ''}\n{e.stderr or ''}",
            )
        if proc.returncode != 0:
            return False, f"exit={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
        return True, proc.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runner-type", required=True, choices=sorted(NPU_FEATURE))
    ap.add_argument(
        "--repo-root", default=Path(__file__).resolve().parent.parent, type=Path
    )
    ap.add_argument("--root", default="programming_examples", type=Path)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    npu_feature = NPU_FEATURE[args.runner_type]
    npu_kind = NPU_KIND[args.runner_type]
    root = (args.repo_root / args.root).resolve()

    lit_files = sorted(root.rglob("run_makefile*.lit")) + sorted(
        root.rglob("run_strix_makefile*.lit")
    )

    passed, xfailed, xpassed, skipped, failed = [], [], [], [], []

    for lit_path in lit_files:
        requires, xfail, run_lines = parse_lit(lit_path)
        rel = lit_path.relative_to(args.repo_root)

        if "makefile_examples" not in requires:
            continue
        if npu_feature not in requires:
            continue
        extras = EXTRA_FEATURES_SKIPPED & set(requires)
        if extras:
            skipped.append((rel, f"requires {sorted(extras)} (out of scope)"))
            continue
        if "chess" in requires and "peano" not in requires:
            skipped.append((rel, "chess-only (out of scope)"))
            continue

        ok, log = run_one(
            lit_path, run_lines, lit_path.parent, args.repo_root, npu_kind, args.timeout
        )
        if not ok and (reason := missing_optional_dependency(log)):
            skipped.append((rel, f"{reason} (out of scope)"))
        elif xfail:
            if ok:
                # Unexpected pass under XFAIL: lit reports this as XPASS, a
                # regression signal (the XFAIL marker is now stale), not a
                # clean result to fold silently into "xfailed".
                xpassed.append(rel)
            else:
                # Expected failure under XFAIL: this is the normal lit outcome.
                xfailed.append(rel)
        elif ok:
            passed.append(rel)
        else:
            failed.append((rel, log))

    print("\n==== programming_examples (wheel-driven) summary ====")
    print(f"  passed:  {len(passed)}")
    print(f"  xfailed: {len(xfailed)} (expected, not counted as failures)")
    print(
        f"  xpassed: {len(xpassed)} (unexpected pass under XFAIL -- treated as a failure)"
    )
    print(f"  skipped: {len(skipped)}")
    print(f"  failed:  {len(failed)}")

    if skipped:
        print("\n-- skipped --")
        for rel, reason in skipped:
            print(f"  {rel}: {reason}")

    if xpassed:
        print("\n-- xpassed (remove the stale XFAIL marker) --")
        for rel in xpassed:
            print(f"  {rel}")

    if failed:
        print("\n-- failed --")
        for rel, log in failed:
            print(f"\n[FAILED] {rel}")
            print(log[-4000:])

    sys.exit(1 if failed or xpassed else 0)


if __name__ == "__main__":
    main()
