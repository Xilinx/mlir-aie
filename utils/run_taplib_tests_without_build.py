#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Run the pure-Python taplib test suite without a built ``aie`` package.

``aie.helpers.taplib`` is plain Python (NumPy arithmetic on sizes and
strides), so its tests under ``test/python/taplib`` do not need the MLIR
bindings. This script builds a throwaway shim package in a temporary
directory (an ``aie/`` package whose ``helpers`` is a symlink to
``python/helpers`` in this checkout, plus a minimal ``aie.utils``), runs
every test in that directory the way lit would (``python <test> | FileCheck
<test>``) and prints PASS/FAIL per test plus a summary. It exits non-zero
when any test fails.

Requirements (all pip-installable): ``numpy``, ``ml_dtypes`` and
``filecheck`` (the Python FileCheck port). If the ``filecheck`` command is
not on PATH, an LLVM ``FileCheck`` binary on PATH is used instead.

Usage, from anywhere::

    python3 utils/run_taplib_tests_without_build.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HELPERS_DIR = REPO / "python" / "helpers"
TEST_DIR = REPO / "test" / "python" / "taplib"


def find_filecheck() -> list[str]:
    """Return the FileCheck command to run, or exit with a clear message."""
    for name in ("filecheck", "FileCheck"):
        path = shutil.which(name)
        if path:
            return [path]
    sys.exit(
        "error: neither 'filecheck' (pip install filecheck) nor an LLVM "
        "'FileCheck' binary was found on PATH"
    )


def make_shim(root: Path) -> Path:
    """Create ``root/aie`` so that ``import aie.helpers.taplib`` works."""
    pkg = root / "aie"
    (pkg / "utils").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    os.symlink(HELPERS_DIR, pkg / "helpers", target_is_directory=True)
    (pkg / "utils" / "__init__.py").write_text(
        "from aie.helpers.npdtypes import ceildiv\n"
    )
    return root


def collect_tests() -> list[Path]:
    return sorted(
        p
        for p in TEST_DIR.glob("*.py")
        if p.name != "util.py" and not p.name.startswith("_")
    )


def run_test(test: Path, env: dict, filecheck: list[str]) -> tuple[bool, str]:
    """Run one test through FileCheck; return (passed, captured output)."""
    proc = subprocess.run(
        [sys.executable, str(test)],
        cwd=TEST_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False, proc.stdout + proc.stderr
    check = subprocess.run(
        filecheck + [str(test)],
        input=proc.stdout,
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        return False, check.stdout + check.stderr
    return True, ""


def main() -> int:
    if not HELPERS_DIR.is_dir() or not TEST_DIR.is_dir():
        sys.exit(f"error: {REPO} does not look like an mlir-aie checkout")
    filecheck = find_filecheck()
    tests = collect_tests()
    if not tests:
        sys.exit(f"error: no tests found in {TEST_DIR}")

    with tempfile.TemporaryDirectory(prefix="taplib-shim-") as tmp:
        shim = make_shim(Path(tmp))
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(shim), str(TEST_DIR)]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        failures = []
        for test in tests:
            passed, output = run_test(test, env, filecheck)
            print(f"{'PASS' if passed else 'FAIL'}: {test.name}", flush=True)
            if not passed:
                failures.append(test.name)
                print(output.rstrip(), file=sys.stderr)

    print(
        f"\n{len(tests) - len(failures)} passed, {len(failures)} failed, "
        f"{len(tests)} total"
    )
    if failures:
        print("failed: " + " ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
