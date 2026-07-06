#!/usr/bin/env python3

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Assert that ABI-coupled build dependencies resolve to the SAME version across
every lockfile that feeds a wheel build or its runtime/test environment.

The lockfiles guard (utils/regenerate_lockfiles.sh + the lintAndFormat
"Lockfiles match inputs" job) only proves each .lock is internally consistent
with its own source manifest. It does NOT catch the case where two manifests
pin the same package with different constraints so their locks resolve to
different versions -- e.g. requirements_dev.txt at nanobind==2.12.0 while the
wheel is built against nanobind>=2.13.0. That mismatch breaks nanobind's
Value-subclass recognition at import time and only surfaces on a hardware smoke
test. This checker makes that divergence a fast, obvious CI failure instead.

Packages listed in SHARED_PACKAGES must resolve identically in ALL locks in
which they appear. A package absent from a given lock is fine (not every env
needs every build tool); the constraint is only that where it IS present, the
version agrees.
"""

import re
import sys
from pathlib import Path

# Lockfiles that must agree on the shared build/ABI dependencies below. These
# are the hash-pinned outputs of utils/regenerate_lockfiles.sh that install
# into either a wheel build (mlir_wheels, mlir_aie_wheels) or the runtime/test
# environment that loads the resulting extension modules (requirements_dev).
LOCKFILES = [
    "python/requirements_dev.lock",
    "utils/mlir_wheels/requirements.lock",
    "utils/mlir_aie_wheels/requirements.lock",
]

# Packages whose version must match across every lock they appear in. nanobind
# and pybind11 compile C-extension ABI into the wheel and must match the
# environment that imports it; the rest are the shared build toolchain, kept in
# lockstep so a Dependabot bump can't move one lock and leave another behind.
SHARED_PACKAGES = [
    "nanobind",
    "pybind11",
    "cmake",
    "setuptools",
    "numpy",
]

# Matches a top-of-entry pin line in a uv/pip-compile lock, e.g.
#   nanobind==2.13.0 \
#   ninja==1.11.1.4 ; sys_platform == 'win32' \
# Captures the package name and the exact version. Continuation/hash lines are
# indented, so anchoring at column 0 avoids matching "# via nanobind" comments.
_PIN_RE = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)==(?P<version>[^\s;\\]+)")


def _normalize(name):
    # PEP 503 name normalization so "pybind11[global]" / underscores / case all
    # compare equal.
    name = name.split("[", 1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(path):
    """Return {normalized_name: version} for the pinned entries in a lockfile."""
    versions = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _PIN_RE.match(line)
        if m:
            versions[_normalize(m.group("name"))] = m.group("version")
    return versions


def main():
    repo_root = Path(__file__).resolve().parent.parent

    parsed = {}
    missing = []
    for rel in LOCKFILES:
        path = repo_root / rel
        if not path.is_file():
            missing.append(rel)
            continue
        parsed[rel] = parse_lock(path)

    if missing:
        print("error: lockfile(s) not found:", file=sys.stderr)
        for rel in missing:
            print(f"  {rel}", file=sys.stderr)
        return 1

    failures = []
    for pkg in SHARED_PACKAGES:
        key = _normalize(pkg)
        # {version: [locks that resolved to it]}
        seen = {}
        for rel, versions in parsed.items():
            if key in versions:
                seen.setdefault(versions[key], []).append(rel)
        if len(seen) > 1:
            failures.append((pkg, seen))

    if failures:
        print(
            "error: ABI-coupled dependencies diverge across lockfiles.\n"
            "These packages must resolve to the same version in every lock; a\n"
            "mismatch means a wheel is built against one version but imported\n"
            "under another. Align the constraints in the source manifests and\n"
            "rerun utils/regenerate_lockfiles.sh.\n",
            file=sys.stderr,
        )
        for pkg, seen in failures:
            print(f"  {pkg}:", file=sys.stderr)
            for version, locks in sorted(seen.items()):
                for rel in locks:
                    print(f"    {version:<16} {rel}", file=sys.stderr)
        return 1

    checked = ", ".join(SHARED_PACKAGES)
    print(f"shared dependency versions agree across all lockfiles ({checked})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
