#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Run clang-tidy against a set of files, matching what CI enforces.
#
# clang-tidy needs a real compile database (compile_commands.json) and the
# tablegen'd headers a file references -- unlike clang-format, it can't run
# as a bare per-file text check. Any existing build works (clang or GCC --
# clang-tidy parses the compile flags with its own bundled frontend
# regardless of which compiler originally built it), as long as its
# tablegen'd headers have actually been generated. This script:
#   - locates a build directory's compile_commands.json (MLIR_AIE_BUILD_DIR
#     env var, defaulting to `build/` at the repo root)
#   - resolves the resource-dir clang-tidy needs to find builtin headers
#     (stddef.h, etc.) -- the pinned clang-tidy PyPI package ships none of
#     its own, unlike a system `apt install clang-tidy`
#   - resolves the real clang-tidy binary directly, bypassing the PyPI
#     package's console-script wrapper: that wrapper's __init__.py does
#     `import pkg_resources` purely to locate this same binary at a fixed
#     relative path, and pkg_resources is no longer bundled by setuptools
#     >=81 -- which requirements_dev.txt requires for unrelated reasons, so
#     downgrading setuptools repo-wide isn't an option. `clang-tidy --version`
#     fails outright in that combination; finding the binary by its known
#     path sidesteps the import entirely.
#   - runs clang-tidy across the given files in parallel (one process per
#     file; clang-tidy itself only parallelizes across files, not within one
#     file's analysis)
#
# Usage:       utils/run_clang_tidy.sh <file>...
#              MLIR_AIE_BUILD_DIR=/path/to/build utils/run_clang_tidy.sh <file>...
#
# Exits non-zero if the compile database is missing, clang++ isn't on PATH,
# or clang-tidy reports any finding on any file.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR="${MLIR_AIE_BUILD_DIR:-$REPO_ROOT/build}"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <file>..." >&2
  exit 1
fi

if [ ! -f "$BUILD_DIR/compile_commands.json" ]; then
  echo "error: no compile_commands.json found in $BUILD_DIR" >&2
  echo "       clang-tidy needs a compile database. If you already have a build" >&2
  echo "       there (from docs/Building.md), generate one in place with:" >&2
  echo "         ninja -C \"$BUILD_DIR\" -t compdb > \"$BUILD_DIR/compile_commands.json\"" >&2
  echo "       or point at a different build with MLIR_AIE_BUILD_DIR=/path/to/build." >&2
  echo "       (see docs/CONTRIBUTING.md's clang-tidy section for details)" >&2
  exit 1
fi

if ! command -v clang++ >/dev/null 2>&1; then
  echo "error: clang++ not found on PATH (needed to resolve clang-tidy's builtin header search path)." >&2
  exit 1
fi

RESOURCE_DIR=$(clang++ -print-resource-dir)

# Prefer the clang-tidy PyPI package's actual binary, found without
# importing its broken wrapper (see note above); fall back to plain
# `clang-tidy` on PATH for a system install that has no such wrapper.
CLANG_TIDY_BIN=$(python3 - <<'PYEOF'
import sysconfig, os
for p in set(sysconfig.get_paths().values()):
    # Match the binary itself only -- clang_tidy/data/bin/ also ships
    # clang-tidy-diff.py, run-clang-tidy.py, clang-apply-replacements, etc.,
    # and an unanchored glob can pick one of those instead (glob.glob order
    # isn't alphabetical, it's directory-iteration order).
    for name in ("clang-tidy", "clang-tidy.exe"):
        candidate = os.path.join(p, "clang_tidy", "data", "bin", name)
        if os.path.isfile(candidate):
            print(candidate)
            raise SystemExit
PYEOF
)
CLANG_TIDY_BIN="${CLANG_TIDY_BIN:-clang-tidy}"

printf '%s\n' "$@" | xargs -P "$(nproc)" -I{} \
  "$CLANG_TIDY_BIN" -p "$BUILD_DIR" --extra-arg="-resource-dir=$RESOURCE_DIR" \
  --warnings-as-errors='*' {}
