#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Run clang-tidy against a set of files, matching what CI enforces.
#
# clang-tidy needs a real compile database (compile_commands.json) and the
# tablegen'd headers a file references -- unlike clang-format, it can't run
# as a bare per-file text check. The build must be configured with clang, as
# CI's is: LLVM's CMake gives a GCC build other warning flags
# (-Wno-missing-field-initializers among them), and clang-tidy reports
# clang-diagnostic-* findings from those flags. Only its tablegen'd headers
# need to have been generated. This script:
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

COMPILER_ID=$(sed -n 's/^set(CMAKE_CXX_COMPILER_ID "\(.*\)")$/\1/p' \
  "$BUILD_DIR"/CMakeFiles/*/CMakeCXXCompiler.cmake 2>/dev/null | head -n 1)
if [ "$COMPILER_ID" != "Clang" ]; then
  echo "error: $BUILD_DIR was configured with ${COMPILER_ID:-an unknown compiler}, but CI's clang-tidy" >&2
  echo "       reads a clang-configured build, whose warning flags differ. Configure one" >&2
  echo "       with -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++" >&2
  echo "       -DCMAKE_EXPORT_COMPILE_COMMANDS=ON, build the header targets CI's" >&2
  echo "       'Prepare compile_commands.json' step lists (lintAndFormat.yml), and point" >&2
  echo "       MLIR_AIE_BUILD_DIR at it." >&2
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
paths = sysconfig.get_paths()
# platlib/purelib only, in that order -- clang-tidy's wheel is a
# platform-specific binary distribution, so platlib is where it normally
# lands; purelib is a fallback for oddly-configured installs. Iterating a
# fixed, ordered key list (rather than set(paths.values())) keeps this
# deterministic: a set's iteration order isn't guaranteed, so if the binary
# somehow existed under more than one path, the pick could vary run to run.
for key in ("platlib", "purelib"):
    p = paths[key]
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

PINNED=$(sed -n 's/^clang-tidy==\([^ ]*\).*/\1/p' "$REPO_ROOT/python/requirements_dev.lock")
FOUND=$("$CLANG_TIDY_BIN" --version 2>/dev/null | sed -n 's/.*LLVM version \([0-9.]*\).*/\1/p' || true)
if [ "$FOUND" != "$PINNED" ]; then
  echo "error: $CLANG_TIDY_BIN is clang-tidy ${FOUND:-of unknown version}, but CI runs $PINNED" >&2
  echo "       (python/requirements_dev.lock), whose checks differ. Install the pinned one" >&2
  echo "       into the python3 on PATH:" >&2
  echo "         python3 -m pip install --require-hashes -r python/requirements_dev.lock" >&2
  exit 1
fi

# clang-tidy 22 flipped --header-filter's default from '' (main file only) to
# '.*' (every non-system header), so without this it now also lints whatever
# MLIR/LLVM headers each file transitively includes. Pin it back to the old
# default: this project's incremental clang-tidy rollout only wants findings
# in the file list below, not in vendored headers.
printf '%s\n' "$@" | xargs -P "$(nproc)" -I{} \
  "$CLANG_TIDY_BIN" -p "$BUILD_DIR" --extra-arg="-resource-dir=$RESOURCE_DIR" \
  --header-filter='' --warnings-as-errors='*' {}
