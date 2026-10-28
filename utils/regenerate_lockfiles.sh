#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Regenerate every hash-pinned lockfile this repo maintains.
#
# Lockfiles are pip-style hashed requirements files compiled from a smaller
# "source" file via `uv pip compile`. Pairs:
#
#   python/requirements_dev.txt              -> python/requirements_dev.lock
#   python/requirements_notebook.txt         -> python/requirements_notebook.lock
#   utils/mlir_aie_wheels/requirements.txt   -> utils/mlir_aie_wheels/requirements.lock
#   utils/mlir_aie_wheels/ci-tools.in        -> utils/mlir_aie_wheels/ci-tools.lock
#   utils/mlir_wheels/requirements.in        -> utils/mlir_wheels/requirements.lock
#   utils/mlir_wheels/ci-tools.in            -> utils/mlir_wheels/ci-tools.lock
#   utils/update_peano_tools.txt             -> utils/update_peano_tools.lock
#   docs/requirements-mkdocs.txt             -> docs/requirements-mkdocs.lock
#   programming_examples/ml/yolo26n/requirements.txt -> .../requirements.lock
#
# python/requirements_ml.txt and utils/peano-requirements.txt are NOT in this
# list: both resolve through a second index (--extra-index-url / -f), and uv
# 0.11.17 generates an incomplete or wrong-platform hash for at least one
# transitive package when it does (reproduced against markupsafe and against
# llvm-aie's own multi-platform release assets -- repros kept out of tree).
# Re-add them once that's fixed upstream; until then those two installs stay
# version-pinned but not hash-pinned.
#
# uv version is pinned so the resolution (and the generated header comment)
# stays deterministic across machines and CI.
#
# This script also generates utils/mlir_wheels/requirements.in from
# utils/mlir_wheels/pyproject.toml's [build-system].requires. Edit
# pyproject.toml to change that list.
#
# Usage:  utils/regenerate_lockfiles.sh
#
# Exits non-zero if uv is missing or any compile fails.

set -euo pipefail

UV_VERSION="0.11.17"

# Locate uv: take whatever's first on PATH so callers control which uv runs
# (this is what `pip install --user uv` expects via ~/.local/bin being on
# PATH, and what CI runners expect via their own toolcache layout). The
# version check below warns if the resolved uv doesn't match UV_VERSION.
# As a convenience fallback, check ~/.local/bin directly so a user who
# installed with `pip install --user` but hasn't yet rehashed their PATH
# still gets a working invocation. Hard-fails if no uv at all.
if command -v uv >/dev/null 2>&1; then
  UV=uv
elif [ -x "$HOME/.local/bin/uv" ]; then
  UV="$HOME/.local/bin/uv"
else
  echo "error: uv not found." >&2
  echo "       install with:  pip install 'uv==${UV_VERSION}'" >&2
  echo "                 or:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

actual_version=$("$UV" --version 2>/dev/null | awk '{print $2}' || true)
if [ "$actual_version" != "$UV_VERSION" ]; then
  echo "warning: uv ${actual_version} is on PATH but this script is pinned to ${UV_VERSION}." >&2
  echo "         resolution may differ; install the pinned version with:" >&2
  echo "           pip install 'uv==${UV_VERSION}'" >&2
fi

# Locate a python3 with tomllib (stdlib since 3.11): plain `python3` on PATH
# is not a reliable signal of this -- it resolves to whatever the caller's
# shell defaults to, which is often an older system interpreter even on a box
# that also has 3.11+ installed under a versioned name.
PY=""
for cand in python3 python3.12 python3.11; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import tomllib' >/dev/null 2>&1; then
    PY="$cand"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "error: no python3.11+ (tomllib) found on PATH." >&2
  echo "       install one or put it first on PATH, e.g. python3.12." >&2
  exit 1
fi

# Resolve repo root regardless of caller's cwd.
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# Compile a single (source -> lock) pair. Runs from the source's parent dir
# so the path comments in the generated lock are short (e.g. "requirements.txt
# -o requirements.lock") and identical regardless of where this script is run.
compile() {
  local src_rel="$1" lock_rel="$2"
  local src_dir src_name lock_name
  src_dir=$(dirname "$src_rel")
  src_name=$(basename "$src_rel")
  lock_name=$(basename "$lock_rel")

  echo "regenerating $lock_rel"
  (
    cd "$REPO_ROOT/$src_dir"
    # unsafe-best-match: the default first-index strategy can stop at the first
    # index that lists a transitive package and silently emit a hash from the
    # wrong platform/version for it when a source carries --extra-index-url or
    # -f (confirmed against python/requirements_ml.txt: default strategy wrote
    # a single macOS-arm64 hash for a Linux-resolved markupsafe). --emit-*
    # keeps the source's index/find-links directives in the lock itself, so
    # installing the lock later needs no extra flags.
    "$UV" pip compile --universal --generate-hashes --python-version=3.11 \
      --index-strategy unsafe-best-match --emit-index-url --emit-find-links \
      "$src_name" -o "$lock_name" --quiet
  )
}

generate_requirements_in() {
  local pyproject="$REPO_ROOT/utils/mlir_wheels/pyproject.toml"
  local out="$REPO_ROOT/utils/mlir_wheels/requirements.in"

  echo "regenerating utils/mlir_wheels/requirements.in from pyproject.toml"

  {
    cat <<'EOF'
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# utils/regenerate_lockfiles.sh generates this file from pyproject.toml's
# [build-system].requires. Edit pyproject.toml, then rerun that script.
EOF
    "$PY" -c '
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    print(*tomllib.load(f)["build-system"]["requires"], sep="\n")
' "$pyproject"
  } > "$out"
}

generate_requirements_in

compile python/requirements_dev.txt              python/requirements_dev.lock
compile python/requirements_notebook.txt         python/requirements_notebook.lock
compile utils/mlir_aie_wheels/requirements.txt   utils/mlir_aie_wheels/requirements.lock
compile utils/mlir_aie_wheels/ci-tools.in        utils/mlir_aie_wheels/ci-tools.lock
compile utils/mlir_wheels/requirements.in        utils/mlir_wheels/requirements.lock
compile utils/mlir_wheels/ci-tools.in            utils/mlir_wheels/ci-tools.lock
compile utils/update_peano_tools.txt             utils/update_peano_tools.lock
compile docs/requirements-mkdocs.txt             docs/requirements-mkdocs.lock
compile programming_examples/ml/yolo26n/requirements.txt programming_examples/ml/yolo26n/requirements.lock

echo "done"
