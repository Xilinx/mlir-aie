#!/usr/bin/env bash
# Regenerate every hash-pinned lockfile this repo maintains.
#
# Lockfiles are pip-style hashed requirements files compiled from a smaller
# "source" file via `uv pip compile`. Pairs:
#
#   python/requirements_dev.txt              -> python/requirements_dev.lock
#   utils/mlir_aie_wheels/requirements.txt   -> utils/mlir_aie_wheels/requirements.lock
#   utils/mlir_aie_wheels/ci-tools.in        -> utils/mlir_aie_wheels/ci-tools.lock
#   utils/mlir_wheels/requirements.in        -> utils/mlir_wheels/requirements.lock
#   utils/mlir_wheels/ci-tools.in            -> utils/mlir_wheels/ci-tools.lock
#
# uv version is pinned so the resolution (and the generated header comment)
# stays deterministic across machines and CI.
#
# Usage:  utils/regenerate_lockfiles.sh
#
# Exits non-zero if uv is missing or any compile fails.

set -euo pipefail

UV_VERSION="0.11.17"

# Locate uv: prefer `$HOME/.local/bin/uv` because `pip install --user uv==...`
# typically lands there, and falling back to whatever's first on PATH would
# silently use an unpinned version. Otherwise take whatever's on PATH and warn
# below if the version differs. Hard-fails if no uv at all.
if [ -x "$HOME/.local/bin/uv" ]; then
  UV="$HOME/.local/bin/uv"
elif command -v uv >/dev/null 2>&1; then
  UV=uv
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
    "$UV" pip compile --universal --generate-hashes --python-version=3.11 \
      "$src_name" -o "$lock_name" --quiet
  )
}

compile python/requirements_dev.txt              python/requirements_dev.lock
compile utils/mlir_aie_wheels/requirements.txt   utils/mlir_aie_wheels/requirements.lock
compile utils/mlir_aie_wheels/ci-tools.in        utils/mlir_aie_wheels/ci-tools.lock
compile utils/mlir_wheels/requirements.in        utils/mlir_wheels/requirements.lock
compile utils/mlir_wheels/ci-tools.in            utils/mlir_wheels/ci-tools.lock

echo "done"
