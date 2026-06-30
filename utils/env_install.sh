#!/usr/bin/env bash
##===- utils/env_install.sh - Setup MLIR-AIE Python env ------*- Script -*-===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Create a Python virtual environment and install MLIR-AIE's Python requirements
# plus the pinned llvm-aie (Peano). Source it so the environment stays active in
# the caller:
#
#   source utils/env_install.sh <venv-dir> [--dev]
#
# --dev also installs the hash-pinned development tooling and vendors eudsl.
# Override the interpreter with the PYTHON environment variable (default:
# python3). Works regardless of the caller's working directory.
#
##===----------------------------------------------------------------------===##

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${PYTHON:-python3}" -m venv "$1"
source "$1/bin/activate"
python3 -m pip install --upgrade pip
python3 -m pip install -r "$ROOT/python/requirements_ml.txt"
python3 -m pip install -r "$ROOT/python/requirements_notebook.txt"

# Development requirements
if [ "${2:-}" = "--dev" ]; then
  python3 -m pip install --require-hashes -r "$ROOT/python/requirements_dev.lock"
  # Vendored eudsl is built into from-source mlir_aie builds, so install the
  # rest of the core requirements without fetching eudsl from the index.
  python3 "$ROOT/utils/mlir_aie_wheels/vendor_eudsl.py" \
      --requirements "$ROOT/python/requirements.txt" \
      --install-non-eudsl
fi

# Peano (llvm-aie)
# Pinned via utils/peano-requirements.txt (bumped by the update-peano workflow).
python3 -m pip install -r "$ROOT/utils/peano-requirements.txt"
export PEANO_INSTALL_DIR="$(pip show llvm-aie | awk '/^Location:/ {print $2 "/llvm-aie"}')"
