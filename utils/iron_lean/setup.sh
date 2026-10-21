#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Install the pinned Lean toolchain and Mathlib for the IronVerify project, then
# build it. Safe to re-run: each step is skipped when already done.
#
# Usage: utils/iron_lean/setup.sh
#   ELAN_HOME  where elan and the toolchain live
#              (default: utils/iron_lean/IronVerify/.elan, gitignored)

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$here/IronVerify"
export ELAN_HOME="${ELAN_HOME:-$project/.elan}"
export PATH="$ELAN_HOME/bin:$PATH"

if [ ! -x "$ELAN_HOME/bin/elan" ]; then
  download="$(mktemp -d)"
  trap 'rm -rf "$download"' EXIT
  curl -sSfL -o "$download/elan-init.sh" \
    https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh
  sh "$download/elan-init.sh" -y --no-modify-path --default-toolchain none
fi

cd "$project"
# elan installs the toolchain named in lean-toolchain on first use.
lake exe cache get
lake build IronVerify ironeval

echo
echo "IronVerify is built. To use lake/ironeval in this shell:"
echo "  export ELAN_HOME=$ELAN_HOME PATH=$ELAN_HOME/bin:\$PATH"
