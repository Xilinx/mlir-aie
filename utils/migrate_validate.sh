#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Validate a single migrated .mlir test by executing its own `// RUN:` lines,
# substituting %s -> the file and resolving aie-opt/aie-translate/FileCheck from
# the build tree. Exits 0 if all RUN lines pass, non-zero otherwise.
#
# Usage: migrate_validate.sh <file.mlir>
set -uo pipefail
ROOT="/scratch/ehunhoff/mlir-aie"
export PATH="$ROOT/build/bin:$ROOT/my_install/mlir/bin:$PATH"
f="$1"

# Skip files whose RUN lines need hardware / peano / chess (npu-xrt run, etc.);
# report as SKIP so the caller can note them for a hardware pass.
if grep -qE "// RUN:.*(run_on_npu|%run_on_npu|xchesscc|aiecc\.py|%PYTHON|clang)" "$f"; then
  echo "SKIP(hw/py): $f"
  exit 0
fi

# Extract RUN lines, join continuations (trailing backslash), run each.
mapfile -t runs < <(python3 - "$f" <<'PY'
import sys, re
txt = open(sys.argv[1]).read()
# join backslash continuations
txt = re.sub(r"\\\n", " ", txt)
for line in txt.splitlines():
    m = re.search(r"//\s*RUN:\s*(.*)", line)
    if m:
        print(m.group(1).strip())
PY
)

if [ ${#runs[@]} -eq 0 ]; then echo "NORUN: $f"; exit 0; fi

fail=0
for cmd in "${runs[@]}"; do
  # substitute lit variables we support; skip commands using unsupported %vars
  c="${cmd//%s/$f}"
  if echo "$c" | grep -qE "%[a-zA-Z_]"; then
    echo "SKIP(var): $f :: $c"
    continue
  fi
  if ! bash -c "$c" >/dev/null 2>/tmp/mv_err.$$; then
    echo "FAIL: $f"
    echo "  cmd: $c"
    sed 's/^/  /' /tmp/mv_err.$$ | head -6
    fail=1
  fi
done
rm -f /tmp/mv_err.$$
[ $fail -eq 0 ] && echo "PASS: $f"
exit $fail
