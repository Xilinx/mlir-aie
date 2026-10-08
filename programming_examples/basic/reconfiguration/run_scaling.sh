#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail
cd "$(dirname "$0")"

output=${1:-benchmark.csv}
iters=${ITERS:-10}
modes=(separate-dispatch load-pdi expand-load-pdis)

echo 'case,mode,cols,rows,nops,switchboxes,reconfigs,iteration,scope,runtime_us' >"$output"

for mode in "${modes[@]}"; do
  for nops in 0 1000 2000 3000 4000; do
    python reconfiguration.py --csv --case progmem --mode "$mode" \
      --nops "$nops" --iters "$iters" >>"$output"
  done
  for switchboxes in 0 6 12 18 24; do
    python reconfiguration.py --csv --case switchboxes --mode "$mode" \
      --switchboxes "$switchboxes" --iters "$iters" >>"$output"
  done
done

echo "Wrote $output"
