#!/bin/bash
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
# Per-tile ablation for M8_TILES=6 standalone.
set -uo pipefail

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
source ironenv/bin/activate
source utils/env_setup.sh install >/dev/null 2>&1
cd programming_examples/ml/yolo26n

OUT=build/ablate_m8_6tile.csv
echo "config,median_ms,fps,std_ms" > "$OUT"

run_one() {
  local label="$1" files="$2"
  echo ""
  echo "=== $label ==="
  rm -rf ~/.npu/cache 2>/dev/null
  make clean >/dev/null 2>&1
  local log
  log=$(M8_TILES=6 NOOP_KERNEL_FILES="$files" make BLOCK=m8 time 2>&1 | tail -3)
  echo "$log"
  local ms fps std
  ms=$(echo "$log" | grep -oE 'median=[0-9.]+ ms' | grep -oE '[0-9.]+' | head -1)
  fps=$(echo "$log" | grep -oE 'throughput @ median = [0-9.]+ fps' | grep -oE '[0-9.]+' | head -1)
  std=$(echo "$log" | grep -oE 'std=[0-9.]+ ms' | grep -oE '[0-9.]+' | head -1)
  echo "$label,$ms,$fps,$std" >> "$OUT"
}

# All tiles use these kernel files (same kernels as 4-tile; B1/B2/C1/C2
# each compile from the same _streamed_m8.o symbol).
FRONT=yolo_m8_front_cv1_split_fused_m8
BACK=yolo_m8_back_cv3_cv2_fused_m8
PCV1=yolo_c3k2_heavy_inner_pair_cv1_streamed_m8
PCV2=yolo_c3k2_heavy_inner_pair_cv2_skip_streamed_m8

run_one "baseline_6t"                 ""
run_one "noop_A_front"                "$FRONT"
run_one "noop_D_back"                 "$BACK"
run_one "noop_pair_cv1_only_B1+C1"    "$PCV1"
run_one "noop_pair_cv2_only_B2+C2"    "$PCV2"
run_one "noop_all_pairs_B1+B2+C1+C2"  "$PCV1 $PCV2"
run_one "noop_all"                    "$FRONT $BACK $PCV1 $PCV2"

echo ""
cat "$OUT"
