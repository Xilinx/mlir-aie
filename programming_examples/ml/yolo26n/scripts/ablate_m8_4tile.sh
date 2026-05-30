#!/bin/bash
# Per-tile ablation for M8_TILES=4 standalone — which worker is the gate?
# NOOP each kernel file in turn; m8 4-tile pipeline still runs locks/DMA so
# wall-time drop = that tile's contribution to the steady-state bottleneck.
set -uo pipefail

cd /scratch/ehunhoff/mlir-aie
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
source ironenv/bin/activate
source utils/env_setup.sh install >/dev/null 2>&1
cd programming_examples/ml/yolo26n

OUT=build/ablate_m8_4tile.csv
echo "config,median_ms,fps,std_ms" > "$OUT"

run_one() {
  local label="$1" files="$2"
  echo ""
  echo "=== $label (NOOP_KERNEL_FILES='$files') ==="
  rm -rf ~/.npu/cache 2>/dev/null
  make clean >/dev/null 2>&1
  local log
  log=$(M8_TILES=4 NOOP_KERNEL_FILES="$files" make BLOCK=m8 time 2>&1 | tail -3)
  echo "$log"
  local ms fps std
  ms=$(echo "$log" | grep -oE 'median=[0-9.]+ ms' | grep -oE '[0-9.]+' | head -1)
  fps=$(echo "$log" | grep -oE 'throughput @ median = [0-9.]+ fps' | grep -oE '[0-9.]+' | head -1)
  std=$(echo "$log" | grep -oE 'std=[0-9.]+ ms' | grep -oE '[0-9.]+' | head -1)
  echo "$label,$ms,$fps,$std" >> "$OUT"
}

# Baseline: m8 4-tile, no noop.
run_one "baseline_4t"               ""
# Tile A only noop'd — exposes B+C+D bottleneck.
run_one "noop_A_front"              "yolo_m8_front_cv1_split_fused_m8"
# Tile D only noop'd — exposes A+B+C bottleneck.
run_one "noop_D_back"               "yolo_m8_back_cv3_cv2_fused_m8"
# Tiles B+C noop'd (both inner-pair kernels) — exposes A+D bottleneck.
run_one "noop_BC_pairs"             "yolo_c3k2_heavy_inner_pair_cv1_streamed_m8 yolo_c3k2_heavy_inner_pair_cv2_skip_streamed_m8"
# A+D noop'd — exposes B+C.
run_one "noop_AD"                   "yolo_m8_front_cv1_split_fused_m8 yolo_m8_back_cv3_cv2_fused_m8"
# All 4 noop'd — pure pipeline/DMA floor.
run_one "noop_all"                  "yolo_m8_front_cv1_split_fused_m8 yolo_m8_back_cv3_cv2_fused_m8 yolo_c3k2_heavy_inner_pair_cv1_streamed_m8 yolo_c3k2_heavy_inner_pair_cv2_skip_streamed_m8"

echo ""
echo "=== RESULTS ($OUT) ==="
cat "$OUT"
