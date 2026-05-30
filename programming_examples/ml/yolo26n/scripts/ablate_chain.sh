#!/bin/bash
# Per-block chain ablation @ CHAIN_N_SAMPLES=15.
# For each block: rm its .o(s), clean_chain (wipes xclbin/mlir), rebuild with
# -DNOOP_KERNEL on that block's kernels, time the chain, log result.
# Δwall vs baseline = that block's steady-state contribution to chain throughput.

set -uo pipefail

cd /scratch/ehunhoff/mlir-aie
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
source ironenv/bin/activate
source utils/env_setup.sh install >/dev/null 2>&1
cd programming_examples/ml/yolo26n

OUT=build/ablate_chain_n15.csv
echo "block,median_ms,fps,std_ms" > "$OUT"

# block -> glob of .o files to remove before rebuild
declare -A OBJS=(
  [m0]="build/yolo_m0_*.o"
  [m1]="build/yolo_conv2dk3_stride2_silu_bias_oiyxi8o8.o"
  [m2]="build/yolo_c3k2_small_*_m2.o"
  [m3]="build/yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_m3.o"
  [m4]="build/yolo_c3k2_small_*_m4.o"
  [m5]="build/yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_m5.o"
  [m6]="build/yolo_c3k2_*_m6.o"
  [m7]="build/yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_m7.o"
  [m8]="build/yolo_m8_*.o build/yolo_c3k2_heavy_*_streamed_m8.o"
  [m9]="build/yolo_m9_*.o"
  [m10]="build/yolo_m10_*.o"
)

# Run baseline first (no NOOP) — re-measure within the same script run.
run_one() {
  local label="$1" noop="$2" rm_glob="$3"
  echo ""
  echo "=== $label (NOOP_BLOCK='$noop') ==="
  rm -rf ~/.npu/cache 2>/dev/null
  # Full clean every iter — Make tracks mtime, not flag changes, so leaving
  # any prior-iter .o around silently carries forward its NOOP setting.
  make clean >/dev/null 2>&1
  local log
  log=$(NOOP_BLOCK="$noop" CHAIN_N_SAMPLES=15 make time_chain 2>&1 | tail -3)
  echo "$log"
  # parse: "chain N=15: per-dispatch n=10 mean=... min=... median=X ms max=... std=Y ms"
  # plus  "chain N=15: per-sample median=Z ms -> W fps"
  local ms fps std
  ms=$(echo "$log" | grep -oE 'per-sample median=[0-9.]+ ms' | grep -oE '[0-9.]+' | head -1)
  fps=$(echo "$log" | grep -oE 'per-sample median=[0-9.]+ ms -> [0-9.]+ fps' | grep -oE '[0-9.]+ fps' | grep -oE '[0-9.]+')
  std=$(echo "$log" | grep -oE 'std=[0-9.]+ ms' | grep -oE '[0-9.]+' | head -1)
  echo "$label,$ms,$fps,$std" >> "$OUT"
}

run_one "baseline" "" ""
for b in m0 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10; do
  run_one "noop_$b" "$b" "${OBJS[$b]}"
done

echo ""
echo "=== RESULTS ($OUT) ==="
cat "$OUT"
