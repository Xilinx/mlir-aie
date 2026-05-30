#!/bin/bash
# Combo NOOP ablation to disambiguate m8/m9/m10 back-end coupling.
set -uo pipefail

cd /scratch/ehunhoff/mlir-aie
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
source ironenv/bin/activate
source utils/env_setup.sh install >/dev/null 2>&1
cd programming_examples/ml/yolo26n

OUT=build/ablate_chain_n15_combos.csv
echo "config,median_ms,fps,std_ms" > "$OUT"

# Every config needs ALL of m8/m9/m10 rebuilt regardless of which is noop'd,
# since flags differ per run.
declare -A OBJS=(
  [m8]="build/yolo_m8_*.o build/yolo_c3k2_heavy_*_streamed_m8.o"
  [m9]="build/yolo_m9_*.o"
  [m10]="build/yolo_m10_*.o"
)

run_combo() {
  local label="$1" noop="$2"
  echo ""
  echo "=== $label (NOOP_BLOCK='$noop') ==="
  rm -rf ~/.npu/cache 2>/dev/null
  for b in m8 m9 m10; do
    # shellcheck disable=SC2086
    rm -f ${OBJS[$b]}
  done
  make clean_chain >/dev/null 2>&1
  local log
  log=$(NOOP_BLOCK="$noop" CHAIN_N_SAMPLES=15 make time_chain 2>&1 | tail -3)
  echo "$log"
  local ms fps std
  ms=$(echo "$log" | grep -oE 'per-sample median=[0-9.]+ ms' | grep -oE '[0-9.]+' | head -1)
  fps=$(echo "$log" | grep -oE 'per-sample median=[0-9.]+ ms -> [0-9.]+ fps' | grep -oE '[0-9.]+ fps' | grep -oE '[0-9.]+')
  std=$(echo "$log" | grep -oE 'std=[0-9.]+ ms' | grep -oE '[0-9.]+' | head -1)
  echo "$label,$ms,$fps,$std" >> "$OUT"
}

# Isolate each of m8/m9/m10 by noop'ing the other two.
run_combo "noop_m9_m10__isolate_m8"  "m9 m10"
run_combo "noop_m8_m10__isolate_m9"  "m8 m10"
run_combo "noop_m8_m9__isolate_m10"  "m8 m9"
# And all three together — pure DMA/glue floor.
run_combo "noop_m8_m9_m10__pure_dma" "m8 m9 m10"

echo ""
echo "=== RESULTS ($OUT) ==="
cat "$OUT"
