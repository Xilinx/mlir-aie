#!/bin/bash
#
# Build + run a chain IRON design (pipeline=bn10..bn12 or cascade=bn13..bn14)
# against the per-chain brevitas fixtures in bottleneck_B|C/data/.
#
# Usage: run_chain_e2e.sh <mode> <mobilenet_srcdir>
#   mode: pipeline | cascade
#
# Output prefix: prints "PASS_CHAIN_E2E <mode>" on success, FAIL_* otherwise.

set -e
MODE="$1"
SRCDIR="$2"

case "$MODE" in
    pipeline) FIX="${SRCDIR}/bottleneck_B/data" ;;
    cascade)  FIX="${SRCDIR}/bottleneck_C/data" ;;
    *) echo "FAIL_CHAIN_E2E: unknown mode $MODE"; exit 1 ;;
esac

mkdir -p "build_${MODE}"

# 1. Generate chain MLIR with per-chain brevitas scales + weights.
python3 "${SRCDIR}/aie2_iron_chain.py" "${MODE}" \
    --data-dir "${FIX}" \
    --scales-json "${FIX}/scale_factors.json" \
    > "build_${MODE}/${MODE}.mlir"

# 2. Compile MLIR -> xclbin (link the .o files from main mobilenet build).
cd "build_${MODE}"
ln -sf "${SRCDIR}/build/"*.o . 2>/dev/null || true
aiecc.py --aie-generate-xclbin --no-compile-host \
    --xclbin-name="${MODE}.xclbin" \
    --no-xchesscc --no-xbridge \
    --dynamic-objFifos=false \
    --aie-generate-npu-insts --npu-insts-name="${MODE}_insts.bin" \
    "${MODE}.mlir"
cd ..

# 3. Run on NPU and bit-exact compare against per-chain brevitas golden.
python3 "${SRCDIR}/test_chain.py" \
    --mode "${MODE}" \
    --xclbin "build_${MODE}/${MODE}.xclbin" \
    --insts "build_${MODE}/${MODE}_insts.bin" \
    --fixture-dir "${FIX}" \
    --atol 0
