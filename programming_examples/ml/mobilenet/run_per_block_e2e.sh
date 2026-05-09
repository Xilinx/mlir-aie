#!/bin/bash
#
# Build + run a single per-block IRON design against bottleneck_A's per-bn
# brevitas fixtures.
#
# Usage: run_per_block_e2e.sh <bn_name> <mobilenet_srcdir>
# Example: run_per_block_e2e.sh bn1 /path/to/programming_examples/ml/mobilenet
#
# Output prefix: prints "PASS_BN_E2E <bn>" on success, "FAIL_BN_E2E <bn>" otherwise.
# Designed to be called from run_per_block_e2e.lit; FileCheck consumes the prefix.

set -e
BN="$1"
SRCDIR="$2"

# 1. Stage weights: copy bottleneck_A/bnN_single.txt -> stage/bnN_chain.txt so
#    the IRON loader (which reads bnN_chain.txt) picks up the per-bn weights.
STAGE="$(pwd)/stage_${BN}"
mkdir -p "$STAGE"
cp "${SRCDIR}/bottleneck_A/data/bn${BN#bn}_single.txt" "${STAGE}/bn${BN#bn}_chain.txt"

# 2. Generate per-block MLIR with per-bn scales + staged weights.
mkdir -p "build_${BN}"
python3 "${SRCDIR}/aie2_iron_per_block.py" "${BN}" \
    --data-dir "${STAGE}" \
    --scales-json "${SRCDIR}/bottleneck_A/data/scale_factors_per_bn.json" \
    > "build_${BN}/${BN}.mlir"

# 3. Compile MLIR -> xclbin (reuses .o files from main mobilenet build dir).
#    The .o files are referenced by relative path inside the MLIR; we run aiecc
#    from a directory where they are accessible.
cd "build_${BN}"
ln -sf "${SRCDIR}/build/"*.o . 2>/dev/null || true
aiecc.py --aie-generate-xclbin --no-compile-host \
    --xclbin-name="${BN}.xclbin" \
    --no-xchesscc --no-xbridge \
    --dynamic-objFifos=false \
    --aie-generate-npu-insts --npu-insts-name="${BN}_insts.bin" \
    "${BN}.mlir"
cd ..

# 4. Run on NPU and compare against per-bn brevitas golden.
python3 "${SRCDIR}/test_per_block.py" \
    --xclbin "build_${BN}/${BN}.xclbin" \
    --insts "build_${BN}/${BN}_insts.bin" \
    --bn "${BN}" \
    --fixture-dir "${SRCDIR}/bottleneck_A/data" \
    --atol 0
