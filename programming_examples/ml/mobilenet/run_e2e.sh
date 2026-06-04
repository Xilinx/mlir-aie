#!/bin/bash
#
# Build + run a per-block or per-chain IRON design on the NPU2 and bit-exact
# compare against brevitas fixtures.
#
# Usage:
#   run_e2e.sh block <bn>         <mobilenet_srcdir>
#   run_e2e.sh chain <pipeline|cascade> <mobilenet_srcdir>
#
# Outputs: PASS_E2E <mode>:<target> on success, FAIL_E2E ... otherwise.

set -e
MODE="$1"
TARGET="$2"
SRCDIR="$3"
TAG="${MODE}_${TARGET}"
ATOL=0   # default — overridden per target below if a known acceptable drift exists.

case "$MODE:$TARGET" in
    block:bn1|block:bn2|block:bn3|block:bn6|block:bn7|block:bn8)
        FIX="${SRCDIR}/bottleneck_A/data"
        # bottleneck_A's per-bn weights live under bnN_single.txt; our IRON loader
        # reads bnN_chain.txt — stage a copy under that name in a temp dir.
        STAGE="$(pwd)/stage_${TAG}"
        mkdir -p "$STAGE"
        cp "${FIX}/bn${TARGET#bn}_single.txt" "${STAGE}/bn${TARGET#bn}_chain.txt"
        DATA_DIR="$STAGE"
        SCALES="${FIX}/scale_factors_per_bn.json"
        BUILDER="aie2_iron_per_block.py ${TARGET}"
        ;;
    chain:regular)
        FIX="${SRCDIR}/bottleneck_A/data"
        DATA_DIR="$FIX"
        SCALES="${FIX}/scale_factors_fused.json"
        BUILDER="aie2_iron_chain.py regular"
        # Original placed-API chain test (test_bottleneckA.py) accepted atol=14
        # — same known-acceptable drift tracked in mlir-aie issue #3009.
        ATOL=14
        ;;
    chain:pipeline)
        FIX="${SRCDIR}/bottleneck_B/data"
        DATA_DIR="$FIX"
        SCALES="${FIX}/scale_factors.json"
        BUILDER="aie2_iron_chain.py pipeline"
        ;;
    chain:cascade)
        FIX="${SRCDIR}/bottleneck_C/data"
        DATA_DIR="$FIX"
        SCALES="${FIX}/scale_factors.json"
        BUILDER="aie2_iron_chain.py cascade"
        ;;
    *)
        echo "FAIL_E2E ${MODE}:${TARGET}: unknown target"
        exit 1
        ;;
esac

# 1. Generate IRON MLIR for this design.
mkdir -p "build_${TAG}"
python3 ${SRCDIR}/${BUILDER} \
    --data-dir "${DATA_DIR}" \
    --scales-json "${SCALES}" \
    > "build_${TAG}/${TAG}.mlir"

# 2. Compile MLIR -> xclbin (link the .o files from main mobilenet build).
cd "build_${TAG}"
# .o files are built into ../build/final_mobilenet.prj/ by
# `make -f %S/Makefile objs` in run_e2e.lit (the @iron.jit pipeline
# writes per-kernel .o files into the xclbin's .prj scratch dir).
# Fall back to ${SRCDIR}/build/final_mobilenet.prj/ for local runs.
ln -sf ../build/final_mobilenet.prj/*.o . 2>/dev/null || true
ln -sf "${SRCDIR}/build/final_mobilenet.prj/"*.o . 2>/dev/null || true
aiecc.py --aie-generate-xclbin --no-compile-host \
    --xclbin-name="${TAG}.xclbin" \
    --no-xchesscc --no-xbridge \
    --dynamic-objFifos=false \
    --aie-generate-npu-insts --npu-insts-name="${TAG}_insts.bin" \
    "${TAG}.mlir"
cd ..

# 3. Run on NPU and bit-exact compare against brevitas golden.
python3 "${SRCDIR}/test_e2e.py" "${MODE}" "${TARGET}" \
    --xclbin "build_${TAG}/${TAG}.xclbin" \
    --insts "build_${TAG}/${TAG}_insts.bin" \
    --fixture-dir "${FIX}" \
    --atol "${ATOL}"
