#!/bin/bash
# Comprehensive debug script for conv3d

set -e

CONV3D_DIR="/scratch/jmelber/mlir-aie/programming_examples/ml/conv3d"
PYTHON="/scratch/jmelber/mlir-aie/ironenv/bin/python3"
AIECC="/scratch/jmelber/mlir-aie/ironenv/lib/python3.12/site-packages/mlir_aie/bin/aiecc"

cd "$CONV3D_DIR"

echo "=== Conv3D Debug Suite ==="
echo ""

# Test 1: Rebuild conv3d with npu2, 4x4x4
echo "Test 1: Building conv3d 4x4x4 with npu2..."
$PYTHON conv3d.py npu2 4 4 4 8 8 0 2>/dev/null > build/aie2_4x4.mlir
cd build
$AIECC --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --no-xchesscc --no-xbridge \
    --xclbin-name=final_4x4.xclbin --npu-insts-name=insts_4x4.bin aie2_4x4.mlir > /dev/null 2>&1
cd ..

if [ -f build/final_4x4.xclbin ]; then
    echo "✅ Build successful"
    echo "Testing on NPU..."
    $PYTHON test.py -x build/final_4x4.xclbin -i build/insts_4x4.bin -k MLIR_AIE -d 4 -ht 4 -wd 4 -ic 8 -oc 8 2>&1 | grep -E "(PASS|FAIL|SUCCESS|NPU time|Timeout)"
else
    echo "❌ Build failed"
fi

echo ""
echo "=== End Debug Suite ==="
