#!/bin/bash
source /scratch/jmelber/mlir-aie/ironenv/bin/activate

echo "=== Testing Spatial Parallelism ==="
echo "8x8x8 volume:"
python3 test_spatial.py build/spatial_2core.xclbin build/spatial_2core_insts.bin 2 2>&1 | grep -E "µs|PASS"

echo ""
echo "Build status:"
ls -lh build/spatial*.xclbin | awk '{print $9, $5}'
