#!/bin/bash
source /scratch/jmelber/mlir-aie/ironenv/bin/activate
cd /scratch/jmelber/mlir-aie/programming_examples/ml/conv3d

echo "Quick Conv3D Sweep: Build and test key sizes"
echo "=============================================="

# Just build, don't test yet (builds are slow)
for size in 32 64 128; do
    for cores in 1 2 4 8; do
        # Skip invalid combos
        if [ $size -eq 32 ] && [ $cores -gt 4 ]; then continue; fi
        if [ $size -eq 64 ] && [ $cores -lt 4 ]; then continue; fi
        if [ $size -eq 128 ] && [ $cores -lt 4 ]; then continue; fi
        
        case $cores in
            1) dev="npu2" ;;
            2) dev="npu2_2col" ;;
            4) dev="npu2_4col" ;;
            8) dev="npu2" ;;
        esac
        
        name="q_d3_s${size}_c${cores}"
        echo "Building ${size}×${size} with $cores cores..."
        
        if python3 conv3d_spatial.py $dev 3 $size $size 8 8 > build/${name}.mlir 2>&1; then
            if (cd build && timeout 180 aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --no-xchesscc --no-xbridge --xclbin-name=${name}.xclbin --npu-insts-name=${name}_insts.bin ${name}.mlir > /dev/null 2>&1); then
                echo "  ✓ Built: build/${name}.xclbin"
            else
                echo "  ✗ Build failed or timeout"
            fi
        else
            echo "  ✗ MLIR failed"
        fi
    done
done

echo ""
echo "Built files:"
ls -lh build/q_*.xclbin 2>/dev/null | awk '{print $9, $5}' || echo "None"
