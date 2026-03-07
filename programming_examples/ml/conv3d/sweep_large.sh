#!/bin/bash
# Sweep large video volumes
set -e
source /scratch/jmelber/mlir-aie/ironenv/bin/activate

mkdir -p build

echo "================================================================================"
echo "Conv3D Large Volume Sweep: 3 frames × 32-256 resolution"
echo "================================================================================"
echo ""

# Test configurations
declare -a configs=(
    "3 32 32 1:Small (32×32, 1-core)"
    "3 32 32 2:Small (32×32, 2-core)"
    "3 32 32 4:Small (32×32, 4-core)"
    "3 64 64 4:Medium (64×64, 4-core)"
    "3 64 64 8:Medium (64×64, 8-core)"
    "3 128 128 8:Large (128×128, 8-core)"
    "3 256 256 8:HD (256×256, 8-core)"
)

results=()

for config in "${configs[@]}"; do
    IFS=':' read -r params desc <<< "$config"
    IFS=' ' read -r d h w c <<< "$params"

    echo "[$((${#results[@]}+1))/${#configs[@]}] $desc"
    echo "    Volume: ${d}×${h}×${w}, Cores: $c"

    # Determine device
    case $c in
        1) device="npu2" ;;
        2) device="npu2_2col" ;;
        4) device="npu2_4col" ;;
        8) device="npu2" ;;
        16) device="npu2" ;;
        *) device="npu2" ;;
    esac

    name="d${d}_h${h}_w${w}_c${c}"

    # Generate MLIR
    if python3 conv3d_spatial.py $device $d $w $h 8 8 > build/${name}.mlir 2>&1; then
        echo "    ✓ MLIR generated"
    else
        echo "    ❌ MLIR failed"
        results+=("$desc|$c|FAIL|MLIR")
        continue
    fi

    # Build
    if (cd build && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts \
        --no-compile-host --no-xchesscc --no-xbridge \
        --xclbin-name=${name}.xclbin --npu-insts-name=${name}_insts.bin ${name}.mlir > /dev/null 2>&1); then
        echo "    ✓ Built"
    else
        echo "    ❌ Build failed"
        results+=("$desc|$c|FAIL|Build")
        continue
    fi

    # Test
    npu_time=$(python3 -c "
import numpy as np, aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime
d,h,w,ci,co=$d,$h,$w,8,8
k=NPUKernel('build/${name}.xclbin','build/${name}_insts.bin',kernel_name='MLIR_AIE')
hand=DefaultNPURuntime.load(k)
np.random.seed(42)
ifm_r=np.random.randint(1,20,(d,1,h,8,w),dtype=np.uint8)
wts_r=np.random.randint(-50,50,(1,1,3,3,3,8,8),dtype=np.int8)
buf=[iron.tensor(ifm_r.flatten(),dtype=np.uint8),iron.tensor(wts_r.flatten(),dtype=np.int8),iron.zeros(d*h*w*co,dtype=np.uint8)]
[DefaultNPURuntime.run(hand,buf) for _ in range(3)]
times=[DefaultNPURuntime.run(hand,buf).npu_time/1000.0 for _ in range(10)]
print(f'{np.mean(times):.1f}')
" 2>&1)

    if [[ $npu_time =~ ^[0-9]+\.?[0-9]*$ ]]; then
        echo "    ✓ NPU time: ${npu_time}µs"
        results+=("$desc|$c|${npu_time}|PASS")
    else
        echo "    ❌ Test failed"
        results+=("$desc|$c|FAIL|Test")
    fi
    echo ""
done

# Print results
echo "================================================================================"
echo "RESULTS"
echo "================================================================================"
printf "%-45s %6s %12s %s\n" "Configuration" "Cores" "Time (µs)" "Status"
echo "--------------------------------------------------------------------------------"
for result in "${results[@]}"; do
    IFS='|' read -r desc cores time status <<< "$result"
    printf "%-45s %6s %12s %s\n" "$desc" "$cores" "$time" "$status"
done
echo "================================================================================"
