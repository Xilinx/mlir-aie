#!/usr/bin/env python3
"""
Sweep large volumes for Conv3D: test 3×32×32 up to 3×256×256
"""
import subprocess
import os
import sys

configs = [
    # (depth, height, width, cores, description)
    (3, 32, 32, 1, "Small video (3 frames, 32×32)"),
    (3, 32, 32, 2, "Small video (2-core)"),
    (3, 32, 32, 4, "Small video (4-core)"),
    
    (3, 64, 64, 2, "Medium video (3 frames, 64×64, 2-core)"),
    (3, 64, 64, 4, "Medium video (4-core)"),
    (3, 64, 64, 8, "Medium video (8-core)"),
    
    (3, 128, 128, 4, "Large video (3 frames, 128×128, 4-core)"),
    (3, 128, 128, 8, "Large video (8-core)"),
    
    (3, 256, 256, 8, "HD video (3 frames, 256×256, 8-core)"),
    (3, 256, 256, 16, "HD video (16-core)"),
]

# Device mapping
device_map = {
    1: "npu2",
    2: "npu2_2col",
    4: "npu2_4col",
    8: "npu2",  # Full NPU2 device (8 columns)
    16: "npu2",  # 8×2 grid
}

results = []

print("="*80)
print("Conv3D Large Volume Sweep")
print("="*80)
print(f"\nBuilding and testing {len(configs)} configurations...\n")

for depth, height, width, cores, desc in configs:
    ci, co = 8, 8
    name = f"d{depth}_h{height}_w{width}_c{cores}"
    device = device_map.get(cores, "npu2")
    
    print(f"\n[{configs.index((depth, height, width, cores, desc))+1}/{len(configs)}] {desc}")
    print(f"    Config: {depth}×{height}×{width}, {cores} cores")
    
    # Generate MLIR
    mlir_file = f"build/sweep_{name}.mlir"
    cmd = f"python3 conv3d_massively_parallel.py {device} {depth} {width} {height} {ci} {co} {cores} > {mlir_file}"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"    ❌ MLIR generation failed: {result.stderr[:100]}")
            results.append((desc, cores, None, "MLIR gen failed"))
            continue
        print(f"    ✓ MLIR generated")
    except Exception as e:
        print(f"    ❌ MLIR generation error: {e}")
        results.append((desc, cores, None, "Error"))
        continue
    
    # Build xclbin
    xclbin = f"build/sweep_{name}.xclbin"
    insts = f"build/sweep_{name}_insts.bin"
    cmd = f"cd build && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --no-xchesscc --no-xbridge --xclbin-name=sweep_{name}.xclbin --npu-insts-name=sweep_{name}_insts.bin sweep_{name}.mlir 2>&1"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"    ❌ Build failed: {result.stderr[:100]}")
            results.append((desc, cores, None, "Build failed"))
            continue
        print(f"    ✓ Built successfully")
    except Exception as e:
        print(f"    ❌ Build error: {e}")
        results.append((desc, cores, None, "Error"))
        continue
    
    # Test
    test_script = f"""
import numpy as np, aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime

depth, height, width, ci, co = {depth}, {height}, {width}, {ci}, {co}
k = NPUKernel("{xclbin}", "{insts}", kernel_name="MLIR_AIE")
h = DefaultNPURuntime.load(k)
np.random.seed(42)
ifm_r = np.random.randint(1, 20, (depth, ci//8, height, 8, width), dtype=np.uint8)
wts_r = np.random.randint(-50, 50, (co//8, ci//8, 3, 3, 3, 8, 8), dtype=np.int8)
buf = [iron.tensor(ifm_r.flatten(), dtype=np.uint8), iron.tensor(wts_r.flatten(), dtype=np.int8), iron.zeros(depth*height*width*co, dtype=np.uint8)]
for _ in range(3): DefaultNPURuntime.run(h, buf)
times = [DefaultNPURuntime.run(h, buf).npu_time/1000.0 for _ in range(10)]
print(f"{{np.mean(times):.1f}}")
"""
    
    try:
        result = subprocess.run(f"python3 -c '{test_script}'", shell=True, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            npu_time = float(result.stdout.strip())
            print(f"    ✓ NPU time: {npu_time:.1f}µs")
            results.append((desc, cores, npu_time, "PASS"))
        else:
            print(f"    ❌ Test failed: {result.stderr[:100]}")
            results.append((desc, cores, None, "Test failed"))
    except Exception as e:
        print(f"    ❌ Test error: {e}")
        results.append((desc, cores, None, "Error"))

# Summary
print(f"\n{'='*80}")
print(f"RESULTS SUMMARY")
print(f"{'='*80}\n")
print(f"{'Configuration':<40} {'Cores':>6} {'Time (µs)':>12} {'Status'}")
print(f"{'-'*80}")

for desc, cores, time, status in results:
    if time:
        print(f"{desc:<40} {cores:>6} {time:>12.1f} {status}")
    else:
        print(f"{desc:<40} {cores:>6} {'N/A':>12} {status}")

print(f"\n{'='*80}\n")

# Calculate speedups for same volume size
volume_groups = {}
for desc, cores, time, status in results:
    if time:
        # Extract volume from desc
        if "32×32" in desc:
            vol = "32×32"
        elif "64×64" in desc:
            vol = "64×64"
        elif "128×128" in desc:
            vol = "128×128"
        elif "256×256" in desc:
            vol = "256×256"
        else:
            continue
        
        if vol not in volume_groups:
            volume_groups[vol] = {}
        volume_groups[vol][cores] = time

print("MULTI-CORE SPEEDUPS")
print("="*80)
for vol, core_times in sorted(volume_groups.items()):
    if 1 in core_times:
        baseline = core_times[1]
        print(f"\n{vol} (baseline: {baseline:.1f}µs)")
        for cores in sorted(core_times.keys()):
            if cores == 1:
                continue
            speedup = baseline / core_times[cores]
            efficiency = (speedup / cores) * 100
            print(f"  {cores:2d}-core: {core_times[cores]:>7.1f}µs  {speedup:>5.2f}× speedup  {efficiency:>5.1f}% efficiency")

print(f"\n{'='*80}\n")
