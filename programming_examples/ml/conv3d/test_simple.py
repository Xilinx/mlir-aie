#!/usr/bin/env python3
# Quick test for simple conv3d

import numpy as np
import aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime

depth, height, width = 8, 8, 8
ci, co = 8, 8

# Create test data
input_data = np.arange(depth * height * width * ci, dtype=np.uint8)
weights = np.ones(ci * co, dtype=np.int8)
output_data = np.zeros(depth * height * width * co, dtype=np.uint8)

# Load kernel
npu_kernel = NPUKernel("build/final_simple.xclbin", "build/insts_simple.bin", kernel_name="MLIR_AIE")
kernel_handle = DefaultNPURuntime.load(npu_kernel)

# Prepare buffers
in_buf = iron.tensor(input_data, dtype=np.uint8)
wt_buf = iron.tensor(weights, dtype=np.int8)
out_buf = iron.tensor(output_data, dtype=np.uint8)

buffers = [in_buf, wt_buf, out_buf]

# Run
print("Running simple conv3d...")
try:
    ret = DefaultNPURuntime.run(kernel_handle, buffers)
    print(f"✅ SUCCESS! NPU time: {ret.npu_time / 1000:.2f}μs")

    output = buffers[-1].numpy() if not isinstance(buffers[-1], np.ndarray) else buffers[-1]
    print(f"Output range: {output.min()} to {output.max()}")
    print(f"First 16 output values: {output[:16]}")

except Exception as e:
    print(f"❌ FAILED: {e}")
    exit(1)
