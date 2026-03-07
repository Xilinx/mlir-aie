# Quick test for spatial multi-core
import sys
import numpy as np
from aie.utils.ml import DataShaper
import aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime

xclbin = sys.argv[1]
insts = sys.argv[2]
n_cores = int(sys.argv[3])

# Volume size can be passed as optional parameters
if len(sys.argv) > 4:
    width = int(sys.argv[4])
    height = int(sys.argv[5])
else:
    width = height = 8

depth = 8
ci, co = 8, 8

# Load kernel
npu_kernel = NPUKernel(xclbin, insts, kernel_name="MLIR_AIE")
kernel_handle = DefaultNPURuntime.load(npu_kernel)

# Create random data
np.random.seed(42)
ifm = np.random.randint(1, 20, (depth, ci, height, width), dtype=np.uint8)
wts = np.random.randint(-50, 50, (co, ci, 3, 3, 3), dtype=np.int8)

# Reorder to D{C/8}H{C8}W
ifm_reordered = np.zeros((depth, 1, height, 8, width), dtype=np.uint8)
for d in range(depth):
    for h in range(height):
        for c in range(8):
            for w in range(width):
                ifm_reordered[d, 0, h, c, w] = ifm[d, c, h, w]

# Reorder weights to {O/8}{I/8}KDHW{I8}{O8}
wts_reordered = np.zeros((1, 1, 3, 3, 3, 8, 8), dtype=np.int8)
for kd in range(3):
    for kh in range(3):
        for kw in range(3):
            for ic in range(8):
                for oc in range(8):
                    wts_reordered[0, 0, kd, kh, kw, ic, oc] = wts[oc, ic, kd, kh, kw]

# Create buffers
in1 = iron.tensor(ifm_reordered.flatten(), dtype=np.uint8)
in2 = iron.tensor(wts_reordered.flatten(), dtype=np.int8)
out = iron.zeros(depth * height * width * co, dtype=np.uint8)

buffers = [in1, in2, out]

print(f"Running {n_cores}-core spatial Conv3D...")
print(f"Buffers: {len(buffers)}")

# Run
ret = DefaultNPURuntime.run(kernel_handle, buffers)
print(f"NPU time: {ret.npu_time / 1000:.1f}µs")
print("PASS!")
