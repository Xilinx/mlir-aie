#!/usr/bin/env python3
import numpy as np
import time
import torch
import torch.nn as nn
import cv2
import aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime

depth, height, width, ci, co = 16, 32, 32, 8, 8

print(f"\n{'='*80}")
print(f"Video Conv3D Benchmark: {depth}×{height}×{width}, {ci}→{co} channels")
print(f"{'='*80}\n")

# 1. PyTorch CPU
model = nn.Conv3d(ci, co, kernel_size=(1,3,3), padding=0, bias=False)
model.eval()
inp = torch.randint(1, 20, (1, ci, depth, height, width)).type(torch.FloatTensor)
wt = torch.randint(-50, 50, (co, ci, 1, 3, 3)).type(torch.FloatTensor)
model.weight.data.copy_(wt)
inp_pad = torch.nn.functional.pad(inp, (1,1,1,1,0,0), mode='replicate')
for _ in range(3):
    _ = model(inp_pad)
t = []
for _ in range(10):
    s = time.perf_counter()
    _ = model(inp_pad)
    t.append((time.perf_counter()-s)*1e6)
pt_time = np.mean(t)
print(f"PyTorch CPU:      {pt_time:>8.0f}µs ± {np.std(t):.0f}µs")

# 2. OpenCV
ifm = np.random.randint(1, 20, (depth, height, width, ci)).astype(np.float32)
ker = np.random.randint(-50, 50, (3, 3, ci, co)).astype(np.float32)
ker_f = np.flip(ker, axis=(0, 1))

def run_cv():
    out = np.zeros((depth, height, width, co), dtype=np.float32)
    for d in range(depth):
        ps = [ifm[max(0,d-1)], ifm[d], ifm[min(depth-1,d+1)]]
        for oc in range(co):
            acc = np.zeros((height, width), dtype=np.float32)
            for ic in range(ci):
                for kd in range(3):
                    acc += cv2.filter2D(ps[kd][:,:,ic], -1, ker_f[:,:,ic,oc], borderType=cv2.BORDER_REPLICATE)
            out[d,:,:,oc] = acc
    return out

for _ in range(2):
    run_cv()
t = []
for _ in range(5):
    s = time.perf_counter()
    run_cv()
    t.append((time.perf_counter()-s)*1e6)
cv_time = np.mean(t)
print(f"OpenCV CPU:       {cv_time:>8.0f}µs ± {np.std(t):.0f}µs")

# NPU helper
def bench_npu(xclbin, insts):
    k = NPUKernel(xclbin, insts, kernel_name="MLIR_AIE")
    h = DefaultNPURuntime.load(k)
    ifm_npu = np.random.randint(1, 20, (depth, ci, height, width), dtype=np.uint8)
    wts_npu = np.random.randint(-50, 50, (co, ci, 3, 3, 3), dtype=np.int8)
    ifm_r = np.zeros((depth, 1, height, 8, width), dtype=np.uint8)
    for d in range(depth):
        for hh in range(height):
            for c in range(8):
                for w in range(width):
                    ifm_r[d, 0, hh, c, w] = ifm_npu[d, c, hh, w]
    wts_r = np.zeros((1, 1, 3, 3, 3, 8, 8), dtype=np.int8)
    for kd in range(3):
        for kh in range(3):
            for kw in range(3):
                for ic in range(ci):
                    for oc in range(co):
                        wts_r[0, 0, kd, kh, kw, ic, oc] = wts_npu[oc, ic, kd, kh, kw]
    buf = [iron.tensor(ifm_r.flatten(), dtype=np.uint8), iron.tensor(wts_r.flatten(), dtype=np.int8), iron.zeros(depth*height*width*co, dtype=np.uint8)]
    for _ in range(3):
        DefaultNPURuntime.run(h, buf)
    t = [DefaultNPURuntime.run(h, buf).npu_time/1000.0 for _ in range(10)]
    return np.mean(t), np.std(t)

# 3. NPU benchmarks
import os
cwd = os.getcwd()
build_path = "build/" if not cwd.endswith("/build") else ""

try:
    npu_1, npu_1_std = bench_npu(f"{build_path}bench_1core.xclbin", f"{build_path}bench_1core_insts.bin")
    print(f"NPU 1-core:       {npu_1:>8.0f}µs ± {npu_1_std:.0f}µs")
except Exception as e:
    print(f"NPU 1-core:       Not available - {e}")
    npu_1 = None

try:
    npu_2, npu_2_std = bench_npu(f"{build_path}bench_2core.xclbin", f"{build_path}bench_2core_insts.bin")
    print(f"NPU 2-core:       {npu_2:>8.0f}µs ± {npu_2_std:.0f}µs")
except Exception as e:
    print(f"NPU 2-core:       Not available - {e}")
    npu_2 = None

try:
    npu_4, npu_4_std = bench_npu(f"{build_path}bench_4core.xclbin", f"{build_path}bench_4core_insts.bin")
    print(f"NPU 4-core:       {npu_4:>8.0f}µs ± {npu_4_std:.0f}µs")
except Exception as e:
    print(f"NPU 4-core:       Not available - {e}")
    npu_4 = None

# Summary
print(f"\n{'='*80}\nSPEEDUP ANALYSIS\n{'='*80}\n")
cpu_base = min(pt_time, cv_time)
print(f"CPU Baseline: {cpu_base:.0f}µs (faster of PyTorch/OpenCV)\n")
if npu_1:
    print(f"NPU 1-core vs CPU:      {cpu_base/npu_1:>6.1f}× faster ⚡")
if npu_2:
    print(f"NPU 2-core vs CPU:      {cpu_base/npu_2:>6.1f}× faster 🚀")
if npu_4:
    print(f"NPU 4-core vs CPU:      {cpu_base/npu_4:>6.1f}× faster 🔥")

if npu_2 and npu_1:
    print(f"\n2-core vs 1-core NPU:   {npu_1/npu_2:>6.2f}× speedup")
if npu_4 and npu_1:
    print(f"4-core vs 1-core NPU:   {npu_1/npu_4:>6.2f}× speedup")
print()
