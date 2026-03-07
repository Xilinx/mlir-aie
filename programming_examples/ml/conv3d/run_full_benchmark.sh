#!/bin/bash
set -e

echo "Building Conv3D designs for video benchmark..."
echo "Volume: 16 frames × 32×32 pixels, 8 channels"

# Build 1-core
echo "[1/3] Building 1-core..."
python3 conv3d_spatial.py npu2 16 32 32 8 8 > build/bench_1core.mlir
cd build && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --no-xchesscc --no-xbridge --xclbin-name=bench_1core.xclbin --npu-insts-name=bench_1core_insts.bin bench_1core.mlir > /dev/null 2>&1 && cd ..
echo "  ✓ 1-core built"

# Build 2-core
echo "[2/3] Building 2-core spatial..."
python3 conv3d_spatial.py npu2_2col 16 32 32 8 8 > build/bench_2core.mlir
cd build && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --no-xchesscc --no-xbridge --xclbin-name=bench_2core.xclbin --npu-insts-name=bench_2core_insts.bin bench_2core.mlir > /dev/null 2>&1 && cd ..
echo "  ✓ 2-core built"

# Build 4-core
echo "[3/3] Building 4-core spatial..."
python3 conv3d_spatial.py npu2_4col 16 32 32 8 8 > build/bench_4core.mlir
cd build && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --no-xchesscc --no-xbridge --xclbin-name=bench_4core.xclbin --npu-insts-name=bench_4core_insts.bin bench_4core.mlir > /dev/null 2>&1 && cd ..
echo "  ✓ 4-core built"

echo ""
echo "Running benchmarks..."

# Create benchmark script
cat > build/run_bench.py << 'EOF'
import numpy as np, time, torch, torch.nn as nn, cv2
import aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime

depth, height, width, ci, co = 16, 32, 32, 8, 8
print(f"\n{'='*80}\nVideo Conv3D Benchmark: {depth}×{height}×{width}, {ci}→{co} channels\n{'='*80}\n")

# PyTorch
model = nn.Conv3d(ci, co, kernel_size=(1,3,3), padding=0, bias=False)
model.eval()
inp = torch.randint(1, 20, (1, ci, depth, height, width)).type(torch.FloatTensor)
wt = torch.randint(-50, 50, (co, ci, 1, 3, 3)).type(torch.FloatTensor)
model.weight.data.copy_(wt)
inp_pad = torch.nn.functional.pad(inp, (1,1,1,1,0,0), mode='replicate')
for _ in range(3): _ = model(inp_pad)
t = []; [t.append((lambda: (time.perf_counter(), model(inp_pad), time.perf_counter()))()[2]-lambda x:x[0]()) for _ in range(10)]
pt_time = np.mean(t) * 1e6
print(f"PyTorch CPU:      {pt_time:>8.0f}µs")

# OpenCV
ifm = np.random.randint(1, 20, (depth, height, width, ci)).astype(np.float32)
ker = np.random.randint(-50, 50, (3, 3, ci, co)).astype(np.float32)
ker_f = np.flip(ker, axis=(0, 1))
def run_opencv():
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
for _ in range(2): run_opencv()
t = []; 
for _ in range(5):
    s = time.perf_counter(); run_opencv(); t.append((time.perf_counter()-s)*1e6)
cv_time = np.mean(t)
print(f"OpenCV CPU:       {cv_time:>8.0f}µs")

# NPU tests
def bench_npu(xclbin, insts, name):
    try:
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
        for _ in range(3): DefaultNPURuntime.run(h, buf)
        t = [DefaultNPURuntime.run(h, buf).npu_time/1000.0 for _ in range(10)]
        return np.mean(t)
    except Exception as e:
        print(f"{name:<18} Not available")
        return None

npu_1 = bench_npu("bench_1core.xclbin", "bench_1core_insts.bin", "NPU 1-core")
if npu_1: print(f"NPU 1-core:       {npu_1:>8.0f}µs")

npu_2 = bench_npu("bench_2core.xclbin", "bench_2core_insts.bin", "NPU 2-core")  
if npu_2: print(f"NPU 2-core:       {npu_2:>8.0f}µs")

npu_4 = bench_npu("bench_4core.xclbin", "bench_4core_insts.bin", "NPU 4-core")
if npu_4: print(f"NPU 4-core:       {npu_4:>8.0f}µs")

# Summary
print(f"\n{'='*80}\nSPEEDUP ANALYSIS\n{'='*80}\n")
cpu_base = min(pt_time, cv_time)
if npu_1: print(f"NPU 1-core vs CPU:     {cpu_base/npu_1:>6.1f}× faster")
if npu_2: print(f"NPU 2-core vs CPU:     {cpu_base/npu_2:>6.1f}× faster")
if npu_4: print(f"NPU 4-core vs CPU:     {cpu_base/npu_4:>6.1f}× faster")
if npu_2 and npu_1: print(f"\n2-core vs 1-core NPU:  {npu_1/npu_2:>6.2f}× speedup")
if npu_4 and npu_1: print(f"4-core vs 1-core NPU:  {npu_1/npu_4:>6.2f}× speedup")
print()
EOF

cd build && python3 run_bench.py
