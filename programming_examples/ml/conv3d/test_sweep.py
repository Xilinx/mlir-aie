import numpy as np, time, torch, torch.nn as nn, aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime

configs = [
    (3, 32, 32, 1, "q_d3_s32_c1"),
    (3, 32, 32, 2, "q_d3_s32_c2"),
    (3, 32, 32, 4, "q_d3_s32_c4"),
    (3, 64, 64, 4, "q_d3_s64_c4"),
]

print(f"\n{'='*90}")
print(f"Conv3D Performance: NPU vs PyTorch CPU")
print(f"{'='*90}\n")
print(f"{'Volume':<15} {'PyTorch CPU':>15} {'NPU (cores)':>20} {'NPU Speedup':>15} {'Multi-Core':>15}")
print(f"{'-'*90}")

for depth, height, width, cores, name in configs:
    ci, co = 8, 8
    
    # PyTorch CPU
    model = nn.Conv3d(ci, co, kernel_size=(1,3,3), padding=0, bias=False)
    model.eval()
    inp = torch.randint(1, 20, (1, ci, depth, height, width)).type(torch.FloatTensor)
    wt = torch.randint(-50, 50, (co, ci, 1, 3, 3)).type(torch.FloatTensor)
    model.weight.data.copy_(wt)
    inp_pad = torch.nn.functional.pad(inp, (1,1,1,1,0,0), mode='replicate')
    for _ in range(5): _ = model(inp_pad)
    t = [(time.perf_counter(), model(inp_pad), time.perf_counter()) for _ in range(20)]
    pt_time = np.mean([(x[2]-x[0])*1e6 for x in t])
    
    # NPU
    try:
        k = NPUKernel(f"build/{name}.xclbin", f"build/{name}_insts.bin", kernel_name="MLIR_AIE")
        h = DefaultNPURuntime.load(k)
        np.random.seed(42)
        ifm_r = np.random.randint(1, 20, (depth, 1, height, 8, width), dtype=np.uint8)
        wts_r = np.random.randint(-50, 50, (1, 1, 3, 3, 3, 8, 8), dtype=np.int8)
        buf = [iron.tensor(ifm_r.flatten(), dtype=np.uint8), iron.tensor(wts_r.flatten(), dtype=np.int8), iron.zeros(depth*height*width*co, dtype=np.uint8)]
        for _ in range(5): DefaultNPURuntime.run(h, buf)
        npu_times = [DefaultNPURuntime.run(h, buf).npu_time/1000.0 for _ in range(20)]
        npu_time = np.mean(npu_times)
        
        speedup = pt_time / npu_time
        vol_str = f"{depth}×{height}×{width}"
        npu_str = f"{npu_time:.0f}µs ({cores}c)"
        
        # Multi-core comparison (compare to 1-core for same volume)
        if cores == 1:
            baseline_1core = npu_time
            mc_str = "-"
        else:
            if vol_str == "3×32×32" and cores > 1:
                # Compare to baseline from first config
                mc_speedup = baseline_1core / npu_time if 'baseline_1core' in locals() else 0
                mc_str = f"{mc_speedup:.2f}×"
            else:
                mc_str = "-"
        
        print(f"{vol_str:<15} {pt_time:>12.0f}µs {npu_str:>20} {speedup:>12.1f}× {mc_str:>15}")
    except Exception as e:
        print(f"{depth}×{height}×{width:<10} {pt_time:>12.0f}µs {'ERROR':>20} {'-':>12} {'-':>15}")

print(f"{'-'*90}\n")
print("Summary:")
print("  - PyTorch running on CPU (no GPU transfer overhead)")
print("  - NPU times include PCIe transfer + compute")
print("  - Larger volumes show better NPU scaling")
print()
