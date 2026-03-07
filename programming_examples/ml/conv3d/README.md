# Conv3D Example

3D Convolution for NPU using vectorized AIE intrinsics with multi-core spatial parallelism.

## Performance Benchmarks

### NPU vs CPU (8×8×8 volume)

| Platform | Time (µs) | Speedup vs OpenCV |
|----------|-----------|-------------------|
| PyTorch CPU | 41-50 | 74-90× |
| **NPU 1-core** | **566** | **6.5×** ⚡ |
| **NPU 2-core** | **386-450** | **8-10×** 🚀 |
| OpenCV CPU | 3,700 | 1.0× (baseline) |

**Key Finding:** For small volumes, PyTorch CPU is fastest. NPU excels at larger volumes (≥32×32) and batch processing.

### Multi-Core Scaling (Spatial Parallelism)

| Volume | 1-core | 2-core | 4-core | Best Config |
|--------|--------|--------|--------|-------------|
| 8×8×8  | 566µs  | 386µs (1.3×) | - | 2-core |
| 32×32  | 2,615µs | 1,667µs (1.56×) | 1,363µs (1.91×) | **2-core** ⭐ |
| 64×64  | Memory fail | Memory fail | 2,545µs | 4-core |

**Sweet Spot:** 32×32 volumes with 2 cores = 1.56× speedup, 78% efficiency

## Quick Start

### Single-Core (Small Volumes)
```bash
source ../../../ironenv/bin/activate
make
make run_py
```

### Multi-Core Spatial (Medium/Large Volumes)
```bash
# Build 2-core design
python3 conv3d_spatial.py npu2_2col 16 32 32 8 8 > build/spatial.mlir
cd build && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts \
    --no-compile-host --no-xchesscc --no-xbridge \
    --xclbin-name=spatial.xclbin --npu-insts-name=spatial_insts.bin spatial.mlir
cd .. && python3 test_spatial.py build/spatial.xclbin build/spatial_insts.bin 2
```

### Massively Parallel (Up to 32 Cores)
```bash
make -f Makefile.massively_parallel N_CORES=8 all run_py
```

## Features

- ✅ True 3D convolution (3×3×3 kernel)
- ✅ Vectorized AIE intrinsics (`aie::mmul<4,8,8>`)
- ✅ Spatial parallelism (2-32 cores)
- ✅ Parallel shim DMA (up to 8 channels)
- ✅ Validated against PyTorch & OpenCV
- ⚡ 30× faster than scalar, up to 10× faster than OpenCV

## Implementations

- **`conv3d.py`**: Single-core vectorized (best for ≤16×16)
- **`conv3d_spatial.py`**: 2-4 core spatial (best for 32×32)
- **`conv3d_massively_parallel.py`**: 8-32 cores (for ≥64×64)

See `BENCHMARK_RESULTS.md` for detailed analysis.
