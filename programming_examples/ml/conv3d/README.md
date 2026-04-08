# Conv3D - 3D Convolution for Video/Volumetric Data

High-performance 3D convolution on AMD Ryzen AI NPU with width tiling for large frames (up to 1024×1024).

## Performance

### NPU vs CPU — Steady-State (Ryzen AI 9 HX 370)

| Volume (D×H×W) | CPU 12T (f32) | NPU 32-core (u8) | Speedup |
|-----------------|---------------|-------------------|---------|
| 8×256×256       | 108 ms        | **4.5 ms**        | **24×** |
| 8×512×512       | 40 ms         | **16.8 ms**       | **2.4×** |
| 8×1024×1024     | 148 ms        | **8.4 ms**        | **18×** |

NPU times are steady-state (warmup excluded). CPU uses PyTorch float32, 12 threads.

### Scaling (8-core IRON vs 32-core Memtile)

| Volume | 8-core | 32-core | Speedup |
|--------|--------|---------|---------|
| 8×256×256 | 17.7 ms | 4.5 ms | 3.9× |
| 8×512×512 | 66.5 ms | 16.8 ms | 4.0× |

## Quick Start

```bash
source ../../../ironenv/bin/activate
source ../../../utils/env_setup.sh ../../../ /opt/xrt

# 8-core, 64×64 (no tiling needed)
make massively_parallel_8core depth=8 height=64 width=64
python3 test.py -x build/mp_8core.xclbin -i build/mp_8core_insts.bin -k MLIR_AIE -d 8 -ht 64 -wd 64

# 32-core, 512×512 (width tiling, memtile split/join)
make memtile_32core_tiled depth=8 width=512 height=512
python3 test.py -x build/mt32_tiled.xclbin -i build/mt32_tiled_insts.bin -k MLIR_AIE -d 8 -ht 512 -wd 512
```

## Designs

| File | Cores | Tiling | Best For |
|------|-------|--------|----------|
| `conv3d.py` | 1-4 | None | Small volumes (≤64×64) |
| `conv3d_massively_parallel.py` | 1-8 | Auto width tiling | Medium frames (≤512×512) |
| `conv3d_32core_tiled_fixed.py` | 32 | Width tiling + memtile split/join | Large frames (512-1024) |

### Width Tiling

For large frames, per-core buffers exceed L1 (64KB). Width tiling splits each row into tiles that fit:

- `tile_width` auto-calculated as largest power-of-2 fitting L1
- Core loop: `for depth: for tile:` with 4D strided DMA
- Backward compatible: when `tile_width >= width`, no tiling occurs

### 32-Core Memtile Architecture

- 8 columns × 4 rows, memtile split/join for data distribution
- Input: shim → memtile (combined) → split → 4 cores per column
- Output: 4 cores → join → memtile (combined) → shim
- Proper buffer sizing: L1 77%, memtile 37% at default config
- Per-plane DMA BDs for frames >512×512 (shim stride limit)

## Data Layout

- Input/Output: `D{C/8}HW{C8}` — kernel expects `(y*W+x)*8+ic` indexing
- Weights: `{O/8}{I/8}KDHW{I8}{O8}` (3×3×3 kernel)

## Technical Notes

- **Kernel:** 3×3×1 per depth plane, 2D conv with replicate padding
- **Quantization:** uint8 activations, int8 weights, scale=10
- **DMA stride limit:** Shim supports 20-bit word strides (~4MB). Frames >512×512 use per-plane BD splits.
- **BD limit:** Memtile has 24 BDs per channel. FIFO depth=2 with 4-way split/join uses 16 BDs.
