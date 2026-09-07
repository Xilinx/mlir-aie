<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# AMD XDNA NPU Architecture Reference

## Devices

| Family | Architecture | NPU columns × rows (compute) | IRON device class |
|--------|--------------|------------------------------|-------------------|
| Phoenix / Hawk Point | **AIE2** a.k.a. XDNA | 4 × 4 (NPU1) | `NPU1`, `NPU1Col1`, `NPU1Col2`, `NPU1Col4` |
| Strix / Krackan Point | **AIE2P** a.k.a. XDNA2 | 8 × 4 (NPU2) | `NPU2`, `NPU2Col1`, `NPU2Col4`, `NPU2Col8` |

```python
from aie.iron.device import NPU1, NPU1Col1, NPU2, NPU2Col4
import aie.iron as iron

dev = iron.get_current_device()  # auto-detect the attached NPU at runtime
# or pick explicitly:
dev = NPU2()                     # all columns of an AIE2P NPU
dev = NPU1Col1()                 # just one column of an AIE2 NPU
```

## Tile types (per column)

Bottom to top:

| Row | Tile | Purpose |
|-----|------|---------|
| 0 | **Shim** | DMA between external (DDR) memory and the AIE array |
| 1 | **Mem** | L2 scratchpad shared between Shim and the column's compute tiles |
| 2..5 | **Compute** | The actual cores running C++ kernels; each has a small L1 |

Throughout these skills, **compute tile**, **core**, and **AIE tile** all refer to these row-2..5 tiles.

Use `tile(col, row)` (lower-level API) only when you need explicit coordinates. In the high-level API, the compiler picks placement; you control parallelism by spawning N `Worker`s.

## Memory hierarchy & sizes (rule-of-thumb)

| Level | Size | Latency | Who fills it |
|-------|------|---------|--------------|
| L1 (per compute tile) | ~64 KB | 1 cycle | ObjectFifo buffer + stack |
| L2 (per mem tile) | ~512 KB | a few cycles | Mem tile DMA / `forward` / `split` / `join` |
| L3 (external DDR) | GBs | 100s of cycles | Shim DMA via ObjectFifoHandle `fill`/`drain` |

Keep working sets in L1; spill to L2 via Mem tile; reach DDR only at the start/end of a phase.

## Vector registers & widths

AIE2/AIE2P has 512-bit vector registers. The number of lanes per `aie::vector<T, N>` follows `N = 512 / bitwidth(T)` for the natural width, but the AIE API supports smaller and (with `grow<>`) larger widths for software pipelining flexibility.

| Type | Bits | Natural lanes / vector | Accumulator |
|------|------|------------------------|-------------|
| `int8_t`, `uint8_t` | 8 | 64 | `acc32` |
| `int16_t`, `uint16_t` | 16 | 32 | `acc32` (or `acc64` for wide multiplies) |
| `int32_t`, `uint32_t` | 32 | 16 | `acc64` |
| `bfloat16` | 16 | 32 | `accfloat` (or `accauto` in MMUL) |
| `float` (fp32) | 32 | 16 | `accfloat` |

Default to the **natural lane count** for the dtype (the middle column above: 32 for `bfloat16`/16-bit, 64 for `int8`, 16 for 32-bit). This is what the examples in this skill use. Drop to a narrower width only if the compiler reports register spills at the natural width.

## MMUL shapes

The `aie::mmul<r, s, t, T_in, T_w, accauto>` intrinsic performs an `r × s` × `s × t` matrix multiply per call into an `r × t` accumulator. AIE2 and AIE2P support different shape sets:

| Input × weight → output | AIE2 shape (r×s×t) | AIE2P shape (r×s×t) | Notes |
|-------------------------|--------------------|---------------------|-------|
| `int8 → int8`           | 4×8×8              | 8×8×8               | AIE2P doubles M |
| `int8 → int16`          | 4×8×8              | 8×8×8               | |
| `int8 → int32`          | 4×8×8              | 8×8×8               | |
| `int16 → int16`         | 4×4×4              | 4×4×8               | AIE2P doubles N |
| `int16 → int32`         | 4×4×4              | 4×4×8               | |
| `bf16 → bf16`           | 4×8×4              | 4×8×8               | AIE2P doubles N |
| `bf16 → float32`        | 4×8×4              | 4×8×8               | |

Building with `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` swaps AIE2P's bf16 MMUL from native
mul-acc to BFP16 emulation, which uses an **8×8×8** micro-kernel instead of 4×8×8. Other
(arch, dtype) combinations are unaffected.

**Don't hardcode these in code.** Read the geometry off the kernel you're actually going to
link, so it can't drift from the compiled `.cc`:

```python
r, s, t = kernels.mm(dim_m=64, dim_k=64, dim_n=64,
                     input_dtype=bfloat16, output_dtype=np.float32).mac_dims
```

The table above is for reasoning about the divisibility constraints below; `.mac_dims` is what
belongs in a design. (`kernels.cascade_mm` is a scalar kernel on both architectures — its
`mac_dims` are `(1, 1, 1)` and its buffers must stay plain row-major.)

Divisibility constraints when calling MMUL with outer expansion (`4x2`, `4x4`, etc.):

```cpp
// Example: matmul_vectorized_4x8x4_bf16_bf16 with 4x4 expansion on AIE2
static_assert(m % (4 * r) == 0);   // 4 rows × r=4  ⇒ m % 16 == 0
static_assert(k % s == 0);          // k % 8  == 0
static_assert(n % (4 * t) == 0);   // 4 cols × t=4 ⇒ n % 16 == 0
```

Violate these and you get either a compile error or silently wrong results.

## Routing / placement constraints (low-level API only)

- Each column has limited switch resources; broadcasting one ObjectFifo to N consumers across distant columns may fail to route — prefer per-column producers or `forward()` through mem tiles.
- Compute tiles only have direct DMA to/from their column's mem tile and adjacent compute tiles. Cross-column data goes through mem-tile broadcast.
- The high-level placer respects these; if you hand-place tiles, verify with `ctx.module.operation.verify()`.

## Performance ballpark (per compute tile, peak)

| Op | AIE2 (Phoenix) | AIE2P (Strix) |
|----|----------------|---------------|
| int8 MAC / cycle | 256 | 512 |
| bf16 MAC / cycle | 128 | 256 |
| Clock | ~1.0–1.3 GHz | ~1.5 GHz |

Multiply per-tile peak by `(rows × columns)` for the full array. You won't hit peak without MMUL, restrict pointers, and pipelined loops.

## Hardware limits to remember

- **Tile L1**: ~64 KB. Bigger tensors must be tiled across iterations.
- **ObjectFifo depth**: practical max ~8 in L1; use mem-tile ObjectFifos for deeper buffering.
- **BD (Buffer Descriptor) addressing dimensions are not uniform across tile types.** On AIE2/AIE2P:

  | Tile type | ND dims per BD | BDs per tile | Max transfer length |
  |-----------|----------------|--------------|---------------------|
  | Mem tile | **4** | 48 | 2^17 − 1 |
  | Compute (core) tile | **3** | 16 | 2^14 − 1 |
  | Shim tile | **3** | 16 | 2^32 − 1 |

  This drives real design decisions: a reshape needing 4 `(size, stride)` dimensions **must**
  land on a mem-tile DMA. If a compute tile's `dims_to_stream` or a shim's tap already needs
  3 dimensions, you have no fourth to spend — route L3→L2→L1 and put the reshape on the
  mem-tile `split()`/`join()`/`forward()` instead. For access patterns beyond the limit, chain
  multiple BDs or use runtime repeat counts.

  DMA transforms aren't the only way to reshape/translate data, though: `aie::vshuffle`
  can do in-register lane permutation on the compute tile itself, and being a vector op it
  can overlap with (be masked by) concurrent/pipelined `MAC`s — worth considering as an
  alternative to an extra DMA hop when the reshape is small enough to fit in a shuffle.
- **Workers per Program**: bounded by available compute tiles (16 on NPU1, 32 on NPU2).
