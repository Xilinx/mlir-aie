<!-- Copyright (C) 2024-2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# C++ AIE kernels

This is the compute code that runs *on* an individual AIE tile. A
[`Worker`](iron.md#worker) places one of these functions on a compute tile; the
kernel does the actual vectorized math on the data streamed to it by
ObjectFifos.

Kernels are written in C++ and compiled for the AIE core (by Peano or, when
available, `xchesscc`). They are a separate layer from the Python API: IRON
handles placement and data movement, while the kernel handles the arithmetic.

## Where they live

The [`aie_kernels`](../../aie_kernels/) directory holds a library of example
kernels, organized by target:

| Directory | Target | Notes |
|-----------|--------|-------|
| [`generic`](../../aie_kernels/generic/) | Any AIE | Portable C — runs on any generation at varying performance. |
| [`aie2`](../../aie_kernels/aie2/) | AIE2 / XDNA | The largest set: eltwise, gemm, reduction, conv, vision. |
| [`aie2p`](../../aie_kernels/aie2p/) | AIE2P / XDNA 2 | Kernels tuned for the newer architecture. |

See the [`aie_kernels` README](../../aie_kernels/) for the full per-kernel
catalog (name, coding style, purpose, datatypes).

## How they are written

Kernels use one of three coding styles, in decreasing order of portability:

- **AIE API** — a C++ header-only library (`#include <aie_api/aie.hpp>`) of
  vector types and operations that lower to efficient per-generation
  intrinsics. This is the recommended style; see the
  [AIE API User Guide](https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/index.html).
- **Low-level intrinsics** — architecture-specific intrinsics used directly
  when the AIE API does not expose a needed operation.
- **Plain C** — scalar code with no vectorization, portable across
  generations (the `generic` kernels).

```cpp
#include <aie_api/aie.hpp>

template <typename T>
void scale_vectorized(T *__restrict a, T *__restrict c, int32_t factor,
                      const int32_t N) {
  event0();
  for (int i = 0; i < N; i += 16) {
    aie::vector<T, 16> v = aie::load_v<16>(a + i);
    aie::store_v(c + i, aie::mul(v, factor));
  }
  event1();
}
```

## Using a kernel from IRON

A C++ kernel is bound into a design as a [`Kernel`](iron.md#kernels) (a
pre-compiled object file) or an `ExternalFunction` (C/C++ source compiled at
JIT time), then handed to a `Worker`. The
[kernel vectorization walkthrough](../programming_guide/section-4/section-4c/README.md)
in the Programming Guide works through writing and tuning one of these kernels.

For ready-made kernels callable directly from Python without writing C++, see
the [Python Kernel Library](kernels.md).
