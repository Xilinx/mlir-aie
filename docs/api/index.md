<!-- Copyright (C) 2024-2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# API & Internals

This section documents the full IRON / MLIR-AIE programming surface, from the
high-level Python API you write designs in down to the low-level MLIR dialects
and C++ compiler internals.

The stack is organized in layers. Most users only need the top layer; the lower
layers are here for advanced designs, op-by-op construction, and compiler work.

## The layers

| Layer | What it is | When you reach for it |
|-------|------------|-----------------------|
| **IRON** (`aie.iron`) | The high-level Python API. Every object is *resolvable* — it lowers to MLIR when compiled: `@iron.jit`, `Worker`, `ObjectFifo`, `Runtime`, `Program`, `Buffer`, `Kernel`, the tensor factories, and advanced primitives like `Flow`, `TileDma`, and `Lock`. | Writing NPU designs. Start here. |
| **Dialect op wrappers** (`aie.dialects.*`) | The low-level Python layer: thin wrappers around individual MLIR ops of the `aie` / `aiex` / `aievec` dialects. This is what `aie.iron` lowers *to*. | Op-by-op construction, custom lowerings, reading emitted MLIR. |
| **C++ AIE kernels** (`aie_kernels`, AIE API) | The per-core compute code that runs *on* an AIE tile, written in C++ with the [AIE API](https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/index.html) header library or low-level intrinsics. | Writing or tuning the vectorized math a `Worker` runs. |
| **Utilities** | `taplib` tensor access patterns, the pre-built Python kernel library, and host-runtime helpers. | Tiling / streaming descriptors, ready-made kernels, host glue. |
| **MLIR / C++** | The `aie` / `aiex` / `aievec` / `adf` dialect definitions, their passes, and the generated C++ API. | Compiler internals, custom passes, dialect op reference. |

## Python API

<div class="grid cards" markdown>

-   :material-rocket-launch: **[IRON (`aie.iron`)](iron.md)**

    ---

    The high-level user API. `Program`, `Worker`, `ObjectFifo`, `Runtime`,
    `Buffer`, `Kernel`, `@iron.jit`, the tensor factories, plus advanced
    resolvable primitives (`Flow`, `TileDma`, `Lock`).

-   :material-tune: **[Dialect op wrappers (`aie.dialects.*`)](dialect_wrappers.md)**

    ---

    The low-level layer IRON lowers to: direct Python wrappers around the
    `aie` / `aiex` / `aievec` MLIR operations.

-   :material-grid: **[Tensor Access Patterns (taplib)](taplib.md)**

    ---

    `TensorAccessPattern`, `TensorAccessSequence`, and `TensorTiler2D` for
    describing how data is tiled and streamed.

-   :material-library: **[Python Kernel Library](kernels.md)**

    ---

    Pre-built `iron.kernels` wrappers for element-wise, reduction, linalg,
    convolution, activation, and vision operations.

-   :material-code-braces: **[C++ AIE kernels](aie_kernels.md)**

    ---

    The C++ compute code that runs on a tile — the `aie_kernels` library and
    the AIE API header library used to write it.

</div>

## Low-level MLIR / C++

The compiler layer beneath the Python API. See the dialect and pass reference
pages in the **MLIR / C++** group of this section:

- [AIE Dialect](../AIEDialect.md), [AIEX Dialect](../AIEXDialect.md),
  [AIEVec Dialect](../AIEVecDialect.md) — op-by-op reference.
- [AIE Passes](../AIEPasses.md), [AIEX Passes](../AIEXPasses.md),
  [AIEVec Passes](../AIEVecPasses.md) — registered compiler passes.
- [C++ API (Doxygen)](cpp_doxygen.md) — generated reference for the dialect
  C++ API, pass infrastructure, and the AIE runtime library.

---

New to IRON? Start with the [Programming Guide](../programming_guide/README.md)
for worked examples, then use this section as a reference.
