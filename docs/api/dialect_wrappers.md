<!-- Copyright (C) 2024-2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Dialect op wrappers (`aie.dialects.*`)

This is the low-level Python layer: thin wrappers around the individual MLIR
operations of the `aie`, `aiex`, and `aievec` dialects. These are what the
high-level [`aie.iron`](iron.md) objects lower **to** — when a `Worker`,
`ObjectFifo`, or `Runtime` resolves, it emits ops from exactly these modules.

Most designs never import these directly. Reach for them when you are writing a
design op-by-op, building a custom lowering, or reading MLIR that the JIT
emitted and want to know which Python call produced a given op.

## How the wrappers are generated

Each op wrapper corresponds one-to-one with an operation defined in the
dialect's TableGen (`*.td`) files. The bindings come in two forms:

- **Generated bindings** (`_aie_ops_gen`, `_aiex_ops_gen`, `_aievec_ops_gen`)
  are produced at build time by `mlir-tblgen` directly from the op definitions.
  There is one class per op, named after the op (e.g. `ObjectFifoCreateOp`,
  `NpuDmaMemcpyNdOp`).
- **Hand-written conveniences** in `aie.py` / `aiex.py` wrap those generated
  classes with friendlier constructors and Python-side helpers — for example
  `object_fifo` (wrapping `ObjectFifoCreateOp`), `core` (wrapping `CoreOp`),
  `buffer` (wrapping `BufferOp`), and `runtime_sequence`.

Because the generated `*_ops_gen` modules only exist inside a built tree, this
page is a conceptual guide rather than an auto-generated symbol dump. For the
authoritative, op-by-op reference, use the dialect pages under
**MLIR / C++** (below).

## The three dialects

| Module | Dialect | What it wraps |
|--------|---------|---------------|
| `aie.dialects.aie` | `aie` | Core structural ops: tiles, buffers, locks, cores, ObjectFifos, flows, and packet flows. |
| `aie.dialects.aiex` | `aiex` | Extended / runtime ops: NPU DMA memcpy, sync, RTP writes, DMA task configuration, and the `runtime_sequence`. |
| `aie.dialects.aievec` | `aievec` | AIE vector ops used by the vectorizing compiler passes. |

```python
# Op-by-op construction with the wrappers (rarely needed directly):
from aie.dialects.aie import tile, core, object_fifo, end
from aie.dialects.aiex import runtime_sequence, npu_dma_memcpy_nd
```

## Reference

The op wrappers are a Python surface over the dialects. For the definitive
per-op documentation — operands, results, attributes, and assembly format —
see the dialect and pass references in the **MLIR / C++** group of this
section:

- [AIE Dialect](../AIEDialect.md), [AIEX Dialect](../AIEXDialect.md),
  [AIEVec Dialect](../AIEVecDialect.md) — op-by-op reference.
- [AIE Passes](../AIEPasses.md), [AIEX Passes](../AIEXPasses.md),
  [AIEVec Passes](../AIEVecPasses.md) — registered compiler passes.
- [C++ API (Doxygen)](cpp_doxygen.md) — the generated C++ dialect API.
