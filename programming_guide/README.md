<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# IRON AIE Application Programming Guide

This is the programming guide for **IRON** — the Python API for programming AMD Ryzen™ AI NPUs. It teaches how to design, run, and optimize code on the AIE-array.

> **First time here?** If you haven't installed the toolchain yet, start with the [installation instructions](../README.md) (driver, XRT, IRON install). Then come back here.
>
> **Want the shortest possible on-ramp?** See the [Mini Tutorial](./mini_tutorial/) — five tiny exercises that get a working design on the NPU in minutes.

-----

## The mental model in 60 seconds

* The NPU is a 2D grid of **AIE tiles**. The interesting ones are **compute tiles** (run code) and **mem tiles** (shared L2 scratchpad). At the edge of the array, **shim tiles** move data to/from main memory.
* Tiles are connected by **stream switches**. The path from main memory to a compute tile is always shim → (mem) → compute, scheduled by per-tile **DMA engines**.
* You describe an NPU program in Python:
    * A **Worker** is the code that runs on one compute tile.
    * An **ObjectFifo** is a streaming channel between two endpoints (host↔tile, tile↔tile). Acquire / release.
    * A **Runtime** sequence is the host-side dance — what tensors get filled into the array, what gets drained back.
* You wrap the whole thing in `@iron.jit`. Calling the decorated function the first time JIT-compiles to an `xclbin` + instruction stream and runs it on the attached NPU. Subsequent calls hit a cache.

```python
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
import numpy as np

@iron.jit
def my_design(a_in: In, b_out: Out):
    of_in  = ObjectFifo(np.ndarray[(1024,), np.dtype[np.int32]], name="in")
    of_out = ObjectFifo(np.ndarray[(1024,), np.dtype[np.int32]], name="out")

    def core_fn(of_in, of_out):
        ai = of_in.acquire(1)
        bo = of_out.acquire(1)
        for i in range_(1024):
            bo[i] = ai[i] + 1
        of_in.release(1); of_out.release(1)

    w = Worker(core_fn, [of_in.cons(), of_out.prod()])

    def sequence(a, b, in_h, out_h):
        in_h.fill(a)
        out_h.drain(b, wait=True)

    rt = Runtime(
        sequence,
        [*[np.ndarray[(1024,), np.dtype[np.int32]]] * 2],
        fn_args=[of_in.prod(), of_out.cons()],
    )

    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

a = iron.arange(1024, dtype=np.int32, device="npu")
b = iron.zeros(1024,  dtype=np.int32, device="npu")
my_design(a, b)              # compile + run + sync back
```

`my_design.as_mlir(a, b)` prints MLIR for the active target, binding an attached runtime target when one is available. `python3 my_design.py --dev npu --emit-mlir` emits an offline target without touching the NPU.

## Where to go next

| You want… | Read |
|---|---|
| Quickest possible on-ramp (5 small kernels) | [Mini Tutorial](./mini_tutorial/) |
| Install + driver setup | [Installation instructions](../README.md) |
| The full guide, top to bottom | [Section 0](./section-0/) onward |
| The shortest end-to-end working program | [Section 3 — My First Program](./section-3/) |
| Optimizing — measure, then tune | [Section 4](./section-4/) (timers, trace, vectorization) |
| Catalog of worked example designs | [Section 5](./section-5/) (basic) and [Section 6](./section-6/) (vision + ML) |
| The implicit-MLIR-context error you just hit | [Implicit MLIR context](./implicit_mlir_context.md) |
| Setting-by-setting configuration (cache dir, tensor backend, log level) | [Configuration options](./iron_configuration.md) |
| What happens between `@iron.jit` and the NPU running | [Compilation stages](./compilation_stages.md) |
| Ready-made compute kernels (matmul, conv, eltwise, vision) | [Kernel library](./kernels_library.md) |

## Sections

* [Section 0 — Getting set up for IRON](./section-0/)
* [Section 1 — Basic AI Engine building blocks](./section-1/) (Worker, Buffer, Runtime, Program, `@iron.jit`)
* [Section 2 — Data movement (ObjectFifos)](./section-2/) (deep dive; 2a–2h)
* [Section 3 — My First Program](./section-3/) (end-to-end vector × scalar, JIT + decomposed XRT)
* [Section 4 — Performance measurement & vector programming](./section-4/) (timers → trace → kernel vectorization)
* [Section 5 — Example vector designs](./section-5/) (catalog)
* [Section 6 — Larger example designs](./section-6/) (vision, ML)

## JIT compile + cache

`@iron.jit` caches compiled artifacts by `(MLIR bytecode + compile-time kwargs)`. The first call to a design compiles; subsequent calls with the same kwargs reuse the cache.

* Cache directory: `${NPU_CACHE_HOME:-~/.npu/cache}`. Set `NPU_CACHE_HOME=/tmp/iron_cache` (or anywhere) to override.
* To force a clean build, `rm -rf "$NPU_CACHE_HOME"` (or the per-design `build/` directory the Makefile writes to).
* First-call compile time depends on design complexity. A simple vector_scalar_mul-style design compiles in single-digit seconds; multi-core matmul or convolution can take 30s+. Subsequent calls with a warm cache are essentially instant.

More [configuration options](./iron_configuration.md) (tensor backend, XRT context cache, log level) are available.

## Glossary

The vocabulary IRON and this documentation use, grouped by topic. Where a term maps to a Python class, the name links to its [API reference](../docs/api/index.md).

### Hardware

| Term | Definition |
|------|------------|
| **NPU** | Neural Processing Unit — the AI Engine array inside AMD Ryzen™ AI processors. Older BIOS menus may label it **IPU**; it is the same hardware. |
| **AI Engine (AIE)** | The vector-capable processor architecture that makes up the NPU. Exposed as a 2D spatial array of tiles. |
| **AIE array** | The 2D grid of AIE tiles, connected by stream switches and per-tile DMA engines. Also written *AIE-array*. |
| **Tile** | One cell of the AIE array. Every tile has a `(column, row)` coordinate. Comes in three kinds — compute, mem, and shim — described below. |
| **Compute tile** | The physical AIE tile that runs your kernel code (e.g. `Tile(0, 2)`). Prefer "compute tile" over "core", which is the same hardware but used inconsistently across older docs. |
| **Mem tile** | An L2 scratchpad tile (typically row 1) for staging data between shim and compute tiles. Prefer "mem tile" over "memory tile". |
| **Shim tile** | A row-0 tile that bridges the AIE array to main memory (DDR) via shim DMA. |
| **Stream switch** | The programmable interconnect that routes AXI streams between tiles. |
| **DMA** | Direct Memory Access engine. Each tile has DMA channels that move data between the tile's local memory and the AXI stream, following programmable access patterns. |
| **XDNA™ / XDNA™ 2** | AMD's NPU hardware generations. XDNA (AIE-ML / AIE2) is in Phoenix/HawkPoint; XDNA 2 (AIE2P) is in Strix/Krackan. |
| **Versal™** | AMD's adaptive SoC family that also contains AI Engine tiles. A secondary IRON target (VCK190); supported but not part of the actively-tested NPU path. |

### Devices

| Term | Definition |
|------|------------|
| **npu1** | The device for Ryzen™ AI Phoenix (e.g. 7940HS) and HawkPoint (e.g. 8040HS) SoCs. AIE2 / XDNA. |
| **npu2** | The device for Ryzen™ AI Strix, Strix Halo, and Krackan Point SoCs. AIE2P / XDNA 2. |
| **npu1_Ncol / npu2_Ncol** | A device representing a physical column-partition of an NPU (e.g. `npu2_4col`), including a DMA shim tile per column. |

### IRON API

| Term | Definition |
|------|------------|
| [**Worker**](../docs/api/iron.md#iron.worker.Worker) | The IRON object describing the code that runs on one compute tile. Prefer "Worker" over "process" or "task". |
| [**ObjectFifo**](../docs/api/iron.md#iron.dataflow.objectfifo.ObjectFifo) | The synchronized streaming-data primitive between two endpoints (host↔tile or tile↔tile). Producers and consumers `acquire` / `release` its elements. Reserve "channel" for AXI stream channels. |
| [**ObjectFifoHandle**](../docs/api/iron.md#iron.dataflow.objectfifo.ObjectFifoHandle) | A producer or consumer handle to an ObjectFifo, obtained via `of.prod()` / `of.cons()`. Worker core functions call `acquire` / `release` on it. |
| [**Runtime**](../docs/api/iron.md#iron.runtime.runtime.Runtime) | The host-side description of `fill` / `drain` operations, defined by the body passed to `Runtime(seq, inputs, fn_args)`. Workers are passed to `Program(workers=...)` rather than started from the body. Distinct from "host code" (the C++/Python testbench that calls the design). |
| [**Program**](../docs/api/iron.md#iron.program.Program) | The top-level container that binds a device and a Runtime and resolves the design to MLIR. |
| [**Buffer**](../docs/api/iron.md#iron.buffer.Buffer) | A named memory region on a tile, accessible by both Workers and the Runtime (often used for runtime parameters). |
| [**Kernel**](../docs/api/iron.md#iron.kernel.Kernel) / [**ExternalFunction**](../docs/api/iron.md#iron.kernel.ExternalFunction) | Wrappers for AIE core functions: `Kernel` for a pre-compiled object file, `ExternalFunction` for C/C++ source compiled at JIT time. |
| [**TensorAccessPattern (TAP)**](../docs/api/taplib.md) | A description of how a tensor is sliced and streamed to/from the NPU across multiple DMA transfers. Passed as `tap=` to `fifo.fill()` / `fifo.drain()`. |
| [**Flow**](../docs/api/iron.md#iron.dataflow.flow.Flow) / [**PacketFlow**](../docs/api/iron.md#iron.dataflow.flow.PacketFlow) | Lower-level explicit-routing primitives: `Flow` for circuit-switched routes, `PacketFlow` for packet-switched routes with caller-controlled packet IDs. |
| [**TileDma**](../docs/api/iron.md#iron.dataflow.tile_dma.TileDma) | A lower-level explicit per-tile DMA program, used when the ObjectFifo abstraction hides too much. |
| [**Runtime sequence**](../docs/api/iron.md#iron.runtime.runtime.Runtime.sequence) | The sequence body passed to `Runtime(seq, inputs, fn_args)` in which `fill` / `drain` operations are declared. |

### Compilation

| Term | Definition |
|------|------------|
| **`@iron.jit`** | The recommended entry point. Decorates a Python function that returns a `Program`; the first call JIT-compiles the design and runs it on the NPU, caching the result. |
| **Peano** | The LLVM-based compiler ([`llvm-aie`](https://github.com/Xilinx/llvm-aie)) that generates AI Engine core code. Extends LLVM with the AIE processor as a target, usable via `clang`. |
| **xchesscc** | AMD's proprietary AIE compiler, available via Vitis™ AIE Essentials. An alternative to Peano; not required for XDNA/XDNA 2 targets. |
| **xclbin** | The compiled binary container that XRT loads on the host to configure the NPU partition and stream per-core ELFs down to the tiles. |
| **instruction stream** | The host-to-NPU control instructions generated alongside the `xclbin`. |
| **MLIR** | [Multi-Level Intermediate Representation](https://mlir.llvm.org/) — the LLVM compiler infrastructure this toolchain is built on. |
| **Dialect** | An MLIR extension defining ops and types for a domain. This project defines the `aie`, `aiex`, `aievec`, and `adf` dialects. |
| **`aie` dialect** | The core MLIR dialect for AIE tiles, buffers, locks, and stream switches. Written lowercase; the legacy uppercase `AIE.` form is deprecated. |
| **XRT** | The [Xilinx Runtime](https://github.com/Xilinx/XRT) — the host-side runtime that loads the `xclbin` and drives the NPU. |
| **XDNA™ driver** | The Linux kernel driver (`amdxdna`) that exposes the NPU to XRT. |

### Data movement

| Term | Definition |
|------|------------|
| **acquire / release** | The ObjectFifo synchronization pair. A producer or consumer `acquire`s elements to read/write them, then `release`s them so the other endpoint can proceed. |
| **fill / drain** | Runtime operations: `fill` streams a host tensor into a producer ObjectFifo; `drain` reads from a consumer ObjectFifo back into a host tensor. |
| **buffer descriptor (BD)** | A DMA descriptor entry describing one transfer: buffer, offset, length, and access pattern. |
| **access pattern** | A multi-dimensional `(size, stride)` description of the order in which a DMA reads or writes elements — the mechanism behind data-layout transformations. |
| **broadcast / split / join** | ObjectFifo data-movement patterns: broadcast sends one producer's data to many consumers; split/join distribute or gather data across multiple sub-fifos. |

-----

## Further reading

* E. Hunhoff, J. Melber, K. Denolf, A. Bisca, S. Bayliss, S. Neuendorffer, J. Fifield, J. Lo, P. Vasireddy, P. James-Roxby, E. Keller. "[Efficiency, Expressivity, and Extensibility in a Close-to-Metal NPU Programming Interface](https://arxiv.org/abs/2504.18430)". In 33rd IEEE International Symposium On Field-Programmable Custom Computing Machines, May 2025. — the IRON paper.
* [AMD XDNA™ NPU in Ryzen™ AI Processors](https://ieeexplore.ieee.org/document/10592049) — IEEE Hot Chips paper.
* AIE architecture manuals: [AIE1 — AM009](https://docs.amd.com/r/en-US/am009-versal-ai-engine/Overview), [AIE2 — AM020](https://docs.amd.com/r/en-US/am020-versal-aie-ml/Overview).
* AIE register references: [AIE1 — AM015](https://docs.amd.com/r/en-US/am015-versal-aie-register-reference/Overview), [AIE2 — AM025](https://docs.amd.com/r/en-US/am025-versal-aie-ml-register-reference/Overview).
* [AIE API User Guide](https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/index.html) — the C++ header-only vector library for kernel code.
* [Summary documentation links (UG1076)](https://docs.amd.com/r/en-US/ug1076-ai-engine-environment/Documentation).

-----
[Section 0 — Getting set up for IRON →](./section-0/)
