<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# <ins>IRON AIE Application Programming Guide</ins>

This is the programming guide for **IRON** — the Python API for programming AMD Ryzen™ AI NPUs (and Versal™ AI Engines). It teaches how to design, run, and optimize code on the AIE-array.

> **First time here?** If you haven't installed the toolchain yet, start at the [repo root README](../README.md) (driver, XRT, IRON install). Then come back here.
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

    rt = Runtime()
    with rt.sequence(*[np.ndarray[(1024,), np.dtype[np.int32]]] * 2) as (a, b):
        rt.start(w)
        rt.fill(of_in.prod(),  a)
        rt.drain(of_out.cons(), b, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()

a = iron.arange(1024, dtype=np.int32, device="npu")
b = iron.zeros(1024,  dtype=np.int32, device="npu")
my_design(a, b)              # compile + run + sync back
```

`my_design.as_mlir(a, b)` or `python3 my_design.py --emit-mlir` prints the lowered MLIR without touching the NPU.

## Where to go next

| You want… | Read |
|---|---|
| Quickest possible on-ramp (5 small kernels) | [Mini Tutorial](./mini_tutorial/) |
| Install + driver setup | [Repo root README](../README.md) |
| The full guide, top to bottom | [Section 0](./section-0/) onward |
| The shortest end-to-end working program | [Section 3 — My First Program](./section-3/) |
| Optimizing — measure, then tune | [Section 4](./section-4/) (timers, trace, vectorization) |
| Catalog of worked example designs | [Section 5](./section-5/) (basic) and [Section 6](./section-6/) (vision + ML) |
| The implicit-MLIR-context error you just hit | [`implicit_mlir_context.md`](./implicit_mlir_context.md) |
| Knob-by-knob configuration (cache dir, tensor backend, log level) | [`iron_configuration.md`](./iron_configuration.md) |
| What happens between `@iron.jit` and the NPU running | [`compilation_stages.md`](./compilation_stages.md) |

## Sections

* [Section 0 — Getting set up for IRON](./section-0/)
* [Section 1 — Basic AI Engine building blocks](./section-1/) (Worker, Buffer, Runtime, Program, `@iron.jit`)
* [Section 2 — Data movement (Object FIFOs)](./section-2/) (deep dive; 2a–2g)
* [Section 3 — My First Program](./section-3/) (end-to-end vector × scalar, JIT + decomposed XRT)
* [Section 4 — Performance measurement & vector programming](./section-4/) (timers → trace → kernel vectorization)
* [Section 5 — Example vector designs](./section-5/) (catalog)
* [Section 6 — Larger example designs](./section-6/) (vision, ML)

## JIT compile + cache

`@iron.jit` caches compiled artifacts by `(MLIR bytecode + compile-time kwargs)`. The first call to a design compiles; subsequent calls with the same kwargs reuse the cache.

* Cache directory: `${NPU_CACHE_DIR:-~/.npu/cache}`. Set `NPU_CACHE_DIR=/tmp/iron_cache` (or anywhere) to override.
* To force a clean build, `rm -rf "$NPU_CACHE_DIR"` (or the per-design `build/` directory the Makefile writes to).
* First-call compile time depends on design complexity. A simple vector_scalar_mul-style design compiles in single-digit seconds; multi-core matmul or convolution can take 30s+. Subsequent calls with a warm cache are essentially instant.

More configuration knobs (tensor backend, XRT context cache, log level) are in [`iron_configuration.md`](./iron_configuration.md).

## Terminology — the words this guide uses

The guide tries to stick to one term per concept:

| Term | Means | NOT |
|---|---|---|
| **Compute tile** | The physical AIE tile that runs your kernel code (e.g., `Tile(0, 2)`). | "Core" — avoid; same hardware concept but inconsistent across docs. |
| **Worker** | The IRON object describing the code that runs on one compute tile. | "Process", "task" — both have been used historically; prefer Worker. |
| **Mem tile** | The L2 scratchpad tile (typically row 1) for staging data between shim and compute. | "Memory tile" — same thing; prefer "mem tile" for consistency with the IRON API. |
| **Shim tile** | The row-0 tile that bridges the AIE-array to main memory via shim DMA. | |
| **ObjectFifo** | The synchronized streaming-data primitive between two endpoints. Acquire / release. | "Channel" — generic; reserve for AXI stream channels. |
| **Runtime sequence** | The host-side description of `fill` / `drain` operations and worker `start`s, declared with `rt.sequence(...)`. | "Host code" — that's the C++ / Python testbench that calls the design. |
| **`@iron.jit`** | The single recommended entry point. Decorates a Python function that returns a `Program`; the first call JIT-compiles and runs on the NPU. | The dialect-direct form (`from aie.dialects.aie import *`) is what the JIT compiles *to*; you rarely write it. |

-----

## Further reading

* [Quick reference](./quick_reference.md) — IRON API cheat sheet.
* AIE architecture manuals: [AIE1 (AM009)](https://docs.amd.com/r/en-US/am009-versal-ai-engine/Overview), [AIE2 (AM020)](https://docs.amd.com/r/en-US/am020-versal-aie-ml/Overview).
* [AMD XDNA™ NPU in Ryzen™ AI Processors](https://ieeexplore.ieee.org/document/10592049) — IEEE Hot Chips paper.

-----
[Section 0 — Getting set up for IRON →](./section-0/)
