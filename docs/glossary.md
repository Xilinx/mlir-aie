<!-- Copyright (C) 2024-2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Glossary

The vocabulary IRON and this documentation use, grouped by topic. Where a
term maps to a Python class, the name links to its
[API reference](api/iron.md).

---

## Hardware

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
| **Versal™** | AMD's adaptive SoC family that also contains AI Engine tiles; a secondary IRON target. |

## Devices

| Term | Definition |
|------|------------|
| **npu1** | The device for Ryzen™ AI Phoenix (e.g. 7940HS) and HawkPoint (e.g. 8040HS) SoCs. AIE2 / XDNA. |
| **npu2** | The device for Ryzen™ AI Strix, Strix Halo, and Krackan Point SoCs. AIE2P / XDNA 2. |
| **npu1_Ncol / npu2_Ncol** | A device representing a physical column-partition of an NPU (e.g. `npu2_4col`), including a DMA shim tile per column. |

## IRON API

| Term | Definition |
|------|------------|
| [**Worker**](api/iron.md#iron.worker.Worker) | The IRON object describing the code that runs on one compute tile. Prefer "Worker" over "process" or "task". |
| [**ObjectFifo**](api/iron.md#iron.dataflow.objectfifo.ObjectFifo) | The synchronized streaming-data primitive between two endpoints (host↔tile or tile↔tile). Producers and consumers `acquire` / `release` its elements. Reserve "channel" for AXI stream channels. |
| [**ObjectFifoHandle**](api/iron.md#iron.dataflow.objectfifo.ObjectFifoHandle) | A producer or consumer handle to an ObjectFifo, obtained via `of.prod()` / `of.cons()`. Worker core functions call `acquire` / `release` on it. |
| [**Runtime**](api/iron.md#iron.runtime.runtime.Runtime) | The host-side description of `fill` / `drain` operations and Worker `start`s, declared with `rt.sequence(...)`. Distinct from "host code" (the C++/Python testbench that calls the design). |
| [**Program**](api/iron.md#iron.program.Program) | The top-level container that binds a device and a Runtime and resolves the design to MLIR. |
| [**Buffer**](api/iron.md#iron.buffer.Buffer) | A named memory region on a tile, accessible by both Workers and the Runtime (often used for runtime parameters). |
| [**Kernel**](api/iron.md#iron.kernel.Kernel) / [**ExternalFunction**](api/iron.md#iron.kernel.ExternalFunction) | Wrappers for AIE core functions: `Kernel` for a pre-compiled object file, `ExternalFunction` for C/C++ source compiled at JIT time. |
| [**TensorAccessPattern (TAP)**](api/taplib.md) | A description of how a tensor is sliced and streamed to/from the NPU across multiple DMA transfers. Passed as `tap=` to `rt.fill()` / `rt.drain()`. |
| [**Flow**](api/iron.md#iron.dataflow.flow.Flow) / [**PacketFlow**](api/iron.md#iron.dataflow.flow.PacketFlow) | Lower-level explicit-routing primitives: `Flow` for circuit-switched routes, `PacketFlow` for packet-switched routes with caller-controlled packet IDs. |
| [**TileDma**](api/iron.md#iron.dataflow.tile_dma.TileDma) | A lower-level explicit per-tile DMA program, used when the ObjectFifo abstraction hides too much. |
| **Runtime sequence** | The `rt.sequence(...)` context in which `fill` / `drain` / `start` operations are declared. |

## Compilation

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

## Data movement

| Term | Definition |
|------|------------|
| **acquire / release** | The ObjectFifo synchronization pair. A producer or consumer `acquire`s elements to read/write them, then `release`s them so the other endpoint can proceed. |
| **fill / drain** | Runtime operations: `fill` streams a host tensor into a producer ObjectFifo; `drain` reads from a consumer ObjectFifo back into a host tensor. |
| **buffer descriptor (BD)** | A DMA descriptor entry describing one transfer: buffer, offset, length, and access pattern. |
| **access pattern** | A multi-dimensional `(size, stride)` description of the order in which a DMA reads or writes elements — the mechanism behind data-layout transformations. |
| **broadcast / split / join** | ObjectFifo data-movement patterns: broadcast sends one producer's data to many consumers; split/join distribute or gather data across multiple sub-fifos. |
