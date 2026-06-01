<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# Custom DMA

This IRON design demonstrates how to program arbitrary DMA patterns from the high-level IRON API by writing a `Resolvable` subclass. The example reads two non-contiguous rows from a matrix stored on a MemTile, skipping an intermediate row — a scatter-read pattern that ObjectFifo cannot express because ObjectFifo elements are uniform and always read from sequential positions.

## Source Files Overview

1. `custom_dma.py`: A Python script that defines the IRON design. It contains a `ScatterReadDMA` class (a `Resolvable` subclass) that emits custom locks, DMA buffer descriptors, and flows alongside standard IRON components (ObjectFifo, Worker, Runtime).

1. `test.py`: Host program to run the design on the NPU and verify the output against expected values.

## Design Overview

A 3×16 matrix of `int32` values is pre-loaded on a MemTile via `initial_value`. A two-BD DMA chain reads row 0 and row 2 using per-BD offsets (`offset=0` and `offset=32`), skipping row 1. Each BD transfers 16 elements to a compute tile, which copies them to an output ObjectFifo that drains to the host.

```
MemTile buffer (48 x i32):
  row 0 [0..15]  = [100, 101, ..., 115]  ← BD1 reads here
  row 1 [16..31] = [300, 301, ..., 315]  (skipped)
  row 2 [32..47] = [200, 201, ..., 215]  ← BD2 reads here
```

The `ScatterReadDMA` class integrates with IRON through the `Resolvable` interface:
- `tiles()` declares tile dependencies so `Program.resolve_program()` resolves them before `resolve()` runs.
- `resolve()` emits the MemTile buffer, locks, flow, and DMA schedule as raw MLIR ops.
- `acquire()` / `release()` provide lock-based synchronization that the Worker's core function calls at kernel time.

The MemTile DMA is triggered from the runtime sequence via `set_lock_value`, which ensures the shim drain is configured before data starts flowing.

## Usage

### Compilation

To compile the design:

```shell
make
```

### Python Testbench

To run the design:

```shell
make run_py
```
