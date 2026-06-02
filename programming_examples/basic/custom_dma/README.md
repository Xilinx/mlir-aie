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

This IRON design demonstrates how to program custom DMA patterns that integrate with the IRON API by using a `Resolvable` subclass. `ScatterReadDMA` is automatically discovered in the Worker's `fn_args` by `Program`, which calls `resolve()` to emit the DMA configuration.

## Source Files Overview

1. `custom_dma.py`: A Python script that defines the IRON design. It contains a `ScatterReadDMA` class (a `Resolvable` subclass) that emits custom locks, DMA buffer descriptors, and flows alongside standard IRON components.

2. `test.py`: Host program to run the design on the NPU and verify the output against expected values.

## Design Overview

A 4×16 matrix of `int32` values is pre-loaded on a MemTile via `initial_value`. A custom three-BD DMA chain reads rows 0, 1, and 3 (skipping row 2) using per-BD offsets with non-uniform spacing. Each BD transfers 16 elements to a compute tile, which copies them to an output ObjectFifo that drains to the host.

```
MemTile buffer (64 x i32):
  row 0 [0..15]  = [100, 101, ..., 115]  ← BD1 reads here
  row 1 [16..31] = [200, 201, ..., 215]  ← BD2 reads here
  row 2 [32..47] = [300, 301, ..., 315]  (skipped)
  row 3 [48..63] = [400, 401, ..., 415]  ← BD3 reads here
```
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
