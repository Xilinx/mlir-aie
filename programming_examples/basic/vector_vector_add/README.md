<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# <ins>Vector Vector Add</ins>

A simple binary operator: a single AIE core adds two vectors element-wise.  The input vectors are processed by the core in sub-tiles of size `16`; the vector length is configurable via the command line and must be a multiple of `16`.  This example shows how compact a binary element-wise design can be when expressed through the `aie.iron.algorithms` library — a single `transform_binary(lambda a, b: a + b, ...)` call handles the ObjectFifo / Worker / Runtime plumbing.

Both input vectors are brought into a Compute tile from a Shim tile, the AIE tile performs the summation, and the Shim tile drains the result back to external memory.

## Source Files Overview

`vector_vector_add.py`: An `@iron.jit`-decorated design that delegates its dataflow body to `aie.iron.algorithms.transform_binary`.  Standalone-runnable: JIT-compiles, executes, and verifies in one shot.

## Ryzen™ AI Usage

```shell
python3 vector_vector_add.py
```

The script uses `iron.tensor(..., device="npu")` for buffer placement; the underlying NPU (Phoenix vs Strix) is selected by the active runtime, not a CLI flag.
