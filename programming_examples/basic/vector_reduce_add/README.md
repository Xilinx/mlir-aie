<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# Vector Reduce Add:

A single AIE compute tile performs a simple reduction: it sums an `N`-element `int32` input vector into a `1`-element `int32` output.  Input data is brought to the local memory of the Compute tile via a Shim tile (default `N = 1024`), the reduction is performed in one kernel invocation, and the single output value is copied back to the Shim tile.

The design body is a single `aie.iron.algorithms.reduce_typed(reduce_add_vector, in_ty, out_ty)` call; the algorithms library handles the ObjectFifo / Worker / Runtime plumbing for the reduce shape (whole-input single-kernel-call).

## Source Files Overview

1. `vector_reduce_add.py`: An `@iron.jit`-decorated design that delegates its dataflow body to `aie.iron.algorithms.reduce_typed`.  Supports standalone (`python3 vector_reduce_add.py`) and compile-only (`--xclbin-path` / `--insts-path`, used by the `Makefile`) modes.

1. `reduce_add.cc`: A C++ implementation of a vectorized `add` reduction for AIE cores. The kernel uses the AIE API, documented [here](https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/index.html).  Source: [here](../../../aie_kernels/aie2/reduce_add.cc).

1. `test.cpp`: C++ testbench. Loads the compiled XCLBIN, supplies input, runs on the NPU, and verifies the result.

## Ryzen™ AI Usage

### Standalone

```shell
python3 vector_reduce_add.py
```

`-d npu2` for Strix; `-n` to override the input length.

### Makefile + C++ testbench

```shell
make
make run
```

For NPU2 (Strix): `make devicename=npu2 && make run devicename=npu2`.
