<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# Vector Reduce Min:

A single AIE compute tile performs a simple reduction: it finds the minimum of an `N`-element `int32` input vector and writes a `1`-element `int32` result.  Default `N = 1024`; configurable via `-n` on the CLI.

The design body is a single `aie.iron.algorithms.reduce_typed(reduce_min_vector, in_ty, out_ty)` call; the algorithms library handles the ObjectFifo / Worker / Runtime plumbing for the reduce shape (whole-input single-kernel-call).

## Source Files Overview

1. `vector_reduce_min.py`: An `@iron.jit`-decorated design that delegates its dataflow body to `aie.iron.algorithms.reduce_typed`. Two invocation modes:

   * standalone — `python3 vector_reduce_min.py`
   * compile-only — `... --xclbin-path=PATH --insts-path=PATH` (used by the `Makefile`)

1. `reduce_min.cc`: A C++ implementation of a vectorized `min` reduction for AIE cores. The kernel uses the AIE API ([docs](https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/index.html)). Source: [here](../../../aie_kernels/aie2/reduce_min.cc).

1. `test.cpp`: C++ testbench. Loads the compiled XCLBIN, supplies input, runs on the NPU, and verifies the result.

## Ryzen™ AI Usage

### Standalone

```shell
python3 vector_reduce_min.py
```

`-d npu2` for Strix; `-n` to override the input length.

### Makefile + C++ testbench

```shell
make
make run
```

For NPU2 (Strix): `make devicename=npu2 && make run devicename=npu2`.
