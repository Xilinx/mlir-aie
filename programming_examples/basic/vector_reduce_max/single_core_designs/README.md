<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# Vector Reduce Max (single-core):

A single AIE compute tile finds the maximum of an N-element input vector and writes the result back.  Supports both `int32` (default; `N = 2048` for the default 8192-byte input) and `bfloat16` (`N = 4096`) element types -- the design picks `reduce_max_vector` or `reduce_max_vector_bfloat16` from `reduce_max.cc` based on the `dtype` CompileTime parameter.

The design body is a single `aie.iron.algorithms.reduce_typed(reduce_max, in_ty, out_ty, trace_size=trace_size)` call; the algorithms library handles the ObjectFifo / Worker / Runtime plumbing, including the optional trace.

## Source Files Overview

1. `vector_reduce_max.py`: An `@iron.jit`-decorated design that delegates its dataflow body to `aie.iron.algorithms.reduce_typed`. Two invocation modes:

   * standalone — `python3 vector_reduce_max.py`
   * compile-only — `... --xclbin-path=PATH --insts-path=PATH` (used by the `Makefile`)

## Ryzen™ AI Usage

### Standalone

```shell
python3 vector_reduce_max.py
```

`-d npu2` for Strix; `-dt bf16` for bfloat16; `-i1s` to override the input size in bytes.

### Makefile + C++ testbench

```shell
make
make run
```

For NPU2 (Strix): `make devicename=npu2 && make run devicename=npu2`. For bfloat16: add `dtype=bf16`.

### Trace

```shell
make trace
```
