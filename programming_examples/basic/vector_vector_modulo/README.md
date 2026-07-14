<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# <ins>Vector Vector Modulo</ins>

A simple binary operator: a single AIE core computes `c = a % b` element-wise on two vectors of length `256`, processed in sub-tiles of `16`. The design body is a single `transform_binary(lambda a, b: a % b, ...)` call through `aie.iron.algorithms`; the kernel runs on AIE tile (`col`, 2) with both inputs streamed in from a Shim tile and the result streamed back out. This reference design runs on a Ryzen™ AI NPU.

## Source Files Overview

1. `vector_vector_modulo.py`: An `@iron.jit`-decorated design that delegates its dataflow body to `aie.iron.algorithms.transform_binary`. Supports standalone (jit + run + verify) and compile-only (`--xclbin-path` / `--insts-path`, used by the NPU `Makefile`) invocation.

1. `test.cpp`: C++ testbench targeting Ryzen™ AI. Loads the compiled XCLBIN, configures the AIE module, supplies input data, executes on the NPU, and verifies the results.

## Ryzen™ AI Usage

### Standalone

```shell
python3 vector_vector_modulo.py
```

### Makefile + C++ testbench

```shell
make
make run
```

For NPU2 (Strix): `make devicename=npu2 && make run devicename=npu2`.
