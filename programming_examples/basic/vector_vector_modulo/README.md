<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# <ins>Vector Vector Modulo</ins>

A simple binary operator: a single AIE core computes `c = a % b` element-wise on two vectors of length `256`, processed in sub-tiles of `16`. The design body is a single `transform_binary(lambda a, b: a % b, ...)` call through `aie.iron.algorithms`; the kernel runs on AIE tile (`col`, 2) with both inputs streamed in from a Shim tile and the result streamed back out. This reference design can be run on either a Ryzen™ AI NPU or a VCK5000.

## Source Files Overview

1. `vector_vector_modulo.py`: An `@iron.jit`-decorated design that delegates its dataflow body to `aie.iron.algorithms.transform_binary`. Supports three invocation modes — standalone (jit + run + verify), compile-only (`--xclbin-path` / `--insts-path`, used by the NPU `Makefile`), and emit-MLIR (`--emit-mlir`, used by the aiecc-based vck5000 path).

1. `test.cpp`: C++ testbench targeting Ryzen™ AI. Loads the compiled XCLBIN, configures the AIE module, supplies input data, executes on the NPU, and verifies the results.

1. `test_vck5000.cpp`: C++ testbench targeting the VCK5000 PCIe card (AIE1).

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

## VCK5000 Usage

```shell
make vck5000
```

(produces `test.elf`; run on the VCK5000 host.)
