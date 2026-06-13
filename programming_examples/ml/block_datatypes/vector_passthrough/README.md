<!---//===- README.md --------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Blocked vector passthrough</ins>

A simple AIE implementation for a vectorized memcpy using block floating point. In this design, a single AIE core performs the memcpy operation on a vector with a default length of `32` block floating point units. Note in particular the difference in the way block floating point elements are referenced in the kernel and the data movement description python files (IRON). In the IRON version, block floating points are referenced as a unit that represents a quantization of 8 floating point values. Therefore, the kernel is being called twice with `16` units of bfp each time for a total of `16x8=128`.

## Source Files Overview

1. `vector_passthrough.py`: An `@iron.jit` IRON design (high-level `ObjectFifo` / `Worker` / `Runtime`). The Makefile invokes it once in compile-only mode (`--xclbin-path` / `--insts-path`) to produce the XCLBIN and instruction binary in a single step.

1. `kernel.cc`: The bfp16 vectorized passthrough core function (Peano-compiled by `ExternalFunction` at design-build time).

1. `test.cpp`: C++ host harness — loads the XCLBIN + `insts.bin`, runs it on the NPU, verifies output.

## Ryzen™ AI Usage

### Compilation

To compile the design and host testbench:
```shell
make devicename=npu2
```

### Run

```shell
make devicename=npu2 run
```

