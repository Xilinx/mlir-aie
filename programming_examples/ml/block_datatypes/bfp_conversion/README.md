<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>BFP conversion and multiplication</ins>

A simple example illustrating how to manipulate blocked datatypes. Blocked datatypes benefit from lowered data movement and computational costs and excel in machine learning applications where the precision loss does not heavily impact accuracy of the models. Conversions between datatypes allow machine learning applications to benefit from blocked datatypes during dot product operations and then rely on higher precision datatypes when necessary.

To illustrate this, this example uses as input from host memory two `8x8` matrices of bf16 (brain floating point, 8 mantissa and 8 exponent bits). These bf16 matrices are moved into a compute core where they will be transformed into two `8x8` bfp16ebs8 matrices (block floating point, 8 mantissas per block, 8 mantissa bits and 8 shared exponent bits). The bfp16ebs8 matrices are then moved into a different compute core that performs a matrix multiplication between them and returns the result in bfp16ebs8 format back to external memory.

Other available datatypes can be consulted [here](https://xilinx.github.io/aie_api/group__group__basic__types.html). 

## Source Files Overview

1. `bfp_conversion.py`: An `@iron.jit` IRON design (high-level `ObjectFifo` / `Worker` / `Runtime`). The Makefile invokes it once in compile-only mode (`--xclbin-path` / `--insts-path`) to produce the XCLBIN and instruction binary in a single step.

1. `kernel.cc`: The bf16→bfp16 conversion + bfp16 matmul core functions (Peano-compiled by `ExternalFunction` at design-build time).

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

