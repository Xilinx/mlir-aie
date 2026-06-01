<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# Eltwise (Add | Mul)

This design implements a `bfloat16` element-wise binary op (addition or multiplication) between two vectors, performed in parallel on two cores in a single column.  Element-wise ops usually end up being I/O bound due to the low compute intensity. In a practical ML implementation, this is the kind of kernel best fused onto a more compute-dense kernel (e.g., a convolution or GEMM).

The op is selected at compile time via the `op` knob (`add` or `mul`); the structural design and host harness are shared.


## Source Files Overview

1. `eltwise.py`: A Python script that defines the AIE array structural design using the IRON API. `op` is a `Compile[str]` knob so the body picks `kernels.add` or `kernels.mul` accordingly; everything else (placement, fifos, runtime sequence) is shared.

1. `add.cc` / `mul.cc`: Vectorized AIE kernels for vector add / multiply, pulled from the IRON kernel library. Sources live under [`aie_kernels/aie2/add.cc`](../../../aie_kernels/aie2/add.cc) and [`mul.cc`](../../../aie_kernels/aie2/mul.cc).

1. `test.cpp`: C++ testbench that loads the compiled XCLBIN, runs the kernel, and verifies the output against a CPU reference. Pass `--op add` or `--op mul` to match the compiled design.


## Usage

### C++ Testbench

Build and run the add variant:
```shell
make op=add
make run op=add
```

Build and run the mul variant:
```shell
make op=mul
make run op=mul
```
