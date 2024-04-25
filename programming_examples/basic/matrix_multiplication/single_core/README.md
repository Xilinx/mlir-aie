<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Matrix Multiplication - Single Core Design

In this design, a single AI Engine compute core performs a matrix-matrix-multiplication. This is a simplification of the 

The matrices are `bfloat16` data type, and the dimensions are set (by default) to `M`&times;`K`&times;`N` = `128`&times;`128`&times;`128`. The kernel operates on chunks of `64`&times;`32`&times;`64` (`m`&times;`k`&times;`n`), so it is invoked multiple times to complete the full result.

> This design is a simplification of the [whole-array design](../whole_array/README.md). Instead of utilizing all available AI Engine compute cores in parallel, this design performs all computation on a single core. To understand this design better, please refer to the discussion of the whole-array design and the below outlined differences.

## Differences from the [Whole-Array Design](../whole_array/README.md)

* This design supports tracing; See [below](#tracing).
* Only a single core performs computations. As such, we only need a single ObjectFIFO for each of the transfers between the levels (shim &rightarrow; memory, memory &rightarrow; compute, and back). These ObjectFIFOs are named `inA`, `inB`, `outC` and `memA`, `memB` and `memC`, respectively. 

## Building and Running the Design

You need C++23 for bfloat16_t support. It can be found in g++-13: https://lindevs.com/install-g-on-ubuntu

To compile design:
```
make
make matrixMultiplication.exe
```

To run the design:
```
make run
```

## Tracing

To get tracing output, set `enable_tracing=True` in `aie2.py` and `ENABLE_TRACING=true` in `test.cpp`.

By default, traces will be written out to `trace.txt`; another output file can be specified using the `--trace` (or `-t`) flag to the host code.
