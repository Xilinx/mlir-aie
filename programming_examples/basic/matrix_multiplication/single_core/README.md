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

In this design, a single AI Engine compute core performs a matrix-matrix-multiplication. By default, the matrices are `int16` data type for the input and `int32` data type for the output, and the dimensions are set (by default) to `M`&times;`K`&times;`N` = `256`&times;`256`&times;`256`. The kernel operates on chunks of `64`&times;`32`&times;`64` (`m`&times;`k`&times;`n`), so it is invoked multiple times to complete the full result.

> This design is a simplification of the [whole-array design](../whole_array/README.md). Instead of utilizing all available AI Engine compute cores in parallel, this design performs all computation on a single core. To understand this design better, please refer to the discussion of the whole-array design and the differences outlined below.

## Differences from the [Whole-Array Design](../whole_array/README.md)

* This design supports tracing; See [below](#tracing).
* Only a single core performs computations. As such, we only need a single ObjectFIFO for each of the transfers between the levels (shim &rightarrow; memory, memory &rightarrow; compute, and back). These ObjectFIFOs are named `inA`, `inB`, `outC` and `memA`, `memB` and `memC`, respectively. 

## Notes on the `aie2_alt.py` Implementation

As in the whole-array design, the `aie2.py` file describes the data movement of the design. This single core example also comes with an alternative implementation, which can be found in `aie2_alt.py`. If you specify `use_alt=1` as an environment variable at compile time, this alternative implementation will be used in place of `aie2.py`.

Functionally, `aie2.py` and `aie2_alt.py` are intended to be identical. However, `aie2_alt.py` is implemented using a new syntax for runtime buffer descriptor configuration on the shim. Specifically, `aie2_alt.py` uses the `aiex.dma_configure_task_for`, `aiex.dma_start_task` and `aiex.dma_await_task` operations instead of `aiex.dma_memcpy_nd`.

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
