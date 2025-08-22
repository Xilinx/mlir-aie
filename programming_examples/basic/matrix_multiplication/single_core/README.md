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

## Notes on the `single_core_placed.py` Implementation

As in the whole-array design, the [`single_core.py`](./single_core.py) file describes the data movement of the design. This single core example also comes with an alternative implementation, which can be found in [`single_core_placed.py`](./single_core_placed.py). If you specify `use_placed=1` as an environment variable at compile time, this placed implementation will be used in place of `single_core.py`.

Functionally, `single_core.py` and `single_core_placed.py` are intended to be identical. However, `single_core_placed.py` is implemented using a new syntax for runtime buffer descriptor configuration on the shim. Specifically, `single_core_placed.py` uses the `aiex.dma_configure_task_for`, `aiex.dma_start_task` and `aiex.dma_await_task` operations instead of `aiex.dma_memcpy_nd`.

## Notes on the `single_core_iron.py` Implementation

There is an implementation of this design found in [`single_core_iron.py`](./single_core_iron.py) using a higher-level version of IRON. If you specify `use_iron=1` as an environment variable at compile time, this placed implementation will be used in place of `single_core.py`.

Functionally, this design is intended to be identical to the other two. However, `single_core_iron.py` currently does not support tracing.

## Building and Running the Design

You need C++23 for bfloat16_t support. It can be found in g++-13: https://lindevs.com/install-g-on-ubuntu

To compile and run design:
```shell
make
make single_core.exe
make run
```
To compile and run the placed design:
```shell
env use_placed=1 make
env use_placed=1 make single_core.exe
env use_placed=1 make run
```

To compile and run the higher-level IRON design:
```shell
env use_iron=1 make
env use_iron=1 make single_core.exe
env use_iron=1 make run
```


## Tracing

To get tracing output, set `enable_tracing=True` in `single_core.py` and `ENABLE_TRACING=true` in `test.cpp`. Tracing is also supported in `single_core_placed.py`.

By default, traces will be written out to `trace.txt`; another output file can be specified using the `--trace` (or `-t`) flag to the host code.
