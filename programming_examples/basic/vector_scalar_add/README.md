<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Vector Scalar Addition:

This design shows an extremely simple single AIE design, which is incrementing every value in an input vector.

It shows a number of features which can then be expanded to more realistic designs.  

Firstly, a simple 1D DMA pattern is set up to access data from the input and output memories. Small `64` element subtiles are accessed from the larger `1024` element input and output vectors.  Thinking about input and output spaces are large grids, with smaller grids of work being dispatched to individual AIE cores is a fundamental, reusable concept.

Secondly, these `64` element subtiles which are now in the mem tile are split into two smaller `32` element subtiles, and sent to the AIE engine to be processed.  This shows how the multi-level memory hierarchy of the NPU can be used.

Thirdly, the design shows how the bodies of work done by each AIE core is a combination of data movement (the object FIFO acquire and releases) together with compute.

Finally, the overall structural design shows how complete designs are a combination of a static design, consisting of cores, connections and some part of the data movement, together with a run time sequence for controlling the design.
A single tile performs a very simple `+` operation where the kernel loads data from local memory, increments the value by `1` and stores it back.

Input data is first brought in to a MemTile using a Shim tile. The size of the input data from the Shim tile is `64xi32`. The data is stored in the MemTile and sent to the AIE tile in smaller pieces of size `32xi32`. Output data from the AIE compute tile to the Shim tile follows the same process, in reverse.


This example does not contain a C++ kernel file. The kernel is expressed in Python bindings that is then compiled with the AIE compiler to generate the AIE core binary.

## Source Files Overview

1. `vector_scalar_add.py`: An Iron (`@iron.jit`) Python design that compiles directly to NPU binaries (XCLBIN + insts.bin) via `--xclbin-path` / `--insts-path`. Running the script standalone (no `--xclbin-path`) JITs the design and verifies it on the NPU end-to-end.

1. `test.cpp`: This C++ code is a testbench for the design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the program verifies the results.

1. `test_runlist.cpp`: An alternate testbench that exercises the **same** xclbin/insts pair from `vector_scalar_add.py` but invokes the kernel twice in a single XRT [runlist](https://xilinx.github.io/XRT/master/html/xrt_native_apis.html), chaining run-0's output (`i + 2`) into run-1's input to produce `i + 3`. NPU2 only — `xrt::runlist` is not implemented on Phoenix (NPU1).

## Usage

### Compilation

To compile the design:
```shell
make
```

To compile the single-run C++ testbench:
```shell
make vector_scalar_add.exe
```

To compile the runlist C++ testbench (NPU2 only):
```shell
make vector_scalar_add_runlist.exe
```

### C++ Testbench

To run the single-run testbench:

```shell
make run
```

To run the runlist testbench (NPU2 only):

```shell
make run_runlist devicename=npu2
```