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

Thirdly, the design shows how the bodies of work done by each AIE core is a combination of data movement (the object FIFO acquire and releases) together with compute, which in this case is expressed using a number of different MLIR dialects, like `arith`, `memref`, etc. next to `mlir-aie`.

Finally, the overall structural design shows how complete designs are a combination of a static design, consisting of cores, connections and some part of the data movement, together with a run time sequence for controlling the design.
A single tile performs a very simple `+` operation where the kernel loads data from local memory, increments the value by `1` and stores it back.

The kernel executes on AIE tile (0, 2). Input data is first brought to the Me tile in (0, 1) using the Shim tile (0, 0). The size of the input data from the Shim tile is `64xi32`. The data is stored in the Mem tile and sent to the AIE tile in smaller pieces of size `32xi32`. Output data from the AIE tile to the Shim tile follows the same process, in reverse.

This example does not contain a C++ kernel file. The kernel is expressed in Python bindings for the `memref` and `arith` dialects that is then compiled with the AIE compiler to generate the AIE core binary.

## Source Files Overview

1. `aie2.py`: defines the AIE array structural design using IRON AIE language bindings. This generates mlir-aie that is then compiled using `aiecc.py` to produce design binaries (ie. XCLBIN and inst.txt for the NPU in Ryzen™ AI). 

1. `test.cpp`: This C++ code is a testbench for the design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the program verifies the results.

## Usage

### C++ Testbench

To compile the design and C++ testbench:

```
make
make vectorScalarAdd.exe
```

To run the design:

```
make run
```
