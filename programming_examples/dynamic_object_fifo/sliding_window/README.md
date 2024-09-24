<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Dynamic Object FIFO - Sliding Window

Contains an example of what a ObjectFIFO lowering may look like that does not statically unroll loops, but instead chooses the buffers dynamically by using MLIR IndexSwitchOps and by keeping the ObjectFIFO state in the tile's local memory.

This design implements the communication from external memory to a compute tile in the AIE array, and back. The input data consists of ten rows of 10xi32 tensors. Every iteration the compute tile acquires up to two input rows and adds the values on each column. It then releases only one of the two rows and continues onto the next input row following a sliding window pattern.

The acquire / release patterns for the first and last rows are different than for the rows in the middle of the input. The first row is added to itself to account for the border effect and as such we only acquire one row during the first iteration and release none. For the last iteration, we release two rows to account for the sliding window of 1.

## Source Files Overview

1. `aie2.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using `aiecc.py` to produce design binaries (i.e., XCLBIN and inst.txt for the NPU in Ryzen™ AI).

2. `aie2_if_else.py`: A variant of the Python script that uses if-else operations inside the compute tile's for loop to isolate the top and bottom rows of the input.

3. `kernel.cc`: A C++ implementation of a simple add operation for AIE cores. The code uses the AIE API, which is a C++ header-only library providing types and operations that get translated into efficient low-level intrinsics, and whose documentation can be found [here](https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/index.html). The source can be found [here](../../../aie_kernels/aie2/add.cc).

4. `test.cpp`: This C++ code is a testbench for the design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the script verifies the memcpy results and optionally outputs trace data.


## Usage

### C++ Testbench

To compile the design and C++ testbench:

```
make
```

To run the design:

```
make run
```
