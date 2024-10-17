<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Dynamic Object FIFO - Two Core Sliding Window

Contains an example of what a ObjectFIFO lowering may look like that does not statically unroll loops, but instead chooses the buffers dynamically by using MLIR IndexSwitchOps and by keeping the ObjectFIFO state in the tile's local memory.

This design implements the communication from external memory to a first compute tile in the AIE array which sends the data to a second tile that computes the final output to send back to external memory. The input data consists of ten rows of 10xi32 tensors. The first compute tile applies a simple passthrough kernel on incoming data before sending it further. For every two rows, the second tile applies an addition following the same sliding window pattern shown in the [sliding_window](../sliding_window/) example.

## Source Files Overview

1. `aie2.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using `aiecc.py` to produce design binaries (i.e., XCLBIN and inst.txt for the NPU in Ryzen™ AI).

2. `kernel.cc`: C++ implementations of passthrough and add operations for AIE cores. The code uses the AIE API, which is a C++ header-only library providing types and operations that get translated into efficient low-level intrinsics, and whose documentation can be found [here](https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/index.html). The source can be found [here](../../../aie_kernels/aie2/add.cc).

3. `test.cpp`: This C++ code is a testbench for the design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the script verifies the memcpy results and optionally outputs trace data.


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
