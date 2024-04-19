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

Single tile performs a very simple `+` operation where the kernel loads data from local memory, increments the value by `1` and stores it back.

The kernel executes on AIE tile (0, 2). Input data is brought to the local memory of the tile from Shim tile (0, 0), through Mem tile (0, 1). The size of the input data from the Shim tile is `16xi32`. The data is stored in the Mem tile and sent to the AIE tile in smaller pieces of size `8xi32`. Output data from the AIE tile to the Shim tile follows the same process, in reverse.

This example does not contain a C++ kernel file. The kernel is expressed in Python bindings for the `memref` and `arith` dialects that is then compiled with the AIE compiler to generate the AIE core binary.

## Source Files Overview

1. `aie2.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using `aiecc.py` to produce design binaries (ie. XCLBIN and inst.txt for the NPU in Ryzen™ AI). 

1. `test.cpp`: This C++ code is a testbench for the design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the program verifies the results.

## Usage

### C++ Testbench

To compile the design and C++ testbench:

```
make
make build/vectorScalarAdd.exe
```

To run the design:

```
make run
```
