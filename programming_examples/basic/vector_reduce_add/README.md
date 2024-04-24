<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Vector Reduce Add:

Single tile performs a very simple reduction operation where the kernel loads data from local memory, performs the `add` reduction and stores the resulting value back.

The kernel executes on AIE tile (0, 2). Input data is brought to the local memory of the tile from Shim tile (0, 0). The size of the input data `N` from the Shim tile is `1024xi32`. The data is copied to the AIE tile, where the reduction is performed. The single output data value is copied from the AIE tile to the Shim tile.

This example does not contain a C++ kernel file. The kernel is expressed in Python bindings for the `memref` and `arith` dialects that is then compiled with the AIE compiler to generate the AIE core binary. This also enables design portability across AIE generations (NOTE: the kernel runs on the scalar processor not the vector processor, and therefore is not optimized).

## Source Files Overview

1. `aie2.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using `aiecc.py` to produce design binaries (ie. XCLBIN and inst.txt for the NPU in Ryzen™ AI). 

1. `test.cpp`: This C++ code is a testbench for the design example targetting Ryzen™ AI (AIE2). The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the program verifies the results.

1. `test_vck5000.cpp`: This C++ code is a testbench for the design example targetting the VCK5000 PCIe card (AIE1). The code is responsible for configuring the AIEs, allocating memory, providing input data, and executing the AIE design on the VCK5000. After executing, the program verifies the results.

## Ryzen™ AI Usage

### C++ Testbench

To compile the design and C++ testbench:

```
make
make reduce_add.exe
```

To run the design:

```
make run
```

## VCK5000 Usage

### C++ Testbench

To compile the design and C++ testbench:

```
make vck5000
```

To run the design:

```
./test.elf
```