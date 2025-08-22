<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Vector Vector Add with BD-level Syntax and Initial Values</ins>

A simple binary operator, which uses a single AIE core to get the addition of two vectors.  The overall vector size in this design is `256` and is processed by the core in smaller sub tiles of size `16`.  This reference design can be run on either a Ryzen™ AI NPU or a VCK5000. 

The kernel executes on AIE tile (`col`, 2). One input vector is brought into the tile from Shim tile (`col`, 0). The other input vector is initialized on the AIE tile directly with the full vector size. The value of `col` is dependent on whether the application is targeting NPU or VCK5000. The AIE tile performs the summation operations and the Shim tile brings the data back out to external memory.

The data movement in this design is decribed at BD-level in the DMA code regions of the AIE tile.

## Source Files Overview

1. `vector_vector_add.py`: defines the AIE array structural design using IRON AIE language bindings. This generates mlir-aie that is then compiled using `aiecc.py` to produce design binaries (ie. XCLBIN and inst.bin for the NPU in Ryzen™ AI). 

1. `test.cpp`: This C++ code is a testbench for the design example targeting Ryzen™ AI (AIE-ML). The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the program verifies the results.

1. `test_vck5000.cpp`: This C++ code is a testbench for the design example targeting the VCK5000 PCIe card (AIE). The code is responsible for configuring the AIEs, allocating memory, providing input data, and executing the AIE design on the VCK5000. After executing, the program verifies the results.

## Ryzen™ AI Usage

### C++ Testbench

To compile the design and C++ testbench:

```shell
make
make vector_vector_add.exe
```

To run the design:

```shell
make run
```

## VCK5000 Usage

### C++ Testbench

To compile the design and C++ testbench:

```shell
make vck5000
```

To run the design:

```shell
./test.elf
```

