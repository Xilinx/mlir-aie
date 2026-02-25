<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Vector Vector Modulo</ins>

A simple binary operator, which uses a single AIE core to get the modulo of two vectors.  The overall vector size in this design is `256` and it processed by the core in smaller sub tiles of size `16`.  It shows how simple it can be to just feed data into the AIEs using the Object FIFO abstraction, and drain the results back to external memory.  This reference design can be run on either a Ryzen™ AI NPU or a VCK5000. 

The kernel executes on AIE tile (`col`, 2). Both input vectors are brought into the tile from Shim tile (`col`, 0). The value of `col` is dependent on whether the application is targeting NPU or VCK5000. The AIE tile performs the summation operations and the Shim tile brings the data back out to external memory.

## Source Files Overview

1. `vector_vector_modulo.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using `aiecc.py` to produce design binaries (ie. XCLBIN and inst.bin for the NPU in Ryzen™ AI). 

1. `vector_vector_modulo_placed.py`: An alternative version of the design in `vector_vector_modulo.py`, that is expressed in a lower-level version of IRON.

1. `test.cpp`: This C++ code is a testbench for the design example targeting Ryzen™ AI (AIE-ML). The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the program verifies the results.

1. `test_vck5000.cpp`: This C++ code is a testbench for the design example targeting the VCK5000 PCIe card (AIE). The code is responsible for configuring the AIEs, allocating memory, providing input data, and executing the AIE design on the VCK5000. After executing, the program verifies the results.

## Ryzen™ AI Usage

### C++ Testbench

### Compilation

To compile the design:
```shell
make
```

To compile the placed design:
```shell
env use_placed=1 make
```

To compile the C++ testbench:
```shell
make vector_vector_modulo.exe
```

### C++ Testbench

To run the design:

```shell
make run
```

