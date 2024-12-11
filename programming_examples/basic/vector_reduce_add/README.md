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

Input data is brought to the local memory of the Compute tile via a Shim tile. The size of the input data `N` from the Shim tile is `1024xi32`. The data is copied to the AIE tile, where the reduction is performed. The single output data value is copied from the AIE tile to the Shim tile.

This example does not contain a C++ kernel file. The kernel is expressed in Python bindings that is then compiled with the AIE compiler to generate the AIE core binary. This also enables design portability across AIE generations (NOTE: the kernel runs on the scalar processor not the vector processor, and therefore is not optimized).

## Source Files Overview

1. `vector_reduce_add.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using `aiecc.py` to produce design binaries (ie. XCLBIN and inst.txt for the NPU in Ryzen™ AI). 

1. `vector_reduce_add_alt.py`: An alternate version of the design in `vector_reduce_add.py`, that is expressed in a lower-level version of IRON.

1. `test.cpp`: This C++ code is a testbench for the design example targetting Ryzen™ AI (AIE2). The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the program verifies the results.

## Ryzen™ AI Usage

### Compilation

To compile the design:
```shell
make
```

To compile the alternative design:
```shell
make env use_alt=1
```

To compile the C++ testbench:
```shell
make vector_reduce_add.exe
```

### C++ Testbench

To run the design:

```shell
make run
```

