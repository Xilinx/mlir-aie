<!---//===- README.md --------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Blocked vector passthrough</ins>

A simple AIE implementation for a vectorized memcpy using block floating point. In this design, a single AIE core performs the memcpy operation on a vector with a default length of `32` block floating point units. Note in particular the difference in the way block floating point elements are referenced in the kernel and the data movement description python files (IRON). In the IRON version, block floating points are referenced as a unit that represents a quantization of 8 floating point values. Therefore, the kernel is being called twice with `16` units of bfp each time for a total of `16x8=128`.

## Source Files Overview

1. `vector_passthrough.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using `aiecc.py` to produce design binaries (ie. XCLBIN and inst.txt for the NPU in Ryzen™ AI). 

1. `vector_passthrough_placed.py`: An alternative version of the design in `vector_passthrough.py`, that is expressed in a lower-level version of IRON.

1. `test.cpp`: This C++ code is a testbench for the design example targeting Ryzen™ AI (AIE-ML). The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the program verifies the results.

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
make vector_vector_add.exe
```

### C++ Testbench

To run the design:

```shell
make run
```

