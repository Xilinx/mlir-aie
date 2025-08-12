<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>BFP conversion and multiplication</ins>

A simple example illustrating how to manipulate blocked datatypes. Blocked datatypes benefit from lowered data movement and computational costs and excel in machine learning applications where the precision loss does not heavily impact accuracy of the models. Conversions between datatypes allow machine learning applications to benefit from blocked datatypes during dot product operations and then rely on higher precision datatypes when necessary.

To illustrate this, this example uses as input from host memory two `8x8` matrices of bf16 (brain floating point, 8 mantissa and 8 exponent bits). These bf16 matrices are moved into a compute core where they will be transformed into two `8x8` bfp16ebs8 matrices (block floating point, 8 mantissas per block, 8 mantissa bits and 8 shared exponent bits). The bfp16ebs8 matrices are then moved into a different compute core that performs a matrix multiplication between them and returns the result in bfp16ebs8 format back to external memory.

Other available datatypes can be consulted [here](https://xilinx.github.io/aie_api/group__group__basic__types.html). 

## Source Files Overview

1. `bfp_conversion.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using `aiecc.py` to produce design binaries (ie. XCLBIN and inst.txt for the NPU in Ryzen™ AI). 

1. `bfp_conversion_placed.py`: An alternative version of the design in `bfp_conversion.py`, that is expressed in a lower-level version of IRON.

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
make bfp_conversion.exe
```

### C++ Testbench

To run the design:

```shell
make run
```

