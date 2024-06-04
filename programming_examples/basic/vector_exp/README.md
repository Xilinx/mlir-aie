<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->


# Vector $e^x$

This example shows how the look up table capability of the AIE can be used to perform approximations to well-known functions like $e^x$. 
This design uses 4 cores, and each core operates on `1024` `bfloat16` numbers.  Each core contains a lookup table approximation of the $e^x$ function, which is then used to perform the operation.  
$e^x$ is typically used in machine learning applications with relatively small numbers, typically around 0..1, and also will return infinity for input values larger than 89, so a small look up table approximation method is often accurate enough compared to a more exact approximation like Taylor series expansion.

## Source Files Overview

1. `aie2.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using `aiecc.py` to produce design binaries (i.e., XCLBIN and inst.txt for the NPU in Ryzen™ AI). 

1. `bf16_exp.cc`: A C++ implementation of vectorized table lookup operations for AIE cores. The lookup operation `getExpBf16` operates on vectors of size `16`, loading the vectorized accumulator registers with the look up table results.  It is then necessary to copy the accumulator register to a regular vector register before storing it back into memory.  The source can be found [here](../../../aie_kernels/aie2/bf16_exp.cc).

1. `test.cpp`: This C++ code is a testbench for the design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the program verifies the results.

The design also uses a single file from the AIE runtime to initialize the look up table contents to approximate the $e^x$ function.


## Usage

### C++ Testbench

To compile the design:

```
make
```

To compile the C++ testbench:

```
make testExp.exe
```

To run the design:

```
make run
```

