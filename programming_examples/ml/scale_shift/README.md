<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Eltwise Multiplication

This design implements a `bfloat16` based element-wise multiplication followed by an element-wise addition of three vectors, performed in parallel on two cores in a single column.  Element-wise multiplication and addition usually is I/O bound due to the low compute intensity. In a practical ML implementation, this is an example of the type of simple kernel fusion passing intermediate results in DDR.


## Source Files Overview

1. `scale_shift.py`: A Python script that defines the AIE array structural design using IRON operations. This generates MLIR that is then compiled using `aiecc.py` to produce design binaries (ie. XCLBIN and inst.bin for the NPU in Ryzen™ AI). 

1. `scale_shift.cc`: A C++ implementation of a vectorized vector multiplication and addition operations for AIE cores. The code uses the AIE API, which is a C++ header-only library providing types and operations that get translated into efficient low-level intrinsics.  The source can be found [here](../../../aie_kernels/aie2/scale_shift.cc). The parameter `is_mul` is used to switch between the two operations at runtime.

1. `test.cpp`: This C++ code is a testbench for the design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the script verifies the memcpy results and optionally outputs trace data.


## Usage

### C++ Testbench

To compile the design and C++ testbench:
```shell
make
```

To run the design:
```shell
make run
```
