<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Eltwise Add

This design implements a `bfloat16` based element wise addition between two vectors, performed in parallel on two cores in a single column. Element-wise addition usually ends up being I/O bound due to the low compute intensity. In a practical ML implementation, it is an example of the type of kernel that is likely best fused onto another more compute-dense kernel (e.g., a convolution or GEMM). Please refer to [bottleneck](../bottleneck/) design on fusing element-wise addition with convolution for the skip addition.


## Source Files Overview

1. `aie2.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using `aiecc.py` to produce design binaries (i.e., XCLBIN and inst.txt for the NPU in Ryzen™ AI). 

1. `add.cc`: A C++ implementation of a vectorized vector addition operation for AIE cores. The code uses the AIE API, which is a C++ header-only library providing types and operations that get translated into efficient low-level intrinsics, and whose documentation can be found [here](https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/index.html).  The source can be found [here](../../../aie_kernels/aie2/add.cc).

1. `test.cpp`: This C++ code is a testbench for the design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the script verifies the memcpy results and optionally outputs trace data.


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

To generate a [trace file](../../../programming_guide/section-4/section-4b/README.md):

```
make trace
```
