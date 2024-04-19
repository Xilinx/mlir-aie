<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# ReLU


ReLU, which stands for Rectified Linear Unit, is a type of activation function that is widely used in neural networks, particularly in deep learning models. It is defined mathematically as: $ReLU(x) = max(0,x)$

This function takes a single number as input and outputs the maximum of zero and the input number. Essentially, it passes positive values through unchanged, and clamps all negative values to zero.

## Key Characteristics of ReLU:
* Non-linear: While it looks like a linear function, ReLU introduces non-linearity into the model, which is essential for learning complex patterns in data.
* Computational Efficiency: One of ReLU's biggest advantages is its computational simplicity. Unlike other activation functions like sigmoid or tanh, ReLU does not involve expensive operations (e.g., exponentials), which makes it computationally efficient and speeds up the training and inference processes.

This design implements a `bfloat16` based ReLU on a vector, performed in parallel on two cores in a single column.  This will end up being I/O bound due to the low compute intensity, and in a practical ML implementation, is an example of the type of kernel that is likely best fused onto another more compute dense kernel (e.g. a convolution or GEMM).


## Source Files Overview

1. `aie2.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using `aiecc.py` to produce design binaries (ie. XCLBIN and inst.txt for the NPU in Ryzen AI). 

1. `relu.cc`: A C++ implementation of a vectorized ReLU operation for AIE cores, which is 1:1 implementation of the inherent function using low level intrinsics.  The AIE2 allows an element-wise max of 32 `bfloat16` numbers against a second vector register containing all zeros, implementing the $ReLU(x) = max(0,x)$ function directly.   The source can be found [here](../../../aie_kernels/aie2/relu.cc).

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