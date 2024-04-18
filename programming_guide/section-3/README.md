<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 3 - My First Program</ins>

<img align="right" width="500" height="250" src="../assets/binaryArtifacts.svg">

This section creates a first program that will run on the AIE-array. As shown in the figure on the right, we will have to create both binaries for the AIE-array (device) and CPU (host) parts. For the AIE-array, a structural description and kernel code is compiled into the AIE-array binaries. The host code loads the AIE-array binaries and contains the test functionality.

For the AIE-array structural description we will combine what you learned in [section-1](../section-1) for defining a basic structural design in python with the data movement part from [section-2](../section-2).

For the AIE kernel code, we will start with non-vectorized code that will run on the scalar processor part of an AIE. [section-4](../section-4) will introduce how to vectorize a compute kernel to harvest the compute density of the AIE.

The host code can be written in either C++ (as shown in the figure) or in Python. We will also introduce some convenience utility libraries for typical test functionality and to simplify context and buffer creation when the [Xilinx RunTime (XRT)](https://github.com/Xilinx/XRT) is used, for instance in the [AMD XDNA Driver](https://github.com/amd/xdna-driver) for Ryzen™ AI devices.

<img align="right" width="410" height="84" src="../assets/vectorScalarMul.svg">

Throughout this section, a [vector scalar multiplication](../../programming_examples/basic/vector_scalar_mul/) (c = a * factor) will be used as an example. Vector scalar multiplication takes an input vector a and computes the output vector c by multiplying each element of a with a factor. This design is also available in the [programming_examples](../../programming_examples) of this repository. We will first introduce the AIE-array structural description, the review the kernel code and then introduce the host code. Finally we will show ho to run the design on Ryzen™ AI enabled hardware.

## AIE-array Structural Description

<img align="right" width="109" height="264" src="../assets/vectorScalarMulPhysicalDataFlow.svg">

Since the input vector a and output vector c are residing in external memory, the AIE-array structural description (see [section-1](../section-1) int the [aie2.py](../../programming_examples/basic/vector_scalar_mul/aie2.py) will deploy both shimDMA (purple) for data movement and a compute core (green) for the multiplication in the operations. Since the compute core can only access L1 memory, input data needs to be explicitly moved to (yellow arrow) and from (orange arrow) the L1 memory of the AIE.

```
    # Device declaration - here using aie2 device NPU
    @device(AIEDevice.ipu)
    def device_body():

        # Tile declarations
        ShimTile = tile(0, 0)
        ComputeTile2 = tile(0, 2)
```

## Kernel Code

## Host Code

## Running the Program
