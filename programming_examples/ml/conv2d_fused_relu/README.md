<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Convolution with Fused ReLU</ins>

## Introduction

In [conv2d](../conv2d), we describe how to implement a two-dimensional convolution kernel on AIE. While [relu](../relu) describes the implementation of the Rectified Linear Unit (ReLU) activation function on AIE. This README provides instructions for fusing convolution with the ReLU activation function on a single AI Engine (AIE) core


## Source Files Overview

```
.
+-- aie2.py             # A Python script that defines the AIE array structural design using MLIR-AIE operations.
+-- Makefile            # Contains instructions for building and compiling software projects.
+-- README.md           # This file.
+-- run.lit             # For LLVM Integrated Tester (LIT) of the design.
+-- test.py             # Python code testbench for the design example.
```

## Fusing ReLU
Fusing ReLU into the convolution operation can optimize the performance by reducing unnecessary data movement, leading to lower external memory bandwidth requirements and computational overhead. The ReLU activation function introduces non-linearity by setting negative values to zero and leaving positive values unchanged. For fixed-point arithmetic, we can utilize the Shift-Round-Saturate (SRS) capability of AIE to apply an appropriate transformation involving shifting out lower-order bits, rounding, and saturation using the SRS family of intrinsics. Using SRS intrinsics, we can efficiently implement ReLU activation while the data is in the accumulation registers. Such an implementation completely eliminates any need for data movement by fusing at the vector register level.

After performing the convolution operation, we use `aie::set_rounding()` and `aie::set_saturation()` to set the rounding and saturation modes for the computed results in the accumulator. Setting round mode `postitive_inf` rounds halfway towards positive infinity while setting saturation to `aie::saturation_mode::saturate` saturation rounds an uint8 range (0, 255). 


```
::aie::set_rounding(
 aie::rounding_mode::positive_inf);     # Needed to rounding properly to uint8
::aie::set_saturation(
 aie::saturation_mode::saturate);       # Needed to saturate properly to uint8
```

The output data is generated in Y{C/8}X{C8} layout. Please refer to our [conv2d](../conv2d) design for details on the data layout. 

### Benefits of Fusing Convolutiona and ReLU :

1. Reduced Memory Bandwidth:
Fusing ReLU into the convolution operation eliminates unnecessary memory accesses and data transfers associated with separate ReLU computations, leading to reduced memory bandwidth requirements.

2. Improved Performance:
Fusing ReLU reduces the number of instructions executed per element, resulting in improved computational efficiency and overall performance of the convolution operation.

3. Enhanced Resource Utilization:
Combining convolution and ReLU operations allows computational resources to be utilized more efficiently, maximizing throughput and achieving better resource utilization.


## Compilation
To compile the design:
```
make
```

To run the design:
```
make run_py
```