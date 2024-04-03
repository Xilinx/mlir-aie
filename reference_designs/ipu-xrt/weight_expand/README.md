<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->


# int4 -> bfloat16 dequantization

This IRON design flow example is a dequantization kernel which converts from `signed int4` weights, **32** of which share a single `bfloat16` scale factor.

The conversion is to take each `signed int4` weight, and multiply it by the scale factor, giving a vector with 32 elements of bfloat16 numbers.  This vector can then be used by another kernel, either running on the same core, or on a different core to do a high-precision operator such as bfloat16 based GEMM or GEMV.  The main use model is for generative AI where the loading of the parameters during the generation phase is the limiting factor.

Though other configurations are possible, the design example has a memory layout consisting of **1024** `signed int4` weights followed by **32** `bfloat16` scale factors, meaning the tile to be input is **576 bytes**, or **144 int32 words** in size.

![Memory layout](memory.png?raw=true "Memory layout")

The example consists of two primary design files: `aie2.py` and `scale.cc`, and a testbench `test.cpp`.

## Overview

1. `aie2.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using `aiecc.py` to produce design binaries (ie. XCLBIN and inst.txt for the NPU in Ryzen AI). 

1. `expand.cc`: A C++ implementation of vectorized dequantization operations for the AIE core.

1. `test.cpp`: This C++ code is a testbench for the dequantization design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the script verifies the dequantized results.

## Design Component Details

### AIE Array Structural Design

This design performs dequantization operations on a vector of input data. The AIE design is described in a python module as follows:

1. **Constants & Configuration:** The script defines input dimensions (`N`, `n`), as well as the block size (the number of weights which share a scale factor)

1. **AIE Device Definition:** `@device` defines the target device. The `device_body` function contains the AIE array design definition.

1. **Dequantization Function Declarations:** `expand_int4_to_bfloat16` is an external function imported from `expand.cc`.

1. **Tile Definitions:** `ShimTile` handles data movement, and `core0` processes the dequantization operations.

1. **Object Fifos:** `inA` and `outB` are defined to facilitate communication between `ShimTile` and `core0`.

1. **Core Definition:** The `core_body` function loops through sub-vectors of the input data, acquiring elements from `inA`, processing using `expand_int4_to_bfloat16`, and outputting the result to `outB`.

1. **Data Movement Configuration:** The `sequence` function configures data movement and synchronization on the `ShimTile` for input and output buffer management.

1. **Generate the design:** The `my_expand()` function triggers the code generation process. The final print statement outputs the MLIR representation of the AIE array configuration.

### AIE Core Kernel Code

`expand.cc` contains a C++ implementation of scalar and vectorized vector scaling operations designed for AIE cores. It consists of three main sections:

1. **Vectorized dequantization:** The `expand()` function processes multiple data elements simultaneously, taking advantage of AIE vector datapath capabilities.

1. **C-style Wrapper Functions:** `expand_int4_to_bfloat16()` is a C-style wrapper functions to call the `expand()` function from the AIE design implemented in `aie2.py`.

## Usage

To compile the design and testbench:

```
make all
```

To run the design:

```
make run
```