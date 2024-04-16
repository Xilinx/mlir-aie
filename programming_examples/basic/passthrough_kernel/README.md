<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Passthrough Kernel:

This IRON design flow example, called "Passthrough Kernel", demonstrates the process of creating a simple AIE implementation for vectorized memcpy on a vector of integers. In this design, a single AIE core performs the memcpy operation on a vector with a default length `4096`. The core kernel does a `1024` vector load, copy and store that is invoked multiple times to complete the full copy. The example consists of two primary design files: `aie2.py` and `passThrough.cc`, and a testbench `test.cpp` or `test.py`.

## Overview

1. `aie2.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using `aiecc.py` to produce design binaries (ie. XCLBIN and inst.txt for the NPU in Ryzen AI). 

1. `passThrough.cc`: A C++ implementation of scalar and vectorized scaling operations for AIE cores.

1. `test.cpp`: This C++ code is a testbench for the Passthrough Kernel design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the script verifies the memcpy results and optionally outputs trace data.

1. `test.py`: This Python code is a testbench for the Passthrough Kernel design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the script verifies the memcpy results and optionally outputs trace data.

## Design Component Details

### AIE Array Structural Design

This design performs scaling operations on a vector of input data. The AIE design is described in a python module as follows:

1. **Constants & Configuration:** The script defines input/output dimensions (`N`, `n`), buffer sizes in `lineWidthInBytes` and `lineWidthInInt32s`, and tracing support.

1. **AIE Device Definition:** `@device` defines the target device. The `device_body` function contains the AIE array design definition.

1. **Scaling Function Declarations:** `passThroughLine` is an external function imported from `passThrough.cc`.

1. **Tile Definitions:** `ShimTile` handles data movement, and `ComputeTile2` processes the scaling operations.

1. **Object Fifos:** `of_in` and `of_out` are defined to facilitate communication between `ShimTile` and `ComputeTile2`.

1. **Tracing Flow Setup (Optional):** A circuit-switched flow is set up for tracing information when enabled.

1. **Core Definition:** The `core_body` function loops through sub-vectors of the input data, acquiring elements from `of_in`, processing using `passThroughLine`, and outputting the result to `of_out`.

1. **Data Movement Configuration:** The `sequence` function configures data movement and synchronization on the `ShimTile` for input and output buffer management.

1. **Tracing Configuration (Optional):** Trace control, event groups, and buffer descriptors are set up in the `sequence` function when tracing is enabled.

1. **Generate the design:** The `passthroughKernel()` function triggers the code generation process. The final print statement outputs the MLIR representation of the AIE array configuration.

### AIE Core Kernel Code

`passThrough.cc` contains a C++ implementation of scalar and vectorized vector scaling operations designed for AIE cores. It consists of three main sections:

1. **Scalar Scaling:** The `scale()` function performs a scalar scaling operation on input data element by element.

1. **Vectorized Scaling:** The `scale_vectorized()` function processes multiple data elements simultaneously, taking advantage of AIE vector datapath capabilities.

1. **C-style Wrapper Functions:** `passThroughLine()` and `passThroughTile()` are two C-style wrapper functions to call the templated `passThrough_aie()` vectorized memcpy implementation from the AIE design implemented in `aie2.py`. The `passThroughLine()` and `passThroughTile()` functions are compiled for `uint8_t`, `int16_t`, or `int32_t` determined by the value the `BIT_WIDTH` variable defines. 

## Usage

To compile the design and C++ testbench:

```
make
make build/passThroughKernel.exe
```

To run the design:

```
make run
```