<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Vector Scalar Multiplication:

This IRON design flow example, called "Vector Scalar Multiplication", demonstrates a simple AIE implementation for vectorized vector scalar multiply on a vector of integers. In this design, a single AIE core performs the vector scalar multiply operation on a vector with a default length `4096`. The kernel is configured to work on `1024` element-sized subvectors, and is invoked multiple times to complete the full scaling. The example consists of two primary design files: `aie2.py` and `scale.cc`, and a testbench `test.cpp` or `test.py`.

## Source Files Overview

1. `aie2.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using `aiecc.py` to produce design binaries (i.e., XCLBIN and inst.txt for the NPU in Ryzen™ AI). 

1. `scale.cc`: A C++ implementation of scalar and vectorized vector scalar multiply operations for AIE cores. Found [here](../../../aie_kernels/aie2/scale.cc).

1. `test.cpp`: This C++ code is a testbench for the Vector Scalar Multiplication design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the script verifies the memcpy results and optionally outputs trace data.

1. `test.py`: This Python code is a testbench for the Vector Scalar Multiplication design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the script verifies the memcpy results and optionally outputs trace data.

## Design Overview

<img align="right" width="300" height="300" src="../../../programming_guide/assets/vector_scalar.svg"> 

This simple example uses a single compute tile in the NPU's AIE array. The design is described as shown in the figure to the right. The overall design flow is as follows:
1. An object FIFO called "of_in" connects a Shim Tile to a Compute Tile, and another called "of_out" connects the Compute Tile back to the Shim Tile. 
1. The runtime data movement is expressed to read `4096` int32_t data from host memory to the compute tile and write the `4096` data back to host memory. A single int32_t scale factor is also transferred form host memory to the Compute Tile.  
1. The compute tile acquires this input data in "object" sized (`1024`) blocks from "of_in" and stores the result to another output "object" it has acquired from "of_out". Note that a scalar or vectorized kernel running on the Compute Tile's AIE core multiplies the data from the input "object" by a scale factor before storing it to the output "object".
1. After the compute is performed, the Compute Tile releases the "objects", allowing the DMAs (abstracted by the object FIFO) to transfer the data back to host memory and copy additional blocks into the Compute Tile,  "of_out" and "of_in" respectively.

It is important to note that the Shim Tile and Compute Tile DMAs move data concurrently, and the Compute Tile's AIE Core also processes data concurrently with the data movement. This is made possible by expressing depth `2` in declaring, for example, `object_fifo("in", ShimTile, ComputeTile2, 2, tile_ty)` to denote ping-pong buffers.

## Design Component Details

### AIE Array Structural Design

This design performs a memcpy operation on a vector of input data. The AIE design is described in a Python module as follows:

1. **Constants & Configuration:** The script defines input/output dimensions (`N`, `n`), buffer sizes in `N_in_bytes` and `N_div_n` blocks, the object FIFO buffer depth, and vector vs scalar kernel selection and tracing support booleans.

1. **AIE Device Definition:** `@device` defines the target device. The `device_body` function contains the AIE array design definition.

1. **Scaling Function Declarations:** `scale_scalar_int32` and `scale_int32` are external functions imported from `scale.cc`.

1. **Tile Definitions:** `ShimTile` handles data movement, and `ComputeTile2` processes the scaling operations.

1. **Object Fifos:** `of_in` and `of_out` are defined to facilitate the vector data communication between `ShimTile` and `ComputeTile2`. Similarly, `of_factor` facilitates the scale factor communication from the `ShimTile` to the `ComputeTile2`.

1. **Tracing Flow Setup (Optional):** A circuit-switched flow is set up for tracing information when enabled.

1. **Core Definition:** The `core_body` function loops through sub-vectors of the input data, acquiring elements from `of_in`, processing using `vector_scalar_mul_aie_scalar()` or `vector_scalar_mul_aie()`, and outputting the result to `of_out`.

1. **Data Movement Configuration:** The `aie.runtime_sequence` operation configures data movement and synchronization on the `ShimTile` for input and output buffer management.

1. **Tracing Configuration (Optional):** Trace control, event groups, and buffer descriptors are set up in the `aie.runtime_sequence` operation when tracing is enabled.

1. **Generate the design:** The `my_vector_scalar()` function triggers the code generation process. The final print statement outputs the MLIR representation of the AIE array configuration.

### AIE Core Kernel Code

`scale.cc` contains a C++ implementation of scalar and vectorized vector scalar multiplication operation designed for AIE cores. It consists of two main sections:

1. **Scalar Scaling:** The `scale_scalar()` function processes one data element at a time, taking advantage of AIE scalar datapath to load, multiply and store data elements.

1. **Vectorized Scaling:** The `scale_vectorized()` function processes multiple data elements simultaneously, taking advantage of AIE vector datapath capabilities to load, multiply and store data elements.

1. **C-style Wrapper Functions:** `vector_scalar_mul_aie_scalar()` and `vector_scalar_mul_aie()` are two C-style wrapper functions to call the templated `scale_vectorized()` and `scale_scalar()` implementations inside the AIE design implemented in `aie2.py`. The functions are provided for `int32_t`.

## Usage

### C++ Testbench

To compile the design:

```
make
```

To complete compiling the C++ testbench and run the design:

```
make run
```

### Python Testbench

To compile the design:

```
make
```

To run the design:

```
make run_py
```
