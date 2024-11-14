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

This IRON design flow example, called "Passthrough Kernel", demonstrates a simple AIE implementation for a non-vectorized (scalar) memcpy on a vector of integers. In this design, a single AIE core performs the memcpy operation on a vector with a default length `4096`. The kernel, defined in Python code as a function, is configured to work on `1024` element-sized subvectors and is invoked multiple times to complete the full copy. The example consists of two primary design files: `aie2.py` and `passThrough.cc`, and a testbench `test.cpp` or `test.py`.

## Source Files Overview

1. `aie2.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. The file generates MLIR that is then compiled using `aiecc.py` to produce design binaries (ie. XCLBIN and inst.txt for the NPU in Ryzen™ AI). 

1. `test.cpp`: This C++ code is a testbench for the Passthrough Kernel design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the script verifies the memcpy results and optionally outputs trace data.

1. `test.py`: This Python code is a testbench for the Passthrough Kernel design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the script verifies the memcpy results and optionally outputs trace data.

1. `passthrough_pykernel.ipynb`: This notebook contains the design (which is duplicated from `aie2.py`) and test code (which is duplicated from `test.py`) for an alternate way of interacting with the example.

## Design Overview

<img align="right" width="300" height="300" src="../../../programming_guide/assets/passthrough_simple.svg"> 

This simple example effectively passes data through a single compute tile in the NPU's AIE array. The design is described as shown in the figure to the right. The overall design flow is as follows:
1. An object FIFO called "of_in" connects a Shim Tile to a Compute Tile, and another called "of_out" connects the Compute Tile back to the Shim Tile. 
1. The runtime data movement is expressed to read `4096` uint8_t data from host memory to the compute tile and write the `4096` data back to host memory. 
1. The compute tile acquires this input data in "object" sized (`1024`) blocks from "of_in" and copies them to another output "object" it has acquired from "of_out". A scalar kernel defined via a Python fucntion is invoked on the Compute Tile's AIE core to copy the data from the input "object" to the output "object".
1. After the copy is performed, the Compute Tile releases the "objects", allowing the DMAs (abstracted by the object FIFO) to transfer the data back to host memory and copy additional blocks into the Compute Tile,  "of_out" and "of_in" respectively.

It is important to note that the Shim Tile and Compute Tile DMAs move data concurrently, and the Compute Tile's AIE Core also processes data concurrently with the data movement. This is made possible by expressing depth `2` in declaring, for example, `object_fifo("in", ShimTile, ComputeTile2, 2, line_ty)` to denote ping-pong buffers.

## Design Component Details

### AIE Array Structural Design

This design performs a memcpy operation on a vector of input data. The AIE design is described in a Python module as follows:

1. **Constants & Configuration:** The script defines input/output dimension (`N`), buffer sizes in `lineWidthInBytes` and `lineWidthInInt32s`.

1. **AIE Device Definition:** `@device` defines the target device. The `device_body` function contains the AIE array design definition.

1. **Kernel Function Declarations:** `passThroughLine` is a function defined in Python that performs a scalar copy of the data.

1. **Tile Definitions:** `ShimTile` handles data movement, and `ComputeTile2` processes the memcpy operations.

1. **Object Fifos:** `of_in` and `of_out` are defined to facilitate communication between `ShimTile` and `ComputeTile2`.

1. **Tracing Flow Setup (Optional):** A circuit-switched flow is set up for tracing information when enabled.

1. **Core Definition:** The `core_body` function loops through sub-vectors of the input data, acquiring elements from `of_in`, processing using `passThroughLine`, and outputting the result to `of_out`.

1. **Data Movement Configuration:** The `aie.runtime_sequence` operation configures data movement and synchronization on the `ShimTile` for input and output buffer management.

1. **Tracing Configuration (Optional):** Trace control, event groups, and buffer descriptors are set up in the `aie.runtime_sequence` operation when tracing is enabled.

1. **Generate the design:** The `passthroughKernel()` function triggers the code generation process. The final print statement outputs the MLIR representation of the AIE array configuration.

## Usage

### C++ Testbench

To compile the design:

```bash
make
```

To complete compiling the C++ testbench and run the design:

```bash
make run
```

### Python Testbench

To compile the design:

```bash
make
```

To run the design:

```bash
make run_py
```

### Notebook

#### Use the Notebook
* Start a jupyter server at the root directory of your clone of `mlir-aie`.
  Make sure you use a terminal that has run the `utils/setup_env.sh` script
  so that the correct environment variables are percolated to jupyter.
  Below is an example of how to start a jupyter server:
  ```bash
  python3 -m jupyter notebook --no-browser --port=8080
  ```
* In your browser, navigate to the URL (which includes a token) which is found
  in the output of the above command.
* Navigate to `programming_examples/basic/passthrough_pykernel`
* Double click `passthrough_pykernel.ipynb` to start the notebook; choose the ipykernel called `ironenv`.
* You should now be good to go!

#### Run the Notebook as a Script
```bash
make clean_notebook
make run_notebook
```