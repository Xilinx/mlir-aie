<!---//===- README.md --------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Block Datatypes - Matrix Multiplication

This folder cotains multiple variations of the standard matrix multiplication example. In order to understand the matrix multiplication example itself, it is recommended to check the [original one](../../matrix_multiplication/whole_array/README.md), along with its corresponding README file, since explanations related to the matrix multiplication will be found there. Here, only differences related to using blocked datatypes will be mentioned and code comments have been removed.

Blocked datatypes require additional attention when declaring the matrix shapes, since they group multiple elements. When going through these examples, notice how matrices must be reshaped to take this into account (usually with divisions such as `[matrix_dimension] // 8`).

These examples are currently only supported when using the chess compiler. Assuming your environment is properly set up, it can be used by calling the with `make run use_chess=1`.

## Block Datatypes - Main changes

### Data Movement

At the IRON level, v8bfp16ebs8 and v8bfp16ebs16 have corresponding byte sizes of 9 and 17 bytes, which make it impossible to tile the subtiles in the correct order to feed them to the cores (the second level of tiling has been removed in these examples) because of the 4 byte granularity that Data Layout Transformations use. For this reason, these subtiles must be pretiled in main memory or apply the corresponding transformations inside the core. Other alternatives may be considered, such as adding padding to the blocks so that they align with the 4 byte granularity of DMAs.

TODO: Add images explaining these issues here

### Core Computations

Once the data has reached a compute tile, block datatypes also have additional complexities. They require additional manipulation in order to be loaded in and stored out of registers. The additional manipulations required to achieve this can be seen in [mm.cc](./mm.cc).

TODO: Add image explaining this issue here too

## Examples

Note that these examples are meant to be orientative and do not aim at being an ideal implementation. They may be used to evaluate the cost and capabilities of different operations and datatypes in the NPU, but should still be worked on to achieve maximal performance.

- [`in_core_shuffle`](./in_core_shuffle/): Minimalist example performing the shuffling described above inside an AIE core using the scalar unit. It may be used to evaluate the efficiency of the scalar unit and the CPU for the shuffling operation.
- `single_core`: Single core implementation of a matrix multiplication
    - [`no_tiling`](./single_core_no_tiling/): This implementation uses hardcoded matrix dimensions and reduces data movement to its minimum by removing tiling completely. Only bfp16ebs8 values are used for both input, output and inside the kernel without any conversion. Use this example as a stepping stone to understand the more complex ones first. Feel free to try to modify the hardcoded values for the dimensions.
    - [`bfp_input_and_output`](./single_core/): This implementation generalizes the matrix multiplication to any shape within the limits of the hardware and the chosen algorithm. Only bfp16ebs8 values are used for both input, output and inside the kernel without any conversion. The matrix B is assumed to be already transposed and all three matrices are assumed to be shuffled in host memory.
    - [`mixed`](./single_core_mixed/): This implementation modifies the previous one by using bf16 for matrices A and C and does the appropriate conversions for them inside the core.
- `whole_array`: Whole array implementation of a matrix multiplication. These examples may be used to evaluate the performance of the cores in conjunction with the data movement inside the NPU.
    - [`bfp_input_and_output`](./whole_array/): See single core explanation above.
    - [`mixed`](./whole_array_mixed/): See single core explanation above.
    - [`shuffle`](./whole_array_shuffle/): This implementation does not assume that the A matrix has been shuffled in host memory and performs the operation inside the cores, using the scalar unit before calling the matrix multiplication on the vectors. B and C are not shuffled. Note that this example is tremendously inefficient (see below)!

## Performance

TODO: Add plots of performance of each example
