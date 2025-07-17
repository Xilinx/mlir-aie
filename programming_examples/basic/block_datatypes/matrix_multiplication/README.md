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

Note that these examples are meant to be instructive and do not aim at being an ideal implementation or maximal performance. They may be used to evaluate the cost and capabilities of different operations and datatypes in the NPU, but should still be worked on to achieve maximal performance.

This folder contains examples of the single core and whole array matrix multiplications in addition to one example of how to use the scalar unit to do the shuffling operation described above in the cores. The examples with no marks like [single_core](./single_core/) use bfp16 as input and output, assume that the B matrix is already transposed, and that all three matrices are shuffled appropriately in host memory ([bfp_test.cpp](./bfp_test.cpp)). The [mixed](./single_core_mixed/) examples use bf16 for the A and C matrices (one input and the output), and assume that the B matrix is in bfp16 format, already transposed and shuffled appropriately ([mixed_test.cpp](./mixed_test.cpp)). Finally, the [shuffle](./whole_array_shuffle/) uses the scalar unit to shuffle the A matrix, B is assumed to be pre-shuffled, and C is outputed without the shuffling operation (note that this implementation is tremendously inefficient!).

## Performance

TODO: Add plots of performance of each example
