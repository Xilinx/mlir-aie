<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Matrix-Vector Multiplication

In this design, one or multiple AI Engine compute cores (spread across hardware columns, configurable as `n_cores`) perform a matrix-*vector* multiplication. We use a `bfloat16` data type, and the dimensions of the `A` matrix `M`&times;`K` are set to `288`&times;`288` by default (`N`, the number of columns in `B`, is always `1`, since `B` is a vector). The kernel itself consumes chunks of `32`&times;`32` (`M`&times;`K`) of `A`, so it is invoked multiple times to complete the full result.

> This design relies on the same basic concepts as the [whole-array matrix-matrix multiplication design](../whole_array/README.md), and it is structured very similarly to that design. Please refer to the in-depth explanation of that design along with the below outlined differences for a better understanding of this design.

## Differences from the [Whole-Array Matrix-Matrix Multiplication Design](../whole_array/README.md)

- A specialized matrix-*vector* microkernel, named `matvec_vectorized` is used in this design, as opposed to the more general matrix-matrix microkernel (`matmul_vectorized`) used in the matrix-matrix-multiplication designs.
- The data movement in this design varies as follows: An identical `32`-element chunk of the vector `B` is **broadcast** to the cores in all columns, whereas _distinct_ subsequent `32`&times;`32`-sized tiles of the `A` matrix are **distributed** to the cores. As such, each core is responsible for a distinct `32`-element chunk of the output vector `C`. These chunks are assembled (**joined**) at the shim tile level (in the `aiex.runtime_sequence()`).
- This design does not use all available compute cores. Instead, it uses at most one core in each hardware column. The variable `n_cores` defines the number of columns to be used. It would however be possible to extend this design to use all cores.

## Building and Running the Design

You need C++23 for `bfloat16_t` support. It can be found in g++-13: https://lindevs.com/install-g-on-ubuntu

To compile design:
```
make
make matrixVectorMultiplication.exe
```

To run the design:
```
make run
```
