<!---//===- README.md --------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Getting Started Programming Examples</ins>

These programming examples provide a good starting point for those new to NPU programming with IRON, and aim to provide an overview of the IRON and NPU capabilities. All the designs are self-contained and operate on fixed problem sizes for simplicity. Please see the [programming guide](../../programming_guide/) for a more detailed guide on developing designs.

* [Memcpy](./00_memcpy/) - This design demonstrates a highly parallel, parameterized implementation of a memcpy operation that uses shim DMAs in every NPU column with the goal to measure memory bandwidth across the full NPU and evaluate how well a design utilizes available memory bandwidth across multiple columns and channels.
* [SAXPY](./01_SAXPY/) - This design demonstrates an implementation of a SAXPY operation (i.e. $Z = a*X + Y$) with both scalar and vectorized kernels.
* [Vector Reduce Max](./02_vector_reduce_max/) - This design demonstrates a vector reduce max implementation using a distributed, parallel approach across multiple AIE cores in one NPU column.
* [Matrix Multiplication Single Core](./03_matrix_multiplication_single_core/) - This design demonstrates a single core implementation of a matrix multiplication where input matrices are tiled at the different levels of the NPU memory hierarchy.
