<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Basic Programming Examples</ins>

These programming examples provide a good starting point to illustrate how to build commonly used compute kernels (both single-core and multi-core data processing pipelines). They serve to highlight how designs can be described in Python and lowered through the mlir-aie tool flow to an executable that runs on the NPU. [Passthrough Kernel](./passthrough_kernel) and [Vector Scalar Mul](./vector_scalar_mul) are good designs to get started with. Please see [section 3](../../programming_guide/section-3/) of the [programming guide](../../programming_guide/) for a more detailed guide on developing designs.

* [Passthrough DMAs](./passthrough_dmas) - This design demonstrates data movement to implement a memcpy operation using object FIFOs just using DMAs without involving the AIE core. 
* [Passthrough Kernel](./passthrough_kernel) - This design demonstrates a simple AIE implementation for vectorized memcpy on a vector of integer involving AIE core kernel programming.
* [Vector Scalar Add](./vector_scalar_add) - Single tile performs a very simple `+` operation where the kernel loads data from local memory, increments the value by `1` and stores it back.
* [Vector Scalar Mul](./vector_scalar_mul) - Single tile performs `vector * scalar` of size `4096`. The kernel does a `1024` vector multiply and is invoked multiple times to complete the full `vector * scalar` compute.
* [Vector Reduce Add](./vector_reduce_add) - Single tile performs a reduction of a vector to return the `sum` of the elements.
* [Vector Reduce Max](./vector_reduce_max) - Single tile performs a reduction of a vector to return the `max` of the elements.
* [Vector Reduce Min](./vector_reduce_min) - Single tile performs a reduction of a vector to return the `min` of the elements.
* [Vector Exp](./vector_exp) - A simple element-wise exponent function, using the look up table capabilities of the AI Engine.
* [Matrix Multiplication](./matrix_multiplication) - This directory contains multiple designs spanning: single core and multi-core (whole array) matrix-matrix multiplication, and matrix-vector multiplication designs. It also contains sweep infrastructure for benchmarking.