<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Matrix Multiplication

Subdirectories in this directory contain example designs that implement matrix multiplication on the AI-Engine-enabled AMD Neural Processing Unit (NPU).

> These designs all follow largely the same structure and rely on the same basic concepts. The [whole-array design](whole_array/README.md) contains a representative in-depth explanation of this structure and these concepts. In the explanations for the other designs, we rely on the whole-array design as a base and only highlight the differences.

* [`single_core`](single_core) - This design performs matrix-matrix multiplication on a single AI Engine core. 
* [`whole_array`](whole_array) - This design evolves `single_core`, by splitting the computation and parallelizing it. It utilizes all available AI Engine cores simultaneously.
* [`matrix_vector`](matrix_vector) - This design is a specialization to the matrix-vector-multiplication case, which poses unique challenges due to lower computation density. *Work in progress.*

## Note on Numerical Tolerances

This directory contains verification code that ensures the designs in the subdirectories produce the correct output.

The designs can be configured to work on different input and output data types, based on the Makefile variables `dtype_in` and `dtype_out`.
In the default configuration, all designs consume integer intputs and produce integer outputs.
For this case, the verification checks for strict equivalence between the reference output computed on the host CPU and the output calculated on the AI Engine.
That is, verification will only pass for integer data types if the output is equivalent bit-by-bit.

For floating point data types, the verification code allows the AI Engine output to deviate from the reference calculated on the host CPU by some limited maximal relative and absolute tolerance (defined in `common.h`).
This standard practice is necessary for the following reasons:

 - Operations on IEEE 754 floating point values are not commutative. That is, the order of operations can affect the results. All designs in the subdirectories perform tiling of the input matrices, multiplying and accumulating sub-matrices in chunks. The reference calculation code on the CPU, on the other hand, does not perform tiling. As such, some differences due to non-commutativity are expected.
 - The reference on the host CPU is always computed in `float32`, even if the input data type is `bfloat16`, since the host CPU does not support native `bfloat16` multiplication. This means results are calculated with higher precision on the CPU and subsequently truncated, whereas the AI Engine is able to calculate results in a more performant manner thanks to natively using the lower precision data type.
 - If the output datatype is lower-precision than the accumulation data type, the tiling in the `K` dimension affects the results. For example, when multiplying `bfloat16` numbers, the AI Engine accumulates results in higher-precision `float32`. Our designs perform such accumulation for `k` (tiling size in `K` dimension) times before writing the results back into the output buffer. If the output buffer is lower-precision, results are truncated at that time. A larger `k` dimension means fewer such truncations take place. The AI Engine also provides a higher-precision "cascade" data path, which can be used to accumulate results between cores, although none of the designs in this directory make use of this currently.

In summary, different choices of data types, tiling strategies, and usage of AI Engine components, can all affect floating point results in slight ways. Deciding on different choices for these factors presents interesting trade-offs that must be considered on a case-by-case basis for the application at hand.
