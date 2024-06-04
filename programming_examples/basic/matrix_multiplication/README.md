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