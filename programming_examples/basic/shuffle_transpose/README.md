<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Shuffle Transpose

This design takes a single input, `in`,
which is a linerized array corresponding to a `16`&times;`16` matrix.
The design uses AIE core shuffle operations to transpose the 
`16`&times;`16` matrix.


## Data Movement

The data movement and call into the kernel (see below) is described in `shuffle_transpose.py`.
A single AIE core is configured to process chunks of `m`&times;`n` of `in`
(`m` and `n` are configured to be 16).
The input and output are tiled into `M/m`&times;`N/n` tiles, and the kernel function is called that number of times -
the example is configured to process one tile, but can be configured to transpose multiple `16`&times;`16`,
one after the other.


## Kernel

The vectorized kernel is implemented in `kernel.cc`.
