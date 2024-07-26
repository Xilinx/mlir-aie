<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Row-wise Bias Addition

This design takes two inputs, `in` and `bias`. 
`in` is a `M`&times;`N` matrix, and `bias` is a `1`&times;`N` row-vector.
The design performs a row-wise addition of `bias` to `in`. 
Conceptually, `bias` is broadcast into a `M`&times;`N` matrix by repeating it `M` times across rows, and then this matrix is added element-wise to `in`.

## Data Movement

The data movement and call into the kernel (see below) is described in `aie2.py`.
A single AIE core is configured to process chunks of `m`&times;`n` of `in` and chunks of `n` of `bias` to produce `m`&times;`n` chunks of output.
Therefore, the output is tiled into `M/m`&times;`N/n` tiles, and the kernel function is called that number of times.
To avoid unnecessarily reloading the `bias` vector, we iterate through these tiles in a column-major fashion.
The `strides` and `sizes` in the `aie.runtime_sequence` operation describe this column-major iteration.

## Kernel

The vectorized kernel is implemented in `kernel.cc`.
The kernel uses vector intrinsics of size `t` to perform the additions.
The computation is designed such that the `bias` vector is not unnecessarily reloaded.
To achieve this, we first load a chunk of `t` elements of `bias`, then produce the results for the first `t` columns of `out` (this is the inner loop).
The outer loop iterates through chunks of `t` columns, loading the next `t` biases at the beginning of each iteration.
