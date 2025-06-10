<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Combined Transpose

This design takes a single input, `in`, which is a row-major `M`&times;`N` matrix.
The design combines DMA data layout transformations and code on the compute core
(`VSHUFFLE` instructions)to produce a transposed matrix.

## Goals / Requirements

* Supports matrices of different sizes `M`, `N`.
* Input matrix is tiled such that sub-tiles of size `m`, `n` are available on 
  the compute tile.
* The `m`x`n`-sized tiles should be completely transposed after kernel
  execution on the compute core, so that a user could add other operations on
  the compute tile after the transpose ("kernel fusion"). This means no
  transposition on the output data path is allowed.
* For optimum efficiency, the vector size at which the `VSHUFFLE` instructions
  operate should be customizable parameters `r`, `s`, such that we can
  increase the amount of contiguous reads from memory.

## Compile-time Environment Variables

You can set numerous environment varialbes to configure this design to different
matrix and tile sizes. There will be compilation errors if you use unsupported
sizes or combinations of sizes. Here is an example compilation command:

```
make clean && M=64 N=32 m=16 n=16 r=8 s=16 make run
```

 * `M, N`: Overall matrix size
 * `m, n`: Size of the smaller matrix tiles that are transposed individually.
   Must be a size supported by the kernel; see kernel comments
   (power of two, limited sizes, ...).
   If using the handwritten kernel, `m=16`and `n=16`.
   `m` and `n` must evenly divide `M` and `N`, respectively, as we do not have
   any provisions for padding or processing leftover elements.
 * `r, s`: Size of the smallest individual matrix tiles that the compute core
   transposes at a time. This must fit in a AIE vector register, so be less
   than 256 bytes total.
