<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Matrix Multiplication - Cascade Design

This matrix multiplication design uses the `4`&times;`4` NPU array with broadcast in each row and cascade in each column.

As an example, the dimensions are set by default to `M`&times;`K`&times;`N` = `512`&times;`512`&times;`512`, and each core operates on the chunk of `64`&times;`64`&times;`64` (`m`&times;`k`&times;`n`). 

Different from the `whole_array` implementation, in this design, the accumulation on `K` is distributed to the four cores belonging to the same column. 

The current design only works for scalar `int16`.

The performance sweep results against `whole_array` can be found at [here](https://gist.github.com/Yu-Zhewen/da3fed9feb278b973f35fb78c2d3a484), no gain observed. 