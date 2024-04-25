<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Matrix Multiplication Array</ins>

Multiple tiles in a single column perform a `matrix * matrix` multiply on bfloat16 data type where `MxKxN` is `256x256x256`. The kernel computes `64x64x64 (MxKxN)`, which is invoked multiple times to complete the full matmul compute.

You need c++23 for bfloat16_t support. It can be found in g++-13: https://lindevs.com/install-g-on-ubuntu

To compile design:
```
make
make matrixMultiplication.exe
```

To run the design:
```
make run
```
