<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Matrix Vector Multiplication</ins>

One tile in one or more columns performs a `matrix * vector` multiply on bfloat16 data type where `MxK` is `288x288`. The kernel itself computes `32x32 (MxK)` so it is invoked multiple times to complete the full matvec compute.

You need c++23 for bfloat16_t support. It can be found in g++-13: https://lindevs.com/install-g-on-ubuntu

To compile design:
```
make
make matrixVectorMultiplication.exe
```

To run the design:
```
make run
```
