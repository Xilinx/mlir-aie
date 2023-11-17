<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Matrix Multiplication</ins>

Single tile performs a `matrix * matrix` multiply on int16 data type where `MxKxN` is `128x128x128`. The kernel itself computes `64x32x64 (MxKxN)` so it is invoked multiple times to complete the full matmul compute.

To compile desing in Windows:
```
make
make matrixMultiplication.exe
```

To run the design:
```
make run
```
