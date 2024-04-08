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

Single tile performs a `matrix * matrix` multiply on bfloat16 data type where `MxKxN` is `128x128x128`. The kernel itself computes `64x32x64 (MxKxN)` so it is invoked multiple times to complete the full matmul compute.

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

## Tracing

To get tracing output, set `enable_tracing=True` in `aie2.py` and `ENABLE_TRACING=true` in `test.cpp`.

By default, traces will be written out to `trace.txt`; another output file can be specified using the `--trace` (or `-t`) flag to the host code.
