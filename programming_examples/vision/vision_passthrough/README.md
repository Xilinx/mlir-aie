<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Vision Pass Through</ins>

Single tile applies a pass through kernel on data from local memory. There are three versions of this pipeline that differ in the sizes of input and output data tensors. This pipeline mainly serves to test whether the data movement between Shim tile (0, 0) and AIE tile (0, 2) works correctly.

To compile desing in Windows:
```
make
make build/passThrough.exe
```

To run the design:
```
make run
```

