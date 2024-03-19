<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Vector Multiplication</ins>

Single tile performs a very simple `*` operations from two vectors loaded into memory. The tile then stores the element wise multiplication of those two vectors back to external memory. 

The kernel executes on AIE tile (6, 2). Both input vectors are brought into the tile from Shim tile (6, 0). The AIE tile performs the multiplication operations and the Shim tile brings the data back out to external memory.

To compile the design:
```
make
```

To run the design:
```
make run
```
