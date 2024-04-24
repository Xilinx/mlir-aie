<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Vector Add</ins>

Single tile performs a very simple `+` operations from two vectors loaded into memory. The tile then stores the sum of those two vectors back to external memory. This reference design can be run on either a RyzenAI NPU or a VCK5000. 

The kernel executes on AIE tile (`col`, 2). Both input vectors are brought into the tile from Shim tile (`col`, 0). The value of `col` is dependent on whether the application is targetting NPU or VCK5000. The AIE tile performs the summation operations and the Shim tile brings the data back out to external memory.

To compile and run the design for NPU:
```
make
make run
```

To compile and run the design for VCK5000:
```
make vck5000
./test.elf
```
