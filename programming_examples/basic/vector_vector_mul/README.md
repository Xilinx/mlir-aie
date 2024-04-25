<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Vector Vector Multiplication</ins>

A single tile performs a very simple `*` operation from two vectors loaded into memory. The tile then stores the element-wise multiplication of those two vectors back to external memory. This reference design can be run on either a Ryzen™ AI NPU or a VCK5000.

The kernel executes on the AIE tile (`col`, 2). Both input vectors are brought into the tile from the Shim tile (`col`, 0). The value of `col` depends on whether the application targets NPU or VCK5000. The AIE tile performs the multiplication operations, and the Shim tile brings the data back out to external memory.

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
