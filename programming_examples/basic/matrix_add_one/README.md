<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Matrix Addition</ins>

Single tile performs a very simple `+` operation where the kernel loads data from local memory, increments the value by `1` and stores it back. The DMA in the Shim tile is programmed to bring the bottom left `8x16` portion of a larger `16x128` matrix into the tile to perform the operation. This reference design can be run on either a RyzenAI NPU or a VCK5000.

The kernel executes on AIE tile (`col`, 2). Input data is brought to the local memory of the tile from Shim tile (`col`, 0). The value of `col` is dependent on whether the application is targetting NPU or VCK5000. The Shim tile is programmed with a 2D DMA to only bring a 2D submatrix into the AIE tile for processing. 

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
