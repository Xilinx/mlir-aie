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

Single tile performs a very simple `+` operation where the kernel loads data from local memory, increments the value by `1` and stores it back. The DMA in the Shim tile is programmed to bring the bottom left `8x16` portion of a larger `16x128` matrix into the tile to perform the operation.

The kernel executes on AIE tile (6, 2). Input data is brought to the local memory of the tile from Shim tile (6, 0). The Shim tile is programmed with a 2D DMA to only bring a 2D submatrix into the AIE tile for processing. 

To compile the design:
```
make
```

To run the design:
```
make run
```

