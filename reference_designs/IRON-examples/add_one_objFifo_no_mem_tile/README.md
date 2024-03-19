<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Add One (with ObjectFIFOs)</ins>

Single tile performs a very simple `+` operation where the kernel loads data from local memory, increments the value by `1` and stores it back.

The kernel executes on AIE tile (6, 2). Input data is brought to the local memory of the tile from Shim tile (6, 0). The size of the input data from the Shim tile is `16xi32` and it is sent to the AIE tile with the same format. Output data from the AIE tile to the Shim tile follows the same process, in reverse.

To compile the design:
```
make
```

To run the design:
```
make run
```

