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

The kernel executes on AIE tile (0, 2). Input data is brought to the local memory of the tile from Shim tile (0, 0), through Mem tile (0, 1). The size of the input data from the Shim tile is `16xi32`. The data is stored in the Mem tile and sent to the AIE tile in smaller pieces of size `8xi32`. Output data from the AIE tile to the Shim tile follows the same process, in reverse.

To compile desing in Windows:
```
make
make build/addOneObjfifo.exe
```

To run the design:
```
make run
```

