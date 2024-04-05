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

Single tile performs a very simple `+` operation where the kernel loads data from local memory, increments the value by `1` and stores it back. This reference design can be run on either a RyzenAI IPU or a VCK5000.

The kernel executes on an AIE tile (`col`, 2). Input data is brought to the local memory of the tile from Shim tile (`col`, 0). The value of `col` is dependent on whether the application is targetting IPU or VCK5000. The size of the input data from the Shim tile is `16xi32` and it is sent to the AIE tile with the same format. Output data from the AIE tile to the Shim tile follows the same process, in reverse.

To compile and run the design for IPU:
```
make
make run
```

To compile and run the design for VCK5000:
```
make vck5000
./test.elf
```

