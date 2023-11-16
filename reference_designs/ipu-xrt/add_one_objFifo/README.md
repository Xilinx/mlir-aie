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

Single tile performs a very simpler + operateion where the kernel loads data from local memory, increments the value by 1 and stores it back.

To compile desing in Windows:
```
make
make build/addOneObjfifo.exe
```

To run the design:
```
make run
```

