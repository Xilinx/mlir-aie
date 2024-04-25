<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Vector Vector Add</ins>

A simple binary operator, which uses a single AIE core to add two vectors together.  The overall vector size in this design is `1024` and it processed by the core in smaller sub tiles of size `16`.  It shows how simple it can be to just feed data into the AIEs using the ObjectFIFO abstraction, and drain the results back to external memory.  This reference design can be run on either a RyzenAI NPU or a VCK5000. 

The kernel executes on AIE tile (`col`, 2). Both input vectors are brought into the tile from Shim tile (`col`, 0). The value of `col` is dependent on whether the application is targetting NPU or VCK5000. The AIE tile performs the summation operations and the Shim tile brings the data back out to external memory.

## Ryzen™ AI Usage

### C++ Testbench

To compile the design and C++ testbench:

```
make
make vectorAdd.exe
```

To run the design:

```
make run
```

## VCK5000 Usage

### C++ Testbench

To compile the design and C++ testbench:

```
make vck5000
```

To run the design:

```
./test.elf
```

