<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Vector max</ins>

This reference design can be run on either a RyzenAI IPU or a VCK5000.

Single tile traverses through a vector in memory and returns the maximum value in the vector. The tile that performs the operation is tile (`col`, 2) and the data is read from and written to external memory through Shim tile (`col`, 0). A buffer in tile (`col`, 2) is used to store the temporary maximum value during processing, which is then pushed through an object FIFO to the Shim tile when processing is complete. The value of `col` is dependent on whether the application is targetting IPU or VCK5000.


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

