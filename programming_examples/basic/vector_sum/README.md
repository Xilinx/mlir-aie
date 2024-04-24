<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Vector sum</ins>

Single tile traverses through a vector in memory and returns the sum of each value in the vector. The tile that performs the operation is tile (`col`, 2) and the data is read from and written to external memory through Shim tile (`col`, 0). A buffer in tile (`col`, 2) is used to store the temporary maximum value during processing, which is then pushed through an object FIFO to the Shim tile when processing is complete. This reference design can be run on either a RyzenAI NPU or a VCK5000. The value of `col` is dependent on whether the application is targetting NPU or VCK5000.

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

