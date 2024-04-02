<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Passthrough</ins>

This test creates a loopback connection between an input and output channel of the DMA, copying the data from a source buffer to a destination buffer. This test does not perform any computation on the data. This reference design can be run on either a RyzenAI IPU or a VCK5000.

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

