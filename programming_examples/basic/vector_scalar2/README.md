<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Vector Scalar</ins>

Single tile performs `vector * scalar` of size `4096`. The kernel does a `1024` vector multiply and is invoked multiple times to complete the full vector scalar compute. This reference design can be run on either a RyzenAI IPU or a VCK5000.

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

