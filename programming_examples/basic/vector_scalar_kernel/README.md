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

Single tile performs `vector * scalar` of size `4096`. The kernel does a `1024` vector multiply and is invoked multiple times to complete the full vector scalar compute. The scale code is contained in scale.cc and linked against aie2.py which programs the data movement and connectivity of tiles.  This reference design can only be run on the VCK5000.

To compile and run the design for VCK5000:
```
make vck5000
./test.elf
```

