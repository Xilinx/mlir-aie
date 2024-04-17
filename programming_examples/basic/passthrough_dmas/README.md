<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Pass Through DMAs

Single tile passes data through using only the DMA infastructure to copy data from local memory source to destination. This serves to demonstrate the object FIFO link operation and the DMA capabilities to move data without involing the AIE core compute.

## Usage

### C++ Testbench

To compile the design and C++ testbench:

```
make
make build/passThroughDMAs.exe
```

To run the design:

```
make run
```