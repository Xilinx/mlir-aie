<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

## MLIR Representation


The final core component, the lock, is actually a sub component of the `AIE.tile` and is central to the pipelined data movement / data processing model of the AI Engine array. It is  declared as `AIE.lock(tileName, lockID)`. An example would be:
```
%lock3_0 = AIE.lock(%tile13, 0)
%lock3_1 = AIE.lock(%tile13, 1)
```
Each tile has 16 locks and each lock is in one of two states (acquired, released) and one of two values (0 - write, 1 - read). The values do not have an explicit meaning but are regularly defined that way. A given tile's locks are accessible only by its neighbors for the purpose of synchronizing shared memory access. 


### Communication
The second major category of basic MLIR-based AI Engine components are to facilitate data communication. As described in the communication section of the basic AI Engine architecture, we communicate data with local memory, stream switches and cascade. 