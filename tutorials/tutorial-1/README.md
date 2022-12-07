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
In the MLIR-based AI Engine representation, every physical component including connections are declared at the top level. All parameters and customizations of these components are then elaborated within the component body. We generally write the MLIR code in a file with the .mlir file extension as it integrates well with the lit based auto-test of LLVM, such as those found in `test` sub-folder.
### Core Components
The two major components of an AI Engine tile is 
(1) the VLIW processor block declared as `AIE.tile(col,row)` and 
(2) the local memory block declared as `AIE.buffer(tileName) : memref<depthxdata_type>`. 
Examples include
```
%tile13 = AIE.tile(1,3)
%tile23 = AIE.tile(2,3)
%tile33 = AIE.tile(3,3)

%buff0 = AIE.buffer(%tile13) : memref<256xi32>
%buff1 = AIE.buffer(%tile13) : memref<256xi32>
```
For the tile, we simple declare its coordinates by column and row 
>**Note:** index values start at 0, with row 0 belonging to the shim which is not a full regular row. The first regular row for AIE1 is row index 1. 

We assign this to an MLIR variable which we can refer to when declaring the associated local memory (buffer). Here, the buffers are defined by a depth and data type width (though the local memory itself is not physically organized in this way). 

The final core component, the lock, is actually a sub component of the `AIE.tile` and is central to the pipelined data movement / data processing model of the AI Engine array. It is  declared as `AIE.lock(tileName, lockID)`. An example would be:
```
%lock3_0 = AIE.lock(%tile13, 0)
%lock3_1 = AIE.lock(%tile13, 1)
```
Each tile has 16 locks and each lock is in one of two states (acquired, released) and one of two values (0 - write, 1 - read). The values do not have an explicit meaning but are regularly defined that way. A given tile's locks are accessible only by its neighbors for the purpose of synchronizing shared memory access. 

We will be introducing more components and the ways these components are customized in subsequent tutorials. Additional definitions for these MLIR-based AI Engine components can be found in the github<area>.io docs [here](https://xilinx.github.io/mlir-aie/AIEDialect.html).

