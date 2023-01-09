<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Tutorial 8 - communication (cascade)</ins>

Cascade is a specialized data transfer between horizontally adjacent tiles in a specific direction depending on the row. For first generation AI Engines, cascades flow in the following way:
* Even rows - right to left
* Odd rows - left to right
> The first row starts at index 0 and is the shim row. Therefore, the first regular tile row is actually an odd row at row index 1 and flows left to right. Note that the last tile in each row cascades to the tile directly above it.

As far as  `mlir-aie` syntax support for cascade connections, we support cascade functionalty by simply placing tiles directly horizontally adjacnet to one another. No additional operators are needed in `mlir-aie`. There are `core` ops that push and pull data from the cascade ports

```
AIE.getCascade()
AIE.putCascade($cascadeValue : type($cascadeValue))
```

This allows us to push and pull data off the cascade ports.
> This feature is currently not working so we rely instead on external compiled kernel functions to push and pull cascade data in our example desgn

## <ins>Tutorial 5 Lab </ins>

1. Read through the [aie.mlir](aie.mlir) design. Based on the tile locations, which tile is pushing cascade data and which tile is pulling it? <img src="../images/answer1.jpg" title="tile(1,3) is sending cascade data to tile(2,3)" height=25>

2. Make the design and simulate ...

3. Move the design to an even row and make the approrpiate code changes. Verify your design works in simulation/ on the board.
