<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Tutorial 8 - Communication (cascade)</ins>

Cascade is a specialized data transfer between horizontally adjacent tiles in a specific direction depending on the row. For first generation AI Engines, cascades flow in the following way:
* Even rows - East to West
* Odd rows - West to East
> **Note:** The first row starts at index 0 and is the shim row. Therefore, the first regular tile row is actually an odd row at row index 1 and flows West to East. 

> **Note:** The last tile in each row cascades into the tile directly North of it which makes the full cascade chain snake from bottom to top.

As far as  `mlir-aie` syntax support for cascade connections, we support cascade functionality by simply placing tiles directly horizontally adjacent to one another. No additional operators are needed in `mlir-aie`. There are `core` ops that push and pull data from the cascade ports

```
AIE.get_cascade()
AIE.put_cascade($cascadeValue : type($cascadeValue))
```

This allows us to push and pull data off the cascade ports.
> **NOTE:** This feature is currently not working so we rely instead on external compiled kernel functions to push and pull cascade data in our example design

## <ins>Tutorial 8 Lab </ins>

1. Read through the [aie.mlir](aie.mlir) design. Based on the tile locations, which tile is pushing cascade data and which tile is pulling it? <img src="../images/answer1.jpg" title="tile(1,3) is sending cascade data to tile(2,3)" height=25>

2. Verify correct functionality by compiling the design and then simulating the design via make.
    ```
    make; make -C aie.mlir.prj/sim
    ```

3. Move the design to an even row and make the necessary code changes. Verify your design works in simulation.

### <ins>Challenge Exercise</ins>
4. Move the design one more time to the end of the row where the two tiles are in different rows and verify correct behavior in simulation.

