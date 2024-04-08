<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 1 - Basic AI Engine building blocks (tiles and buffers)</ins>

When we program for AI Engines, our MLIR-AIE framework serves as the entry point to declare and configure the structural building blocks of the entire AI Engine array. Details for these building blocks, along with the general architecture of AI Engines are breifly described in the top page of [MLIR tutorials](../tutorials) materials. Read through that synopsis first before continuing here.

In this programming guide, we will be utilizing the python bindings of MLIR-AIE components to describe our system and the tile level. Later on, when we focus in on kernel programming, we will programming in C/C++. Let's first look at a basic python source file for a MLIR-AIE design.

```
from aie.dialects.aie import *                     # primary mlir-aie dialect definitions
from aie.extras.context import mlir_mod_ctx        # mlir ctx wrapper 

# My program definition
def mlir_aie_design():
    # ctx wrapper - needed to convert python to mlir
    with mlir_mod_ctx() as ctx:

        # device declaration - here using aie2 device xcvc1902
        @device(AIEDevice.xcvc1902)
        def device_body():

            # Tile declarations
            ComputeTile = tile(1, 4)

            # Buffer declarations
            ComputeBuffer = buffer(ComputeTile, (8,), T.i32(), name = "a14")

    # print the mlir conversion
    print(ctx.module)

# Call my program
mlir_aie_design()
```

* Introduce AI Engine building blocks with references to Tutorial material
* Give example of python binded MLIR source for defining tiles and buffers



