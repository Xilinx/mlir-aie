<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 1 - Basic AI Engine building blocks</ins>

When we program the AIE-array, we need to declare and configure its structural building blocks: compute tiles for vector processing, memory tiles as larger level-2 shared scratchpads, and shim tiles supporting data movement to external memory. In this programming guide, we will be utilizing the IRON python bindings for MLIR-AIE components to describe our design at the tile level of granularity. Later on, when we focus on kernel programming, we will explore vector programming in C/C++. But let's first look at a basic python source file (named [aie2.py](./aie2.py)) for an MLIR-AIE design.

## <ins>Walkthrough of python source file (aie2.py)</ins>
At the top of this python source, we include modules that define the mlir-aie dialect and the mlir ctx wrapper which encapsulates the definition of our AI Engine enabled device (e.g. ipu or xcvc1902) and its associated structural building blocks.

```
from aie.dialects.aie import *                     # primary mlir-aie dialect definitions
from aie.extras.context import mlir_mod_ctx        # mlir ctx wrapper 
```
Then we declare a structural design function that expands into mlir code when called because it contains the ctx wrapper. This wrapper, defined in the `mlir_mod_ctx()` module, contains custom python bindings definitions that leverage python to simplify some of the more detailed mlir block definitions. 
```
# AI Engine structural design function
def mlir_aie_design():
    # ctx wrapper - to convert python to mlir
    with mlir_mod_ctx() as ctx:
```
Within our ctx wrapper, we finally get down to declaring our AIE device via `@device(AIEDevice.ipu)` or `@device(AIEDevice.xcvc1902)` and the blocks within the device. Inside the `def device_body():` , we instantiate our AI Engine blocks, which in this first example are simply AIE compute tiles. The arguments for the tile declaration are the tile coordinates (column, row) and we assign it a variable tile name in our python program.

> **NOTE:**  The actual tile coordinates run on the device may deviate from the ones declared here. For example, on the NPU on Ryzen AI (`@device(AIEDevice.ipu)`), these coordinates tend to be relative coordinates as the runtime scheduler may assign it to a different available column.

```
        # Device declaration - here using aie2 device xcvc1902
        @device(AIEDevice.xcvc1902)
        def device_body():

            # Tile declarations
            ComputeTile = tile(1, 3)
            ComputeTile = tile(2, 3)
            ComputeTile = tile(2, 4)
```
Once we are done declaring our blocks (and connections), we print the ctx wrapped design and the python defined design is then converted to mlir and printed to stdout. We finish our python code by calling the structural design function that we defined.
```
    # print the mlir conversion
    print(ctx.module)

# Call my program
mlir_aie_design()
```

## <ins>Other Tile Types</ins>
Next to the compute tiles, an AIE-array also contains data movers for accessing L3 memory (also called shim DMAs) and larger L2 scratchpads (called mem tiles) which are available since the AIE-ML generation - see [the introduction of this programming guide](../README.md). Declaring these other types of structural blocks follows the same syntax but requires physical layout details for the specific target device. Shim DMAs typically occupy row zero, while mem tiles (when available) often reside on the following row(s). The following code segment declares all the different tile types found in a single NPU column.

```
        # Device declaration - here using aie2 device ipu
        @device(AIEDevice.ipu)
        def device_body():

            # Tile declarations
            ShimTile     = tile(0, 0)
            MemTile      = tile(0, 1)
            ComputeTile1 = tile(0, 2)
            ComputeTile2 = tile(0, 3)
            ComputeTile3 = tile(0, 4)
            ComputeTile4 = tile(0, 5)
```

## <u>Exercises</u>
1. To run our python program from the command line, we type `python3 aie2.py` which converts our python structural design into mlir source code. This works from the command line if our design environment already contains the mlir-aie python binded dialect module. We included this in the [Makefile](./Makefile) so go ahead and run `make` now. Then take a look at the generated mlir source under `build/aie.mlir`.

2. Run `make clean` to remove the generated files. Then introduce an error to the python source such as misspelling `tile` to `tilex` and then run `make` again. What messages do you see? <img src="../../mlir_tutorials/images/answer1.jpg" title="There is python error because tilex is not recognized." height=25>

3. Run `make clean` again. Now change the error by renaming `tilex` back to `tile` but change the coordinates to (-1,3) which is an inavlid location. Run `make` again. What messages do you see now? <img src="../../mlir_tutorials/images/answer1.jpg" title="No error is generated." height=25>

4. No error is generated but our code is invalid. Take a look at the generated mlir code under `build/aie.mlir`. This generated mlir syntax is invalid and running our mlir-aie tools on this mlir source will generate an error. We do, however, have some additional python structural syntax checks that can be enabled if we change the `print(ctx.module)` to `print(ctx.module.operation.verify())`. Make this change and run `make` again. What message do you see now? <img src="../../mlir_tutorials/images/answer1.jpg" title="It now says column value fails to satisfy the constraint because the minimum value is 0" height=25> 

