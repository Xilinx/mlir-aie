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

When we program the AIE-array, we need to declare and configure its structural building blocks: compute tiles for vector processing, memory tiles as larger level-2 shared scratchpads, and shim tiles supporting data movement to external memory. In this programming guide, we will be utilizing the IRON python bindings for MLIR-AIE components to describe our design at the tile level of granularity. Later on, when we focus on kernel programming, we will explore vector programming in C/C++. But let's first look at a basic python source file (named [aie2.py](./aie2.py)) for an IRON design.

## <ins>Walkthrough of python source file (aie2.py)</ins>
At the top of this python source, we include modules that define the IRON AIE language bindings `aie.dialects.aie` and the mlir-aie context `aie.extras.context` which binds to MLIR definitions for AI Engines.

```
from aie.dialects.aie import * # primary mlir-aie dialect definitions
from aie.extras.context import mlir_mod_ctx # mlir-aie context
```
Then we declare a structural design function that will expand into mlir code when it will get called from within an mlir-aie context (see last part of this subsection).
```
# AI Engine structural design function
def mlir_aie_design():
    <... AI Engine device, blocks and connections ...>
```
Let's look at how we declare the AI Engine device, blocks and connections. We start off by declaring our AIE device via `@device(AIEDevice.npu)` or `@device(AIEDevice.xcvc1902)`. The blocks and connections themselves will then be declared inside the `def device_body():`. Here, we instantiate our AI Engine blocks, which in this first example are simply AIE compute tiles. 

The arguments for the tile declaration are the tile coordinates (column, row) and we assign it a variable tile name in our python program.

> **NOTE:**  The actual tile coordinates used on the device when the program is run may deviate from the ones declared here. For example, on the NPU on Ryzen™ AI (`@device(AIEDevice.npu)`), these coordinates tend to be relative coordinates as the runtime scheduler may assign it to a different available column during runtime.

```
    # Device declaration - here using aie2 device NPU
    @device(AIEDevice.npu)
    def device_body():

        # Tile declarations
        ComputeTile = tile(1, 3)
        ComputeTile = tile(2, 3)
        ComputeTile = tile(2, 4)
```
Once we are done declaring our blocks (and connections) within our design function, we move onto the main body of our program where we call the function and output our design in MLIR. This is done by first declaring the MLIR context via the `with mlir_mod_ctx() as ctx:` line. This indicates that subsequent indented python code is in the MLIR context and we follow this by calling our previosly defined design function `mlir_aie_design()`. This means all the code within the design function is understood to be in the MLIR context and contains the IRON custom python binding definitions of the more detailed mlir block definitions. The final line is `print(ctx.module)` which takes the code defined in our MLIR context and prints it stdout. This will then convert our python binded code to its MLIR equivalent and print it to stdout. 
```
# Declares that subsequent code is in mlir-aie context
with mlir_mod_ctx() as ctx:
    mlir_aie_design() # Call design function within the mlir-aie context
    print(ctx.module) # Print the python-to-mlir conversion to stdout
```

## <ins>Other Tile Types</ins>
Next to the compute tiles, an AIE-array also contains data movers for accessing L3 memory (also called shim DMAs) and larger L2 scratchpads (called mem tiles) which are available since the AIE-ML generation - see [the introduction of this programming guide](../README.md). Declaring these other types of structural blocks follows the same syntax but requires physical layout details for the specific target device. Shim DMAs typically occupy row 0, while mem tiles (when available) often reside on the following row(s). The following code segment declares all the different tile types found in a single NPU column.

```
    # Device declaration - here using aie2 device NPU
    @device(AIEDevice.npu)
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
1. To run our python program from the command line, we type `python3 aie2.py` which converts our python structural design into MLIR source code. This works from the command line if our design environment already contains the mlir-aie python binded dialect module. We included this in the [Makefile](./Makefile) so go ahead and run `make` now. Then take a look at the generated MLIR source under `build/aie.mlir`.

2. Run `make clean` to remove the generated files. Then introduce an error to the python source such as misspelling `tile` to `tilex` and then run `make` again. What messages do you see? <img src="../../mlir_tutorials/images/answer1.jpg" title="There is python error because tilex is not recognized." height=25>

3. Run `make clean` again. Now change the error by renaming `tilex` back to `tile` but change the coordinates to (-1,3) which is an inavlid location. Run `make` again. What messages do you see now? <img src="../../mlir_tutorials/images/answer1.jpg" title="No error is generated." height=25>

4. No error is generated but our code is invalid. Take a look at the generated MLIR code under `build/aie.mlir`. This generated output is invalid MLIR syntax and running our mlir-aie tools on this MLIR source will generate an error. We do, however, have some additional python structural syntax checks that can be enabled if we use the function `ctx.module.operation.verify()`. This verifies that our python binded code has valid operation within the mlir-aie context. 

    Qualify the `print(ctx.module)` call with a check on `ctx.module.operation.verify()` using a code block like the following:
    ```
    res = ctx.module.operation.verify()
    if(res == True):
        print(ctx.module)
    else:
        print(res)
    ```
    Make this change and run `make` again. What message do you see now? <img src="../../mlir_tutorials/images/answer1.jpg" title="It now says column value fails to satisfy the constraint because the minimum value is 0" height=25>

-----
[[Prev - Section 0](../section-0/)] [[Top](..)] [[Next - Section 2](../section-2/)]
