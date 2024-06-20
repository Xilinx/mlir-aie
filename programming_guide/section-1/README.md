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

When we program the AIE-array, we need to declare and configure its structural building blocks: compute tiles for vector processing, memory tiles as larger level-2 shared scratchpads, and shim tiles supporting data movement to external memory. In this programming guide, we will utilize the IRON Python bindings for MLIR-AIE components to describe our design at the tile level of granularity. Later on, we will explore vector programming in C/C++ when we focus on kernel programming. But let's first look at a basic Python source file (named [aie2.py](./aie2.py)) for an IRON design.

## <ins>Walkthrough of Python source file (aie2.py)</ins>
At the top of this Python source, we include modules that define the IRON AIE language bindings `aie.dialects.aie` and the mlir-aie context `aie.extras.context`, which binds to MLIR definitions for AI Engines.

```python
from aie.dialects.aie import * # primary mlir-aie dialect definitions
from aie.extras.context import mlir_mod_ctx # mlir-aie context
```
Then we declare a structural design function that will expand into MLIR code when it will get called from within an mlir-aie context (see last part of this subsection).
```python
# AI Engine structural design function
def mlir_aie_design():
    <... AI Engine device, blocks, and connections ...>
```
Let's look at how we declare the AI Engine device, blocks, and connections. We start off by declaring our AIE device via `@device(AIEDevice.npu1_1col)` or `@device(AIEDevice.xcvc1902)`. The blocks and connections themselves will then be declared inside the `def device_body():`. Here, we instantiate our AI Engine blocks, which are AIE compute tiles in this first example.

The arguments for the tile declaration are the tile coordinates (column, row). We assign each declared tile to a variable in our Python program.

> **NOTE:**  The actual tile coordinates used on the device when the program is run may deviate from the ones declared here. For example, on the NPU on Ryzen™ AI (`@device(AIEDevice.npu)`), these coordinates tend to be relative coordinates as the runtime scheduler may assign it to a different available column during runtime.

```python
    # Device declaration - here using aie2 device NPU
    @device(AIEDevice.npu1_1col)
    def device_body():

        # Tile declarations
        ComputeTile1 = tile(1, 3)
        ComputeTile2 = tile(2, 3)
        ComputeTile3 = tile(2, 4)
```
Once we are done declaring our blocks (and connections) within our design function, we move onto the main body of our program where we call the function and output our design in MLIR. This is done by first declaring the MLIR context via the `with mlir_mod_ctx() as ctx:` line. This indicates that subsequent indented Python code is in the MLIR context, and we follow this by calling our previously defined design function `mlir_aie_design()`. This means all the code within the design function is understood to be in the MLIR context and contains the IRON custom Python binding definitions of the more detailed MLIR block definitions. The final line is `print(ctx.module)`, which takes the code defined in our MLIR context and prints it to stdout. This will then convert our Python-bound code to its MLIR equivalent and print it to stdout. 
```python
# Declares that subsequent code is in mlir-aie context
with mlir_mod_ctx() as ctx:
    mlir_aie_design() # Call design function within the mlir-aie context
    print(ctx.module) # Print the Python-to-MLIR conversion to stdout
```

## <ins>Other Tile Types</ins>
Next to the compute tiles, an AIE-array also contains data movers for accessing L3 memory (also called shim DMAs) and larger L2 scratchpads (called mem tiles), which have been available since the AIE-ML generation - see [the introduction of this programming guide](../README.md). Declaring these other types of structural blocks follows the same syntax but requires physical layout details for the specific target device. Shim DMAs typically occupy row 0, while mem tiles (when available) often reside on row 1. The following code segment declares all the different tile types found in a single NPU column.

```python
    # Device declaration - here using aie2 device NPU
    @device(AIEDevice.npu1)
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
1. To run our Python program from the command line, we type `python3 aie2.py`, which converts our Python structural design into MLIR source code. This works from the command line if our design environment already contains the mlir-aie Python-bound dialect module. We included this in the [Makefile](./Makefile), so go ahead and run `make` now. Then take a look at the generated MLIR source under `build/aie.mlir`.

2. Run `make clean` to remove the generated files. Then introduce an error to the Python source, such as misspelling `tile` to `tilex`, and then run `make` again. What messages do you see? <img src="../../mlir_tutorials/images/answer1.jpg" title="There is a Python error because tilex is not recognized." height=25>

3. Run `make clean` again. Now change the error by renaming `tilex` back to `tile`, but change the coordinates to (-1,3), which is an invalid location. Run `make` again. What messages do you see now? <img src="../../mlir_tutorials/images/answer1.jpg" title="No error is generated." height=25>

4. No error is generated but our code is invalid. Take a look at the generated MLIR code under `build/aie.mlir`. This generated output is invalid MLIR syntax and running our mlir-aie tools on this MLIR source will generate an error. We do, however, have some additional Python structural syntax checks that can be enabled if we use the function `ctx.module.operation.verify()`. This verifies that our Python-bound code has valid operation within the mlir-aie context. 

    Qualify the `print(ctx.module)` call with a check on `ctx.module.operation.verify()` using a code block like the following:
    ```python
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
    ```
    Make this change and run `make` again. What message do you see now? <img src="../../mlir_tutorials/images/answer1.jpg" title="It now says 'column value fails to satisfy the constraint' because the minimum value is 0." height=25>

-----
[[Prev - Section 0](../section-0/)] [[Top](..)] [[Next - Section 2](../section-2/)]
