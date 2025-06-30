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

When we program the AIE-array, we need to declare and configure its structural building blocks: compute tiles for vector processing, memory tiles as larger level-2 shared scratchpads, and shim tiles supporting data movement to NPU-external memory (i.e., main memory). In this programming guide, we will utilize the IRON Python library, which allows us to describe our overall NPU design, including selecting which AI Engine tiles we wish to use, what code each tile should run, how to move data between tiles, and how our design can be invoked from the CPU-side. Later on, we will explore vector programming in C/C++, which will be useful for optimizing computation kernels for individual compute tiles.

## <ins>Walkthrough of Python source file (aie2.py)</ins>

Let's first look at a basic Python source file (named [aie2.py](./aie2.py)) for an IRON design at the highest level of abstraction:

At the top of this Python source, we include modules that define the IRON libraries `aie.iron` for high-level abstraction constructs, resource placement algorithms `aie.iron.placers` and target architecture `aie.iron.device`.
```python
from aie.iron import Program, Runtime, Worker, LocalBuffer
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, Tile
```
Data movement inside the AIE-array is usually also declared at this step, however that part of design configuration has its own dedicated [section](../section-2/) and is not covered in detail here.
```python
# Dataflow configuration
# described in a future section of the guide...
```
In the AIE array, computational kernels are run on compute tiles, which are represented by Workers. 
A Worker takes as input a routine to run, and the list of arguments needed to run it. The Worker class is defined below and can be found in [worker.py](../../python/iron/worker.py). The Worker can be explicitly placed on a `placement` tile in the AIE array or its the placement can be left to the compiler, as is explained further in this section. Finally, the `while_true` input is set to True by default as Workers typically run continuously once the design is started.
```python
class Worker(ObjectFifoEndpoint):
    def __init__(
        self,
        core_fn: Callable | None,
        fn_args: list = [],
        placement: PlacementTile | None = AnyComputeTile,
        while_true: bool = True,
    )
```
In our simple design there is only one Worker which will perform the `core_fn` routine. The compute routine iterates over a local data buffer and initializes each entry to zero. The compute routine in this case has no inputs. As we will see in the next section of the guide, computational tasks usually run on data that is brought into the AIE array from external memory and the output produced is sent back out. Note that in this example design the Worker is explicitly placed on a Compute tile with coordinates (0,2) in the AIE array.
```python
# Task for the worker to perform
def core_fn():
    local = LocalBuffer(data_ty, name="local")
    for i in range_(data_size):
        local[i] = 0

# Create a worker to perform the task
my_worker = Worker(core_fn, [], placement=Tile(0, 2), while_true=False)
```
> **NOTE 1:**  Did you notice the underscore in `range_`? Although IRON makes NPU designs look mostly like normal Python programs, it is important to understand that the code you write here is _not_ directly executed on the NPU; instead, the code you write in an IRON design _generates other code_ (metaprogramming), kind of like if you wrote a print-statement with a string of code inside. Our toolchain then compiles this generated other code, and it can then run directly on the NPU. 
>
> All of this means that if you wrote `range` instead of `range_` in the example above, the resulting generated NPU code would contain a many `local[i] = 0` instructions, but no loop at all (the loop is "unrolled", which can lead to a large binary and means the number of loop iterations must be fixed at code-generation-time). On the other hand, when you use `range_`, Python only executes the loop body once (to collect the instructions contained therein), then emits a loop into the NPU code. The NPU then executes the loop.
> The same applies to other branching constructs like `if`; using Python's native construct will mean no actual branches are emitted for the NPU code!

> **NOTE 2:**  The Worker in the code above is instantiated with `while_true=False`. By default, this attribute is set to `True`, in which case the kernel code expressed by the task will be wrapped in a for loop that iterates until `sys.maxsize` with a step of one. This simulates a `while(True)` with the intention to loop over the code in the Worker infinitely. Depending on the task code, such as when creating a local buffer with a unique name, this can cause compiler issues.

In the previous code snippet it was mentioned that the data movement between Workers needs to be configured. This does not include data movement to/from the AIE array which is handled inside the `Runtime` sequence. The programming guide has a dedicated [section](../section-2/section-2d/) for runtime data movement. In this example, as we do not look in-depth at data movement configuration, the runtime sequence will only start the Worker.
```python
# Runtime operations to move data to/from the AIE-array
rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (_, _, _):
    rt.start(my_worker)
```
All the components are tied together into a `Program` which represents all design information needed to run the design on a device. It is also at this stage that the previously unplaced Workers are mapped onto AIE tiles using a `Placer`. Currently, only one placement algorithm is available in IRON, the `SequentialPlacer()` as is seen in the code snippet below. Other placers can be added with minimal effort and we encourage all users to experiment with these tools which can be found in [placers.py](../../python/iron/placers.py). Finally, the program is printed to produce the corresponding MLIR definitions from the IRON library and python language bindings.
```python
# Create the program from the device type and runtime
my_program = Program(NPU1Col1(), rt)

# Place components (assign them resources on the device) and generate an MLIR module
module = my_program.resolve_program(SequentialPlacer())

# Print the generated MLIR
print(module)
```

> **NOTE:**  All components described or mentioned above inherit from the `resolvable` interface which defers the creation of MLIR operations until their `resolve()` function is called. That is the task of the `resolve_program()` function of the `Program` which will raise an error if one of the IRON classes does not have enough information to generate its MLIR equivalent.

## <ins>Walkthrough of Python source file (aie2_placed.py)</ins>

IRON also enables users to describe their design at the tile level of granularity where components are explicitly placed on AIE tiles using coordinates. Let's again look through a basic Python source file (named [aie2_placed.py](./aie2_placed.py)) for an IRON design at this level.

At the top of this Python source, we include modules that define the IRON AIE libraries `aie.dialects.aie` and the mlir-aie context `aie.extras.context`, which binds to MLIR definitions for AI Engines.
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
Let's look at how we declare the AI Engine device, blocks, and connections. We start off by declaring our AIE device via `@device(AIEDevice.npu1_1col)` or `@device(AIEDevice.npu2)`. The blocks and connections themselves will then be declared inside the `def device_body():`. Here, we instantiate our AI Engine blocks, which are AIE compute tiles in this first example.

The arguments for the tile declaration are the tile coordinates (column, row). We assign each declared tile to a variable in our Python program.

> **NOTE:**  The actual tile coordinates used on the device when the program is run may deviate from the ones declared here. For example, on the NPU on Ryzen™ AI (`@device(AIEDevice.npu)`), these coordinates tend to be relative coordinates as the runtime scheduler may assign it to a different available column during runtime.

```python
    # Device declaration - here using aie2 device NPU
    @device(AIEDevice.npu1)
    def device_body():

        # Tile declarations
        ComputeTile1 = tile(1, 3)
        ComputeTile2 = tile(2, 3)
        ComputeTile3 = tile(2, 4)
```
Compute cores can be mapped to compute tiles. They can also be linked to external kernel functions that can then be called from within the body of the core, however that is beyond the scope of this section and is explained further in the guide. In this example design the compute core declares a local data tensor, iterates over it and initializes each entry to zero.
```python
        data_size = 48
        data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]

        # Compute core declarations
        @core(ComputeTile1)
        def core_body():
            local = buffer(ComputeTile1, data_ty, name="local")
            for i in range_(data_size):
                local[i] = 0
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

2. Run `make clean` to remove the generated files. In the worker's code (the `core_fn`) replace `range_` with `range` (no underscore). What do you expect to happen? Investigate the generated code in `build/aie.mlir` and observe how the generated code changed. <img src="../../mlir_tutorials/images/answer1.jpg" title="The generated MLIR code does not contain a loop; instead, the same instructions are repeated many times." height=25>

3. Run `make clean` again. Then introduce an error to the Python source, such as misspelling `sequence` to `sequenc`, and then run `make` again. What messages do you see? <img src="../../mlir_tutorials/images/answer1.jpg" title="There is a Python error because sequenc is not recognized." height=25>

4. Run `make clean` again. Now change the error by renaming `sequenc` back to `sequence`, but place the Worker on a tile with coordinates (-1, 3), which is an invalid location. Run `make` again. What message do you see now? <img src="../../mlir_tutorials/images/answer1.jpg" title="There is a partial placement error." height=25>

5. Run `make clean` again. Restore the Worker tile to its original coordinates. Remove the `while_true=False` attribute from the Worker and run `make` again. What do you observe? <img src="../../mlir_tutorials/images/answer1.jpg" title="The Worker task code is nested within a for loop." height=25>

6. Now let's take a look at the placed version of the code. Run `make placed` and look at the generated MLIR source under `build/aie_placed.mlir`.

7. Run `make clean` to remove the generated files. Introduce the same error as above by changing the coordinates of `ComputeTile1` to (-1,3). Run `make placed` again. What message do you see now? <img src="../../mlir_tutorials/images/answer1.jpg" title="There is no error." height=25>

8. No error is generated but our code is invalid. Take a look at the generated MLIR code under `build/aie_placed.mlir`. This generated output is invalid MLIR syntax and running our mlir-aie tools on this MLIR source will generate an error. We do, however, have some additional Python structural syntax checks that can be enabled if we use the function `ctx.module.operation.verify()`. This verifies that our Python-bound code has valid operation within the mlir-aie context. 

    Qualify the `print(ctx.module)` call with a check on `ctx.module.operation.verify()` using a code block like the following:
    ```python
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
    ```
    Make this change and run `make placed` again. What message do you see now? <img src="../../mlir_tutorials/images/answer1.jpg" title="It now says 'column value fails to satisfy the constraint' because the minimum value is 0." height=25>

-----
[[Prev - Section 0](../section-0/)] [[Top](..)] [[Next - Section 2](../section-2/)]
