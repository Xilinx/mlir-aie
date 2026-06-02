<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# <ins>Section 1 - Basic AI Engine building blocks</ins>

When we program the AIE-array, we need to declare and configure its structural building blocks: compute tiles for vector processing, memory tiles as larger level-2 shared scratchpads, and shim tiles supporting data movement to NPU-external memory (i.e., main memory). In this programming guide, we will utilize the IRON Python library, which describes our overall NPU design — selecting which AI Engine tiles to use, what code each tile should run, how to move data between tiles, and how the design is invoked from the CPU side. Later on, we will explore vector programming in C/C++, which is useful for optimizing computation kernels for individual compute tiles.

## <ins>Walkthrough of Python source file (aie2.py)</ins>

Let's look at a minimal IRON design in [aie2.py](./aie2.py). The whole design is one function decorated with `@iron.jit`: the first time you call it, IRON JIT-compiles the design and runs it on the attached NPU; `--emit-mlir` prints the lowered MLIR instead.

```python
import aie.iron as iron
from aie.iron import Buffer, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
```

A `Worker` represents the code that runs on one compute tile. It takes the function to run and the arguments needed to run it. The `tile` argument either pins the Worker to a specific compute tile, or you can leave it off to let the compiler place it. `while_true=False` here means "run once" — the default is `True`, which wraps the body in a `while True` loop so the Worker spins continuously.

```python
class Worker(ObjectFifoEndpoint):
    def __init__(
        self,
        core_fn: Callable | None,
        fn_args: list = [],
        tile: Tile = AnyComputeTile,
        while_true: bool = True,
    )
```

In our example, one Worker writes zeros into a local `Buffer`. The Worker is pinned to compute tile (0, 2):

```python
buf = Buffer(data_ty, name="buff")

def core_fn(buff):
    for i in range_(data_size):
        buff[i] = 0

my_worker = Worker(core_fn, [buf], tile=Tile(0, 2), while_true=False)
```

> **NOTE 1:** Did you notice the underscore in `range_`? Although IRON makes NPU designs look mostly like normal Python programs, the code you write here is _not_ directly executed on the NPU; instead, it _generates other code_ (metaprogramming) — kind of like writing a print statement with a string of code inside. Our toolchain then compiles that generated code so it can run on the NPU.
>
> If you wrote `range` instead of `range_`, the generated NPU code would contain many `buff[i] = 0` instructions back-to-back, with no loop at all (the loop is "unrolled"). On the other hand, when you use `range_`, Python executes the loop body once to collect the instructions, then emits a real loop into the NPU code. The same applies to other branching constructs like `if`; using Python's native `if` will emit no actual branches into the NPU code.

> **NOTE 2:** The Worker above is instantiated with `while_true=False`. By default this is `True`, which wraps the kernel body in a `while True`-style loop simulated by a `for _ in range(sys.maxsize):`. Depending on the body (e.g., creating a local buffer with a unique name) the infinite-loop wrapper can cause compiler issues.

Data movement between Workers will get its own [section](../section-2/section-2d/); host-to/from-NPU data movement is configured inside the `Runtime` sequence. In this minimal example the runtime sequence has one host-facing tensor argument and just starts the Worker:

```python
rt = Runtime()
with rt.sequence(data_ty) as _:
    rt.start(my_worker)
```

Finally we wrap everything in a `Program`. The program emits `aie.logical_tile` ops for any unplaced tiles (none here, since we pinned the Worker) and the `--aie-place-tiles` compiler pass assigns physical tile coordinates during compilation. Wrapping the design in `@iron.jit` (at the top of the function) means a call site like `section_one(out)` triggers compile + run end-to-end.

```python
@iron.jit
def section_one(b_out: Out):
    ...
    return Program(iron.get_current_device(), rt).resolve_program()
```

> **NOTE:** Every IRON component above inherits from the `resolvable` interface, which defers the creation of MLIR operations until `resolve()` is called. The `Program.resolve_program()` call ties them together and raises if anything is under-specified.

## <ins>Other Tile Types</ins>

Besides compute tiles, an AIE-array also contains data movers for accessing L3 memory (shim DMAs) and larger L2 scratchpads (mem tiles), which have been available since the AIE-ML generation — see [the introduction of this programming guide](../README.md). Shim DMAs typically occupy row 0; mem tiles (when available) often reside on row 1. In IRON, you usually let the compiler place these, but you can also pin them explicitly. The following snippet shows pinned `Tile(col, row)` declarations covering all the tile types found in a single NPU column:

```python
from aie.iron.device import Tile

ShimTile     = Tile(0, 0)
MemTile      = Tile(0, 1)
ComputeTile1 = Tile(0, 2)
ComputeTile2 = Tile(0, 3)
ComputeTile3 = Tile(0, 4)
ComputeTile4 = Tile(0, 5)
```

## <ins>Inspecting the generated MLIR</ins>

`@iron.jit` lowers your design through the AIE dialect on its way to a binary. To see the MLIR without running anything, pass `--emit-mlir`:

```shell
python3 aie2.py --emit-mlir
```

The Makefile also exposes `make emit-mlir` which redirects the output to `build/aie.mlir`.

## <u>Exercises</u>

1. Run `make` (the default target runs the design on the NPU) — you should see `PASS!`. Then run `make emit-mlir` and inspect `build/aie.mlir`.

2. Run `make clean`. In the worker's body, replace `range_` with `range` (no underscore). What changes in `build/aie.mlir`? <img src="../../mlir_exercises/images/answer1.jpg" title="The generated MLIR contains no scf.for loop; the same memref.store instructions are emitted many times in a row." height=25>

3. Run `make clean`. Introduce an error in the Python source — e.g., misspell `sequence` as `sequenc`. What message do you see? <img src="../../mlir_exercises/images/answer1.jpg" title="A Python AttributeError, raised before any MLIR is produced." height=25>

4. Run `make clean`. Restore the spelling, then change the Worker's tile to `Tile(-1, 3)` (an invalid location). What message do you see? <img src="../../mlir_exercises/images/answer1.jpg" title="A placement / tile-coordinate constraint error." height=25>

5. Run `make clean`. Restore the Worker tile to `(0, 2)`. Remove the `while_true=False` argument and run `make emit-mlir`. What changed in the MLIR? <img src="../../mlir_exercises/images/answer1.jpg" title="The core body is now nested inside an scf.for that bounds to sys.maxsize — the simulated while-true wrapper." height=25>

-----
[[Prev - Section 0](../section-0/)] [[Top](..)] [[Next - Section 2](../section-2/)]
