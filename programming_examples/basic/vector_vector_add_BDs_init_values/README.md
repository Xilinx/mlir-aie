<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Vector Vector Add with BD-level Syntax and Initial Values</ins>

A simple binary operator, which uses a single AIE core to add two vectors.  The overall vector size in this design is `256` and is processed by the core in smaller sub tiles of size `16`.  This reference design can be run on either a Ryzen™ AI NPU or a VCK5000.

Two pedagogical points, both visible in the design body:

1. **BD-level data movement.**  Instead of letting `ObjectFifo` manage routing, DMA, and lock handshakes, this design hand-wires them via the iron BD-level primitives `Flow`, `Lock`, `TileDma`, `DmaChannel`, `Bd`, `Acquire`, `Release`.  The core body explicitly acquires and releases the producer / consumer locks that synchronise with each BD.

2. **`PreInitializedConstantBuffer`.**  The second operand is a 256-element constant baked into the core's L1 at design startup via a thin `Buffer` subclass (`PreInitializedConstantBuffer(np.arange(256))`) defined in this example file.  This demonstrates the `Buffer(initial_value=...)` mechanism as a named, reusable component — no shim DMA is needed for that operand.  See `programming_examples/basic/custom_dma/` for a richer user-side `Resolvable` example (`ScatterReadDMA`).

The kernel executes on AIE tile (`col`, 2).  The value of `col` is dependent on whether the application is targeting NPU or VCK5000.  Operand `A` is brought in from Shim tile (`col`, 0) via the BD-level path described above; operand `B` lives entirely on the AIE tile.  The Shim tile brings the output back out to external memory.

## Source Files Overview

1. `vector_vector_add.py`: defines the IRON design using the BD-level primitives (`Flow` / `Lock` / `TileDma` / `DmaChannel` / `Bd`).  Contains `PreInitializedConstantBuffer` (a thin `Buffer` subclass) and the `@iron.jit`-decorated `vector_vector_add` generator.  On NPU the Makefile drives `compile_mlir_module` (via `--xclbin-path` / `--insts-path`) to produce the XCLBIN and `insts.bin`; the VCK5000 path still goes through `aiecc`.

1. `test.cpp`: This C++ code is a testbench for the design example targeting Ryzen™ AI (AIE-ML). The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the program verifies the results.

1. `test_vck5000.cpp`: This C++ code is a testbench for the design example targeting the VCK5000 PCIe card (AIE). The code is responsible for configuring the AIEs, allocating memory, providing input data, and executing the AIE design on the VCK5000. After executing, the program verifies the results.

## Ryzen™ AI Usage

### C++ Testbench

To compile the design and C++ testbench:

```shell
make
make vector_vector_add.exe
```

To run the design:

```shell
make run
```

## VCK5000 Usage

### C++ Testbench

To compile the design and C++ testbench:

```shell
make vck5000
```

To run the design:

```shell
./test.elf
```

