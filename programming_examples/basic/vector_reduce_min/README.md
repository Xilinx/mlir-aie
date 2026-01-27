<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Vector Reduce Min:

This example showcases both **JIT** and **non-JIT** approaches for running IRON designs. A single tile performs a very simple reduction operation where the kernel loads data from local memory, performs the `min` reduction and stores the resulting value back.

Input data is brought to the local memory of the Compute tile from a Shim tile. The size of the input data `N` from the Shim tile is configurable (default: `1024xi32` for the non-JIT version, customizable via command-line arguments for the JIT version). The data is copied to the AIE tile, where the reduction is performed. The single output data value is copied from the AIE tile to the Shim tile. Both approaches offer different compilation workflows with the JIT version adding microseconds runtime overhead.

## Source Files Overview

### JIT Approach Files

1. **`vector_reduce_min_jit.py`**: A JIT (Just-In-Time) compiled version using IRON's `@iron.jit` decorator. This approach offers faster development iteration by compiling and executing the design at runtime, with support for command-line arguments to customize the number of elements.

### Non-JIT Approach Files

1. **`vector_reduce_min.py`**: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using `aiecc.py` to produce design binaries (ie. XCLBIN and inst.bin for the NPU in Ryzen™ AI). 

1. **`vector_reduce_min_placed.py`**: An alternative version of the design in `vector_reduce_min.py`, that is expressed in a lower-level version of IRON.

1. **`test.cpp`**: This C++ code is a testbench for the non-JIT design example targetting Ryzen™ AI (AIE2). The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the program verifies the results.

### Shared Files

1. **`reduce_min.cc`**: A C++ implementation of a vectorized `min` reduction operation for AIE cores. The code uses the AIE API, which is a C++ header-only library providing types and operations that get translated into efficient low-level intrinsics, and whose documentation can be found [here](https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/index.html).  The source can be found [here](../../../aie_kernels/aie2/reduce_min.cc).

## Usage

### JIT Approach (Just-In-Time Compilation)

The JIT approach uses IRON's `@iron.jit` decorator for runtime compilation, offering faster development iteration and more flexible parameterization.

#### Running the JIT Version

To run the JIT version with default parameters (1024 elements):
```shell
python vector_reduce_min_jit.py
```

To run with custom number of elements:
```shell
python vector_reduce_min_jit.py --num-elements 2048
```

Or using the short form:
```shell
python vector_reduce_min_jit.py -n 512
```

### Non-JIT Approach

The non-JIT approach uses traditional MLIR-AIE compilation where the design is compiled ahead-of-time to produce binaries.

#### Compilation

To compile the design:
```shell
make
```

To compile the placed design:
```shell
env use_placed=1 make
```

To compile the C++ testbench:
```shell
make vector_reduce_min.exe
```

#### C++ Testbench

To run the design:
```shell
make run
```

#### JIT vs Non-JIT Comparison

| Aspect | Non-JIT Approach | JIT Approach |
|--------|------------------|--------------|
| **Compilation** | Ahead-of-time via `aiecc.py` | Runtime compilation |
| **Development Speed** | Slower (manual make/compilation) | Faster (compilation integrated) |
| **Host Code** | C++ testbench (`test.cpp`) | Python script |
| **Performance** | Baseline execution time | Microseconds overhead from JIT runtime |
| **Flexibility** | Fixed at compile time | Runtime parameterization |
| **Use Case** | Explicit XCLBIN management | Dynamic compilation |
| **Binary Output** | Generates XCLBIN/inst.bin | Cached binaries in `NPU_CACHE_HOME` (defaults to `~/.npu/`) |

**When to use each approach:**
- **Use JIT** for rapid prototyping, experimentation, runtime flexibility, and when you don't need control over XCLBINs
- **Use non-JIT** when you need explicit XCLBIN control, working with existing MLIR-AIE workflows, or distributing pre-compiled binaries

