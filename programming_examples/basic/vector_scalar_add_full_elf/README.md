<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# Vector Scalar Addition with Full ELF

This design shows an extremely simple single AIE design, which is incrementing every value in an input vector.

It is the same as the (`vector_scalar_add`)[../vector_scalar_add/README.md] example, but expressed as a full-ELF design.

## Source Files Overview

1. `vector_scalar_add.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using `aiecc` to produce design binaries (ie. XCLBIN and inst.bin for the NPU in Ryzen™ AI).

1. `vector_scalar_add_placed.py`: An alternative version of the design in `vector_scalar_add.py`, that is expressed in a lower-level version of IRON.

1. `test.cpp`: This C++ code is a testbench for the design example. The code is responsible for loading the ELF file,providing input data, and executing the AIE design on the NPU. After executing, the program verifies the results.

## Usage

### Compilation

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
make vector_scalar_add.exe
```

### C++ Testbench

To run the design:

```shell
make run
```
