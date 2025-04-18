<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Vector Vector Add</ins>

A simple binary operator, which uses a single AIE core to add two vectors together.The input vectors are processed by the core in smaller sub tiles of size `16`. The input vector size is configurable via command line arguments but must be multiple of `16`.  The example shows how simple it can be to just feed data into the AIEs using the `ObjectFifo` abstraction, and drain the results back to external memory. This reference design can be run on Ryzen™ AI NPU.

Both input vectors are brought into a Compute tile from a Shim tile. In the placed design, the value of `col` is dependent on whether the application is targeting NPU. The AIE tile performs the summation operations and the Shim tile brings the data back out to external memory.

## Source Files Overview

1. `vector_vector_add.py`: A Python script that defines a JIT-compiled AIE array structural design using MLIR-AIE operations alongside the host-side code for launching the kernel on the NPU in Ryzen™ AI. 

1. `vector_vector_add_placed.py`: An alternative version of the design in `vector_vector_add.py`, that is expressed in a lower-level version of IRON.


## Ryzen™ AI Usage

To run the design on Strix:

```shell
python3 vector_vector_add.py --device npu2
```

and on Phoenix:

```shell
python3 vector_vector_add.py --device npu
```


To run the placed design on Strix:

```shell
python3 vector_vector_add_placed.py --device npu2
```

and on Phoenix:

```shell
python3 vector_vector_add_placed.py --device npu
```


