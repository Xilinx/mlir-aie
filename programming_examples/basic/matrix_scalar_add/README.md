<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Matrix Scalar Addition</ins>

This design shows an extremely simple single AIE design, which is incrementing every value in an input matrix.

It shows a number of features which can then be expanded to more realistic designs.  

Firstly, a 2D DMA pattern is set up to access data from the input and output memories. Small `8x16` subtiles are accessed from the larger `16x128` input and output matrix.  Thinking about input and output spaces are large grids, with smaller grids of work being dispatched to individual AIE cores is a fundamental, reusable concept.

Secondly, the design shows how the bodies of work done by each AIE core is a combination of data movement (the object FIFO acquire and releases) together with compute, which in this case is expressed using a number of different MLIR dialects, like arith, memref, etc. next to mlir-aie.

Finally, the overall structural design shows how complete designs are a combination of a static design, consisting of cores, connections and some part of the data movement, together with a run time sequence for controlling the design.

## Functionality

A single AIE core performs a very simple `+` operation where the kernel loads data from its local memory, increments the value by `1` and stores it back to the local memory. The DMA in the Shim tile is programmed to bring the bottom left `8x16` portion of a larger `16x128` matrix into the tile to perform the operation. This reference design can be run on either a RyzenAI NPU or a VCK5000.

The kernel executes on AIE tile (`col`, 2) - this is actually the first core in a column, as the shim tile is on row 0, and the mem tile is on row 1. Input data is brought to the local memory of the tile from Shim tile (`col`, 0). The value of `col` is dependent on whether the application is targeting NPU or VCK5000. 


## Usage

### NPU

To compile the design and C++ testbench:


```
make
make matrixAddOne
```

To run the design:

```
make run
```

### VCK5000

To compile the design and C++ testbench:
```
make vck5000
```

To run the design 

```
./test.elf
```
