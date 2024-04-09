<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Quick Reference</ins>

## Python Bindings

| Syntax | Definition | Example | Notes |
|--------|------------|---------|-------|
| \<name\> = tile(column, row) | Declare AI Engine tile | ComputeTile = tile(1,3) | The actual tile coordinates run on the device may deviate from the ones declared here. In Ryzen AI, for example, these coordinates tend to be relative corodinates as the runtime scheduler may assign it to a different available column. |

## Object FIFO Bindings

| Syntax | Definition | Example | Notes |
|--------|------------|---------|-------|
| \<name\> = object_fifo(name, producerTile, consumerTiles, depth, datatype) | Initialize Object FIFO | of0 = object_fifo("objfifo0", A, B, 3, T.memref(256, T.i32())) | The `producerTile` and `consumerTiles` inputs are AI Engine tiles. The `consumerTiles` may also be specified as an array of tiles for multiple consumers. |
| \<name\> = \<objfifo_name\>.acquire(port, num_elem) | Acquire from Object FIFO | elem0 = of0.acquire(ObjectFifoPort.Produce, 1) | The `port` input is either `ObjectFifoPort.Produce` or `ObjectFifoPort.Consume`. The output may be either a single object or an array of objects which can then be indexed in an array-like fashion. |
| \<objfifo_name\>.release(port, num_elem) | Release from Object FIFO | of0.release(ObjectFifoPort.Consume, 2) | The `port` input is either `ObjectFifoPort.Produce` or `ObjectFifoPort.Consume`. |
| object_fifo_link(fifoIns, fifoOuts) | Create a link between Object FIFOs | object_fifo_link(of0, of1) | The inputs `fifoIns` and `fifoOuts` may be either a single Object FIFO or a list of them. Both can be specified either using their python variables or their names. Currently, if one of the two inputs is a list of ObjectFIFOs then the other can only be a single Object FIFO. |

## Python helper functions
| Function | Description |
|----------|-------------|
| print(ctx.module) | Converts our ctx wrapped structural code to mlir and prints to stdout|
| print(ctx.module.operation.verify()) | Runs additional structural verficiation on the python binded source code and prints to stdout |


