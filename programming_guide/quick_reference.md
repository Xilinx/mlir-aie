<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>IRON Quick Reference</ins>

## Python Bindings

| Function Signature  | Definition | Parameters | Return Type | Example | 
|---------------------|------------|------------|-------------|---------|
| `tile(column, row)` | Declare AI Engine tile | `column`: column index number <br> `row`: row index number | `<tile>` | ComputeTile = tile(1,3) |
| `external_func(name, inputs, output)` | Declare external kernel function that will run on AIE Cores|  `name`: external function name <br> `input`: list of input types <br> `output`: list of output types | `<external_func>` | scale_scalar = external_func("vector_scalar_mul_aie_scalar", inputs=[memRef_ty, memRef_ty, T.memref(1, T.i32()), T.i32()]) | |
| `npu_dma_memcpy_nd(metadata, bd_id, mem, sizes)` | configure n-dimensional DMA accessing external memory | `metadata`:  String with name of `object_fifo`<br> `bd_id`: Identifier number<br> `mem`: memory for transfer<br> `sizes`: 4-D transfer size in 4B granularity | `None` | npu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, 1, N]) |
| `npu_sync(column, row, direction, channel, column_num=1, row_num=1)` | configure host-ShimDMA syncronization for accessing external memory | `column` and `row`: Specify the tile location for initiating the synchronization. <br> `direction`: Indicates the DMA direction (0 for write to host, 1 for read from host). <br> `channel`: Identifies the DMA channel (0 or 1) for the synchronization token <br> `column_num` and `row_num` (optional): Define the range of tiles to wait for synchronization| `None` | npu_sync(column=0, row=0, direction=0, channel=1) |
| **Object FIFO** |||
| `object_fifo(name, producerTile, consumerTiles, depth, datatype)` | Construct Object FIFO | `name`: Object FIFO name <br> `producerTile`: producer tile object <br> `ConsumerTiles`: list of consumer tile objects <br> `depth`: number of object in Object FIFO <br> `datatype`: type of the objects in the Object FIFO| `<object_fifo>` | of0 = object_fifo("objfifo0", A, B, 3, T.memref(256, T.i32())) |
| `<object_fifo>.acquire(port, num_elem)` | Acquire from Object FIFO | `port`: `ObjectFifoPort.Produce` or `ObjectFifoPort.Consume` <br> `num_elem`: number of objects to acquire | `<objects>` | elem0 = of0.acquire(ObjectFifoPort.Produce, 1) |  |
| `object_fifo.release(port, num_elem)` | Release from Object FIFO | `port`: `ObjectFifoPort.Produce` or `ObjectFifoPort.Consume` <br> `num_elem`: | `None` | of0.release(ObjectFifoPort.Consume, 2) |
| `object_fifo_link(fifoIns, fifoOuts)` | Create a link between Object FIFOs | `fifoIns`: list of Object FIFOs (variables or names)<br> `fifoOuts`: list of Object FIFOs (variables or names) | `None` | object_fifo_link(of0, of1) |
| **Routing Bindings (relevant for trace and low-level design)** |||
| `flow(source, source_bundle, source_channel, dest, dest_bundle, dest_channel)` | Create a circuit switched flow between src and dest | `source`: source tile of the flow <br> `source_bundle`: type of source WireBundle (see full list in [AIEAttrs.td](../include/aie/Dialect/AIE/IR/AIEAttrs.td)) <br> `source_channel`: source channel index <br> `dest`: destination tile of the flow <br> `dest_bundle`: type of destination WireBundle (see full list in [AIEAttrs.td](../include/aie/Dialect/AIE/IR/AIEAttrs.td)) <br> `dest_channel`: destination channel index | `None` | flow(ComputeTile, WireBundle.DMA, 0, ShimTile, WireBundle.DMA, 1) | In the case when we're routing for trace, the srcPort and srcChannel can be WireBundle.Trace and 0 respectively|
| `packetflow(pkt_id, source, source_port, source_channel, dest, dest_port, dest_channel, keep_pkt_header)` | Create a packet switched flow between src and dest | `pkt_id`: unique packet ID <br>  `source`: source tile of the packet flow <br> `source_port`: type of source WireBundle (see full list in [AIEAttrs.td](../include/aie/Dialect/AIE/IR/AIEAttrs.td)) <br> `source_channel`: source channel index <br> `dest`: destination tile of the packet flow <br> `dest_port`: type of destination WireBundle (see full list in [AIEAttrs.td](../include/aie/Dialect/AIE/IR/AIEAttrs.td)) <br> `dest_channel`: destination channel index <br>`keep_pkt_header`: boolean flag to keep header | `None` | packetflow(1, ComputeTile2, WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1, keep_pkt_hdr=True) | Example shows trace routing. If you want to route from the core memory trace unit, then we would use channel 1 |
|||||

> **NOTE:** `tile`: The actual tile coordinates run on the device may deviate from the ones declared here. In Ryzen AI, for example, these coordinates tend to be relative coordinates as the runtime scheduler may assign it to a different available column.

> **NOTE:** `object_fifo`: The `producerTile` and `consumerTiles` inputs are AI Engine tiles. The `consumerTiles` may also be specified as an array of tiles for multiple consumers.

> **NOTE:** `<object_fifo>.{acquire,release}`: The output may be either a single object or an array of objects which can then be indexed in an array-like fashion.

> **NOTE:** `object_fifo_link` The tile that is used as the shared tile in the link must currently be a Mem tile. The inputs `fifoIns` and `fifoOuts` may be either a single Object FIFO or a list of them. Both can be specified either using their python variables or their names. Currently, if one of the two inputs is a list of ObjectFIFOs then the other can only be a single Object FIFO.

## Python helper functions
| Function Signature | Description |
|--------------------|-------------|
| `print(ctx.module)` | Converts our ctx wrapped structural code to mlir and prints to stdout|
| `ctx.module.operation.verify()` | Runs additional structural verficiation on the python binded source code and return result to stdout |


## AI Engine documentation
* [Summary Documentation Links in UG1076](https://docs.amd.com/r/en-US/ug1076-ai-engine-environment/Documentation)
* [AIE1 Architecture Manual - AM009](https://docs.amd.com/r/en-US/am009-versal-ai-engine/Overview)
* [AIE1 Register Reference - AM015](https://docs.amd.com/r/en-US/am015-versal-aie-register-reference/Overview)
* [AIE2 Architecture Manual - AM020](https://docs.amd.com/r/en-US/am020-versal-aie-ml/Overview)
* [AIE2 Register Reference - AM025](https://docs.amd.com/r/en-US/am025-versal-aie-ml-register-reference/Overview)
* [AIE API User Guide - v2023.2](https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_intrinsics/intrinsics/index.html)

## AIE Detailedd References
* [AIE2 - Table of suported data types and vector sizes (AIE API)](https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/group__group__basic__types.html)
