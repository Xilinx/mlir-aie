<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 3b - Key Object FIFO Patterns</ins>

The Object FIFO primitive supports several data movement patterns through its inputs and its member functions. We will now describe each of the currently supported patterns and provide links to more in-depth practical code examples that showcase each of them. 

### Reuse

* TODO: explain the concept of a sliding window

### Broadcast

As was explained in the Introduction [section](../section-3a/README.md#initializing-an-object-fifo) section, the `consumerTiles` input can be either a single tile or an array of tiles. When the input is specified as an array of tiles, this creates a broadcast communication from a single producer tile to multiple consumer tiles. The data from the producer tile's memory module is sent to each of the consumer tiles' memory modules via the AXI stream interconnect, which handles the back-pressure from consumers with different execution times. The AXI stream is also where the data is copied at a low-level before being sent to each of the consumers.

For more low-level details regarding how the objects in the Object FIFO are transferred via the AXI stream through the DMAs of the producer and consumer tiles please see the MLIR-AIE tutorials [tutorials](/mlir-aie/tutorials/tutorial-7/).

Below is an example of an Object FIFO of depth 3 with one producer tile A and three consumer tiles B, C and D:
```
A = tile(1, 2)
B = tile(1, 3)
C = tile(2, 3)
D = tile(3, 3)
of0 = object_fifo("objfifo0", A, [B, C, D], 3, T.memref(256, T.i32()))
```

The `depth` input of an Object FIFO can also be specified as an array of integers, which describe the number of objects that are available to each tile (the producer tile as well as each consumer tile) when accessing the Object FIFO. This functionality of the Object FIFO primitive exposes what is actually going on at the hardware level when the data movement is established for a broadcast. The object pool of the Object FIFO is not a single structure but rather composed of several pools of objects that are allocated in the memory module of each tile involved in the data movement. Specifying the `depth` as an array of integers allows the user full control to set the sizes of the pools on each individual tile.

The main advantage of being able to specify the individual depths comes during a situation like the one showcased in the example below, which we refer to as a broadcast with a <u>skip-connection</u>. In the example below two Object FIFOs are created: `of0` is a broadcast from producer tile A to consumer tiles B and C, while `of1` is a 1-to-1 data movement from producer tile B to consumer tile C. We refer to `of1` as a skip-connection because it is a dependency between the two consumer tiles of the same broadcast connection. Furthermore, we can see in the code that is executing on its core that C requires one object from both `of0` and `of1` before it can proceed with its execution. However, B also requires an object from `of0` before it can produce the data for `of1`.
```
A = tile(1, 2)
B = tile(1, 3)
C = tile(2, 3)
of0 = object_fifo("objfifo0", A, [B, C], 1, T.memref(256, T.i32()))
of1 = object_fifo("objfifo1", B, C, 1, T.memref(256, T.i32()))

@core(C)
def core_body():
    elem0 = of0.acquire(ObjectFifoPort.Consume, 1)
    elem1 = of1.acquire(ObjectFifoPort.Consume, 1)
    call(test_func2, [elem0, elem1])
    of0.release(ObjectFifoPort.Consume, 1)
    of1.release(ObjectFifoPort.Consume, 1)

@core(B)
def core_body():
    elem0 = of0.acquire(ObjectFifoPort.Consume, 1)
    elem1 = of1.acquire(ObjectFifoPort.Produce, 1)
    call(test_func2, [elem0, elem1])
    of0.release(ObjectFifoPort.Consume, 1)
    of1.release(ObjectFifoPort.Produce, 1)
```
Because C is waiting on B, the two tiles do not have the same rate of consumption from the broadcast connection and this results in the production rate of A being impacted. To avoid this, an additional object is required by B for `of0` which can then be used to buffer data from `of0` while waiting for the data from B via `of1`. To achieve this `of0` is created with an array of integers for its `depth`:
```
of0 = object_fifo("objfifo0", A, [B, C], [1, 1, 2], T.memref(256, T.i32()))
```
where tiles A and B retain the original depth of 1 while C now has a depth of 2 objects.

### Object FIFO Link

By design an Object FIFO handles both the configuration of the data movement between the producer and consumer tiles as well as the allocation of objects over the memory modules of the tiles. In order to put data consumed from one Object FIFO into another Object FIFO the user could explicitly do this in the core code of a shared tile between the two FIFOs. However, if the goal is to simply copy data from one Object FIFO to the other without modifying it, doing it in the manner described above results in allocating more objects than necessary, i.e., the data being copied to the second Object FIFO is already available in the first one. Additionally, Shim tiles and Mem tiles do not have a core on which the copy can be done explicitly.

Instead of an explicit copy, the Object FIFO API provides an implicit copy via an `object_fifo_link`, which can be initialized using its class constructor (defined in [aie.py](../../../python/dialects/aie.py)):
```
class object_fifo_link(ObjectFifoLinkOp):
    def __init__(
        self,
        fifoIns,
        fifoOuts,
    )
```
A link allows the user to specify a set of input Object FIFOs via the `fifoIns` input and a set of output ones via the `fifoOuts` input. Each Object FIFO may be specified either using its `name` or its variable. Both inputs can be either a single Object FIFO or an array of them. It is required that there exists at least one shared tile between the consumer tiles of `fifoIns` and the producer tiles of `fifoOuts` for a link to be valid. This is because the implicit copy of data will be done using the DMAs of that tile.

Below is an example of a link created between two FIFOs `of0` and `of1`.
```
A = tile(1, 0)
B = tile(1, 1)
C = tile(1, 2)
of0 = object_fifo("objfifo0", A, B, 2, T.memref(256, T.i32()))
of1 = object_fifo("objfifo1", B, C, 2, T.memref(256, T.i32()))
object_fifo_link(of0, of1)
```
Similarly the link could've been created using the names of the FIFOs:
```
object_fifo_link("objfifo0", "objfifo1")
```

Depending on how many Object FIFOs are specified in `fifoIns` and `fifoOuts`, two different data patterns can be achieved: a Distribute or a Join. They are described in the two next subsections. Currently, it is not possible to do both patterns at once, i.e., if `fifoIns` is an array then `fifoOuts` can only be a single Object FIFO, and the to other way around.

### Link & Distribute

### Link & Join


