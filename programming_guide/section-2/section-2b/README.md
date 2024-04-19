<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 2b - Key Object FIFO Patterns</ins>

The Object FIFO primitive supports several data movement patterns through its inputs and its member functions. We will now describe each of the currently supported patterns and provide links to more in-depth practical code examples that showcase each of them. 

### Reuse

During the previous [section](../section-2a/README.md#accessing-the-objects-of-an-object-fifo) it was mentioned that the Object FIFO acquire and release functions can be paired together to achieve the behaviour of a sliding window with data reuse. Specifically, this communication pattern occurs when a producer or a consumer of an Object FIFO releases less objects than it had previously acquired. As acquiring from an Object FIFO does not destroy the data, unreleased objects can be reused without requiring new copies of the data.

It is important to note that each new acquire function will return a new object or array of objects that a process can access, which <u>includes unreleased objects from previous acquires</u>. The process should always use the result of the <u>most recent</u> acquire call to access unreleased objects to ensure a proper lowering through the Object FIFO primitive.

In the example below `of0` is created between producer A and consumer B with a depth of 3 objects: object0, object1, and object2. Consumer B first acquires 2 elements from `of0` in the variable `elems`. As this is the first time that B acquires, it will have access to object0 and object1. B releases the oldest acquired object, object0, and keeps object1. The next time B acquires 2 elements in the variable `elems_2` it will have access to object1 from before and to the newly acquired object2. B again only releases a single object and keeps object2. Finally, the third time B acquires in `elems_3` it will have access to object2 and object0.
```
A = tile(1, 3)
B = tile(2, 4)
of0 = object_fifo("objfifo0", A, B, 3, T.memref(256, T.i32())) # 3 objects: object0, object1, object2

@core(B)
def core_body():
    elems = of0.acquire(ObjectFifoPort.Consume, 2) # acquires object0 and object1
    call(test_func2, [elems[0], elems[1]])
    of0.release(ObjectFifoPort.Consume, 1) # releases object0

    elems_2 = of0.acquire(ObjectFifoPort.Consume, 2) # acquires object2; object1 was previously acquired
    call(test_func2, [elems_2[0], elems_2[1]])
    of0.release(ObjectFifoPort.Consume, 1) # releases object1

    elems_3 = of0.acquire(ObjectFifoPort.Consume, 2) # acquires object0; object2 was previously acquired
```

### Broadcast

As was explained in the Introduction [section](../section-2a/README.md#initializing-an-object-fifo) section, the `consumerTiles` input can be either a single tile or an array of tiles. When the input is specified as an array of tiles, this creates a broadcast communication from a single producer tile to multiple consumer tiles. The data from the producer tile's memory module is sent to each of the consumer tiles' memory modules via the AXI stream interconnect, which handles the back-pressure from consumers with different execution times. The AXI stream is also where the data is copied at a low-level before being sent to each of the consumers. To achieve the broadcast, the lowering will use one output port of the producer tile to establish a connection to all consumer tiles.

For more low-level details regarding how the objects in the Object FIFO are transferred via the AXI stream through the DMAs of the producer and consumer tiles please see the MLIR-AIE tutorials [tutorials](/mlir-aie/tutorials/tutorial-7/). They are, however, not required to understand or use the Object FIFO API.

Below is an example of an Object FIFO of depth 3 with one producer tile A and three consumer tiles B, C and D:
```
A = tile(1, 1)
B = tile(1, 3)
C = tile(2, 3)
D = tile(3, 3)
of0 = object_fifo("objfifo0", A, [B, C, D], 3, T.memref(256, T.i32()))
```

The `depth` input of an Object FIFO can also be specified as an array of integers, which describe the number of objects that are available to each tile (the producer tile as well as each consumer tile) when accessing the Object FIFO. This functionality of the Object FIFO primitive exposes what is actually going on at the hardware level when the data movement is established for a broadcast. The object pool of the Object FIFO is not a single structure but rather composed of several pools of objects that are allocated in the memory module of each tile involved in the data movement. Specifying the `depth` as an array of integers allows the user full control to set the sizes of the pools on each individual tile.

The main advantage of being able to specify the individual depths comes during a situation like the one showcased in the example below, which we refer to as a broadcast with a <u>skip-connection</u>. In the example below two Object FIFOs are created: `of0` is a broadcast from producer tile A to consumer tiles B and C, while `of1` is a 1-to-1 data movement from producer tile B to consumer tile C. We refer to `of1` as a skip-connection because it is a dependency between the two consumer tiles of the same broadcast connection. Furthermore, we can see in the code that is executing on its core that C requires one object from both `of0` and `of1` before it can proceed with its execution. However, B also requires an object from `of0` before it can produce the data for `of1`.
```
A = tile(1, 3)
B = tile(2, 3)
C = tile(2, 4)
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

Below is an example of a link created between two FIFOs `of0` and `of1`, where tile B is the shared tile between them:
```
A = tile(1, 0)
B = tile(1, 1)
C = tile(1, 3)
of0 = object_fifo("objfifo0", A, B, 2, T.memref(256, T.i32()))
of1 = object_fifo("objfifo1", B, C, 2, T.memref(256, T.i32()))
object_fifo_link(of0, of1)
```
Similarly the link could've been created using the names of the FIFOs:
```
object_fifo_link("objfifo0", "objfifo1")
```

Depending on how many Object FIFOs are specified in `fifoIns` and `fifoOuts`, two different data patterns can be achieved: a Distribute or a Join. They are described in the two next subsections. Currently, it is not possible to do both patterns at once, i.e., if `fifoIns` is an array then `fifoOuts` can only be a single Object FIFO, and the to other way around.

A full design example that uses this features is available in Section 2e: [03_external_mem_to_core_L2](../section-2e/03_external_mem_to_core_L2/).

### Link & Distribute

By using the link with one input Object FIFO and multiple output Object FIFOs a user can describe a distribute pattern where parts of data in every object from the producer tile are distributed to each output FIFO. The `datatype` of the output FIFOs should be of a smaller size than the input one, and the sum of the sizes of the output FIFOs should equal to the size of the `datatype` of the input FIFO.

Currently, the Object FIFO lowering uses the order in which the output FIFOs are specified to know which part of the input object should go to each output FIFO. To achieve the distribute, the lowering will use one output port of the shared tile to establish a connection per output FIFO.

The example below shows three Object FIFOs: `of0` has a producer tile A and a consumer tile B, while `of1` and `of2` have B as their producer tile and C and D respectively as their consumer tiles. The link specifies that data from `of0` is distributed to `of1` and `of2`. In this link, B is the shared tile where the implicit data copy will take place via B's data movement hardware accelerators. We can also note how `of1` and `of2`'s datatypes are half of `of0`'s, which means that the first half of objects in `of0` will go to `of1` and the second half to `of2`, based on their order in the link.
```
A = tile(1, 0)
B = tile(1, 1)
C = tile(1, 3)
D = tile(2, 3)
of0 = object_fifo("objfifo0", A, B, 2, T.memref(256, T.i32()))
of1 = object_fifo("objfifo1", B, C, 2, T.memref(128, T.i32()))
of2 = object_fifo("objfifo2", B, D, 2, T.memref(128, T.i32()))
object_fifo_link(of0, [of1, of2])
```

A full design example that uses this features is available in Section 2e: [04_distribute_L2](../section-2e/04_distribute_L2/).

### Link & Join

The join pattern is the opposite of the distribute pattern in that the link will have multiple input Object FIFOs and a single output Object FIFO. With this pattern the user can combine the smaller inputs from multiple sources into a single bigger output data movement. The `datatype` of the input FIFOs should be of a smaller size than the output one, and the sum of the sizes of the input FIFOs should equal to the size of the `datatype` of the output FIFO.

Similarly, the order of the input Object FIFOs specifies which input object will make up which part of the larger objects of the output Object FIFO. To achieve the join, the lowering will use one input port of the shared tile to establish a connection per input FIFO.

The example below shows three Object FIFOs: `of2` has a producer tile B and a consumer tile A, while `of0` and `of1` have C and D respectively as their producer tiles and B as their consumer tile. The link specifies that data from `of0` and `of1` is joined into `of2`. In this link, B is the shared tile where the implicit data copy will take place via B's data movement hardware accelerators. We can also note how `of0` and `of1`'s datatypes are half of `of2`'s, which means that objects from `of0` will become the first half of objects in `of2` while objects in `of1` will become the second half, based on their order in the link.
```
A = tile(1, 0)
B = tile(1, 1)
C = tile(1, 3)
D = tile(2, 3)
of0 = object_fifo("objfifo0", C, B, 2, T.memref(128, T.i32()))
of1 = object_fifo("objfifo1", D, B, 2, T.memref(128, T.i32()))
of2 = object_fifo("objfifo2", B, A, 2, T.memref(256, T.i32()))
object_fifo_link([of0, of1], of2)
```

A full design example that uses this features is available in Section 2e: [05_join_L2](../section-2e/05_join_L2/).
