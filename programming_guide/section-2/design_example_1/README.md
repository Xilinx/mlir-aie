<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Tutorial ? - Data Movement : Introduction to the ObjectFifo Abstraction</ins>

This part of the tutorial introduces the `objectfifo` abstraction, which is used to setup communication between tiles without explicit configuration of the dataflow movement. The abstraction is then lowered via MLIR conversion passes onto the physical `mlir-aie` components.

This abstraction consists of several `AIE.objectfifo` operations which are gradually introduced in this tutorial and the following ones. The code in [aie.py](aie.py) starts by creative an `AIE.module` and an `AIE.device` targetting the `xcve2802` architecture. 

Firstly, an objectfifo is created between tiles S=(1,0) and N=(1,3) with the operation:
```
class object_fifo:
    def __init__(
        self,
        name,
        producerTile,
        consumerTiles,
        depth,
        datatype,
        dimensionsToStream=None,
        dimensionsFromStreamPerConsumer=None,
    )
```
An `objectfifo` describes both the data allocation and its movement. Its actors, which are separated into one or multiple consumerTiles and a producerTile, can synchronously access the pre-allocated objects of the specified datatype in a pool of the size specified by the depth input.

In this tutorial, tile `S` is the producer tile and tile `N` is the consumer tile and the `object_fifo` established between them has a depth of one object of datatype `memref<256xi32>`. This is shown in the diagram below (TODO).


TODO: describe dimensionsToStream & dimensionsFromStreamPerConsumer

To achieve deadlock-free communication, actors must acquire and release objects from the objectfifo. In this example, there is only one object to acquire. The operation, 
```
class object_fifo:
    def acquire(self, port, num_elem)
```
returns either one element or an array of the specified number of elements. Individual elements can then be accessed in an array-like fashion with the operation.

When an object is no longer required for computation, the actor which acquired it should release it with the operation:
```
class object_fifo:
    def release(self, port, num_elem)
``` 
such that other actors may acquire it in the future. The acquire and release operations both take an additional port attribute which can be either "Produce" or "Consume". The use of this attribute is ... (TODO).

## <ins>Tutorial 2 Lab </ins>

1. Read through the [/objectfifo_ver/aie.mlir](aie.mlir) design. In which tile and its local memory will the objectfifo lowering generate the buffer and its lock? <img src="../../images/answer1.jpg" title="On even rows tiles have local memories to their left, so the shared memory is that of tile (2,4). That is where the lowering will generate the shared buffer and lock." height=25>

2. Run `make` and `make -C aie.mlir.prj/sim` to compile the design with `aiecc.py` and then simulate that design with aiesimulator.

3. Change the locations of tiles (1,4) and (2,4) to (1,3) and (2,3). Navigate to the location of the [/objectfifo_ver/aie.mlir](aie.mlir) design and apply the objectfifo lowering on it. In which tile's local memory module did the lowering generate the buffer and its lock this time? <img src="../../images/answer1.jpg" title="On odd rows tiles have local memories to their right, so the shared memory is that of tile (1,3). That is where the lowering will generate the shared buffer and lock." height=25>

4. Increase the objectfifo size to 2. Apply the objectfifo lowering again. How many buffer/lock pairs are created and in which memory module? <img src="../../images/answer1.jpg" title="2 buffer/lock pairs are created in the shared memory of tile (1,3)." height=25>
