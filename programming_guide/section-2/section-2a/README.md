<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 2a - Introduction</ins>

### Initializing an Object FIFO

An Object FIFO represents the data movement connection between a point A and a point B. In the AIE array, these points are AIE tiles (see [Section 1 - Basic AI Engine building blocks](../../section-1/)). Under the hood, the data movement configuration for different types of tiles (Shim tiles, Mem tiles, and compute tile) is different, but there is no difference between them when using an Object FIFO. 

To initialize an Object FIFO, users can use the `object_fifo` class constructor (defined in [aie.py](../../../python/dialects/aie.py)):
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
We will now go over each of the inputs, what they represents and why they are required by the abstraction. We will first focus on the mandatory inputs and in a later section of the guide on the default valued ones (see Data Layout Transformations in [section-2c](../section-2c/README.md#data-layout-transformations)).

First of all, an Object FIFO has a unique `name`. It functions as an ordered buffer that has `depth`-many objects of specified `datatype`. Currently, all objects in an Object FIFO have to be of the same datatype. The datatype is a tensor-like attribute where the size of the tensor and the type of the individual elements are specified at the same time (i.e. `<16xi32>`). The `depth` can be either an integer or an array of integers. The latter is used to support a specific dependency that can arise when working with multiple Object FIFOs and it is further explained in the Key Object FIFO Patterns [section](../section-2b/README.md#broadcast).

An Object FIFO is created between a producer or source tile and a consumer or destination tile. Below, you can see an example of an Object FIFO created between producer tile A and consumer tile B:
```
A = tile(1, 2)
B = tile(1, 3)
of0 = object_fifo("objfifo0", A, B, 3, T.memref(256, T.i32()))
```
The created Object FIFO is stored in the `0f0` variable and is named `objfifo0`. It has a depth of `3` objects of datatype `<256xi32>`.

As you will see in the Key Object FIFO Patterns [section](../section-2b/README.md#key-object-fifo-patterns), an Object FIFO can have multiple consumer tiles, which describes a broadcast connection from the source tile to all of the consumer tiles. As such, the `consumerTiles` input can be either a single tile or an array of tiles. This is not the case for the `producerTile` input as currently the Object FIFO does not support multiple producers.

### Accessing the objects of an Object FIFO

An Object FIFO can be accessed by the processes running on the producer and consumer tiles registered to it. Before a process can have access to the objects it has to acquire them from the Object FIFO. This is because the Object FIFO is a synchronized communication primitive and two processes may not access the same object at the same time. Once a process has finished working with an object and has no further use for it, it should release it so that another process will be able to acquire and access it. The patterns in which a producer or a consumer process acquires and releases objects from an Object FIFO are called `access patterns`. We can specifically refer to the acquire and release patterns as well.

To acquire one or multiple objects users should use the acquire function of the `object_fifo` class:
```
def acquire(self, port, num_elem)
```
Based on the `num_elem` input representing the number of acquired elements, the acquire function will either directly return an object, or an array of objects that can be accessed in an array-like fashion.

To release one or multiple objects users should use the release function of the `object_fifo` class:
```
def release(self, port, num_elem)
```
A process may release one, some or all of the objects it has acquired. The release function will release objects from oldest to youngest in acquired order. If a process does not release all of the objects it has acquired, then the next time it acquires objects the oldest objects will be those that were not released. This functionality is intended to achieve the behaviour of a sliding window through the Object FIFO primitive. (TODO: add link to ref design or subsection) (TODO: merge PR to make the port optional) (TODO: make it clear that to access old unreleased objects, users should use the result of the new acquire)

Below you can see an example of two processes that are accessing the `of0` Object FIFO that we initialized in the previous section, one running on the producer tile and the other on the consumer tile. The producer process runs a loop of ten iterations and during each of them it acquires one object from `of0`, calls a `test_func` function on the acquired object, and releases the object. The consumer process only runs once and acquires two objects from `of0`. It then calls a `test_func2` function to which it gives as input each of the two objects it acquired, before releasing them both at the end.
```
A = tile(1, 2)
B = tile(1, 3)
of0 = object_fifo("objfifo0", A, B, 3, T.memref(256, T.i32()))

@core(A)
def core_body():
    for _ in range_(10):
        elem0 = of0.acquire(ObjectFifoPort.Produce, 1)
        call(test_func, [elem0])
        of0.release(ObjectFifoPort.Produce, 1)
        yield_([])

@core(B)
def core_body():
    elems = of0.acquire(ObjectFifoPort.Consume, 2)
    call(test_func2, [elems[0], elems[1]])
    of0.release(ObjectFifoPort.Consume, 2)
```

TODO: add description of initializing an OF with the same producer and consumer tile
