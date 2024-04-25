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

* [Section 2 - Data Movement (Object FIFOs)](../../section-2/)
    * Section 2a - Introduction
    * [Section 2b - Key Object FIFO Patterns](../section-2b/)
    * [Section 2c - Data Layout Transformations](../section-2c/)
    * [Section 2d - Programming for multiple cores](../section-2d/)
    * [Section 2e - Practical Examples](../section-2e/)
    * [Section 2f - Data Movement Without Object FIFOs](../section-2f/)
    * [Section 2g - Runtime Data Movement](../section-2g/)

-----

### Initializing an Object FIFO

An Object FIFO represents the data movement connection between a point A and a point B. In the AIE array, these points are AIE tiles (see [Section 1 - Basic AI Engine building blocks](../../section-1/)). Under the hood, the data movement configuration for different types of tiles (Shim tiles, Mem tiles, and compute tile) is different, but there is no difference between them when using an Object FIFO. 

To initialize an Object FIFO, users can use the `object_fifo` class constructor (defined in [aie.py](../../../python/dialects/aie.py)):
```python
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
We will now go over each of the inputs, what they represent and why they are required by the abstraction. We will first focus on the mandatory inputs and in a later section of the guide on the default-valued ones (see Data Layout Transformations in [section-2c](../section-2c/README.md#data-layout-transformations)).

First of all, an Object FIFO has a unique `name` which is required for the lowering steps. The Object FIFO functions as an ordered buffer that has  a count of `depth` objects of specified `datatype`. Currently, all objects in an Object FIFO have to be of the same datatype. The `datatype` is a tensor-like attribute where the size of the tensor and the type of the individual elements are specified at the same time (i.e. `<16xi32>`). The `depth` can be either an integer or an array of integers. The latter is explained further down in this section.

An Object FIFO is created between a producer, or source tile, and a consumer, or destination tile. The tiles are where producer and consumer processes accessing the Object FIFO will be executed. These processes are also refered to as the "actors" of the Object FIFO, based on dataflow theory terminology. Below, you can see an example of an Object FIFO created between producer tile A and consumer tile B:
```python
A = tile(1, 3)
B = tile(2, 4)
of0 = object_fifo("objfifo0", A, B, 3, T.memref(256, T.i32()))
```
The created Object FIFO is stored in the `of0` variable and is named `objfifo0`. It has a depth of `3` objects of datatype `<256xi32>`. The figure below represents a logical view of `of0` where no assumptions are made about where the tiles and the Object FIFO resources are placed:

<img src="./../../assets/ObjectFifo.svg" height="200">

As you will see in the ["Key Object FIFO Patterns" section](../section-2b/README.md#key-object-fifo-patterns), an Object FIFO can have multiple consumer tiles, which describes a broadcast connection from the source tile to all of the consumer tiles. As such, the `consumerTiles` input can be either a single tile or an array of tiles. This is not the case for the `producerTile` input, as currently the Object FIFO does not support multiple producers.

### Accessing the objects of an Object FIFO

An Object FIFO can be accessed by the processes running on the producer and consumer tiles registered to it. Before a process can have access to the objects, it has to acquire them from the Object FIFO. This is because the Object FIFO is a synchronized communication primitive that leverages the synchronization mechanism available in the target hardware architecture to ensure that two processes cannot access the same object at the same time. Once a process has finished working with an object and has no further use for it, it must release it so that another process will be able to acquire and access it. The patterns in which a producer or a consumer process acquires and releases objects from an Object FIFO are called "access patterns". We can specifically refer to the acquire and release patterns as well.

To acquire one or multiple objects users should use the acquire function of the `object_fifo` class:
```python
def acquire(self, port, num_elem)
```
Based on the `num_elem` input representing the number of acquired elements, the acquire function will either directly return an object, or an array of objects. 

The Object FIFO is an ordered primitive and the API keeps track for each process which object is the next one that they will have access to when acquiring, based on how many they have already acquired and released. Specifically, the first time a process acquires an object it will have access to the first object of the Object FIFO, and after releasing it and acquiring a new one, it'll have access to the second object, and so on until the last object, after which the order starts from the first one again. When acquiring multiple objects and accessing them in the returned array, the object at index 0 will always be the <u>oldest</u> object that process has access to, which may not be the first object in the pool of that Object FIFO.

To release one or multiple objects users should use the release function of the `object_fifo` class:
```python
def release(self, port, num_elem)
```
A process may release one, some or all of the objects it has acquired. The release function will release objects from oldest to youngest in acquired order. If a process does not release all of the objects it has acquired, then the next time it acquires objects the oldest objects will be those that were not released. This functionality is intended to achieve the behaviour of a sliding window through the Object FIFO primitive. This is described further in the ["Key Object FIFO Patterns" section](../section-2b/01_Reuse/README.md#object-fifo-reuse-pattern). 

When acquiring the objects of an Object FIFO using the acquire function it is important to note that any <u>unreleased objects from a previous acquire</u> will also be returned by the <u>most recent</u> acquire call. Unreleased objects will not be reacquired in the sense that the synchronization mechanism used under the hood has already been set in place such that the process already has the sole access rights to the unreleased objects from the previous acquire. As such, two acquire calls back-to-back without a release call in-between will result in the same objects being returned by both acquire calls. This decision was made to facilitate the understanding of releasing objects between calls to the acquire function as well as to ensure a proper lowering through the Object FIFO primitive. A code example of this behaviour is available in the ["Key Object FIFO Patterns" section](../section-2b/01_Reuse/README.md#object-fifo-reuse-pattern).

The `port` input of both the acquire and the release functions represents whether that process is a producer or a consumer process and it is an important indication for the Object FIFO lowering to properly leverage the underlying synchronization mechanism. Its value may be either `ObjectFifoPort.Produce` or `ObjectFifoPort.Consume`. However, an important thing to note is that the terms producer and consumers are used mainly as a means to provide a logical reference for a human user to keep track of what process is at what end of the data movement, but it <u>does not restrict the behaviour of that process</u>, i.e., a producer process may simply access an object to read it and is not required to modify it.

Below you can see an example of two processes that are <u>iterating over the objects of the Object FIFO</u> `of0` that we initialized in the previous section, one running on the producer tile and the other on the consumer tile. To do this, the producer process runs a loop of three iterations, equal to the depth of `of0`, and during each iteration it acquires one object from `of0`, calls a `test_func` function on the acquired object, and releases the object. The consumer process only runs once and acquires all three objects from `of0` at once and stores them in the `elems` array, from which it can <u>access each object individually in any order</u>. It then calls a `test_func2` function three times and in each call it gives as input one of the objects it acquired, before releasing all three objects at the end.
```python
A = tile(1, 3)
B = tile(2, 4)
of0 = object_fifo("objfifo0", A, B, 3, T.memref(256, T.i32()))

@core(A)
def core_body():
    for _ in range_(3):
        elem0 = of0.acquire(ObjectFifoPort.Produce, 1)
        call(test_func, [elem0])
        of0.release(ObjectFifoPort.Produce, 1)
        yield_([])

@core(B)
def core_body():
    elems = of0.acquire(ObjectFifoPort.Consume, 3)
    call(test_func2, [elems[0]])
    call(test_func2, [elems[1]])
    call(test_func2, [elems[2]])
    of0.release(ObjectFifoPort.Consume, 3)
```

The figure below illustrates this code: Each of the 4 drawings represents the state of the system during one iteration of execution. In the first three iterations, the producer process on tile A, drawn in blue, progressively acquires the elements of `of0` one by one. Once the third element has been released in the fourth iteration, the consumer process on tile B, drawn in green, is able to acquire all three objects at once.

<img src="./../../assets/AcquireRelease.png" height="400">

Examples of designs that use these features are available in Section 2e: [01_single_double_buffer](../section-2e/01_single_double_buffer/) and [02_external_mem_to_core](../section-2e/02_external_mem_to_core/).

### Object FIFOs with the same producer / consumer

An Object FIFO can be created with the same tile as both its producer and consumer tile. This is mostly done to ensure proper synchronization within the process itself, as opposed to synchronization across multiple processes running on different tiles, as we have seen in examples up until this point. Composing two kernels with access to a shared buffer is an application that leverages this property of the Object FIFO, as showcased in the code snippet below, where `test_func` and  `test_func2` are composed using `of0`:
```python
A = tile(1, 3)
of0 = object_fifo("objfifo0", A, A, 3, T.memref(256, T.i32()))

@core(A)
def core_body():
    for _ in range_(3):
        elem0 = of0.acquire(ObjectFifoPort.Produce, 1)
        call(test_func, [elem0])
        of0.release(ObjectFifoPort.Produce, 1)

        elem1 = of0.acquire(ObjectFifoPort.Consume, 1)
        call(test_func2, [elem1])
        of0.release(ObjectFifoPort.Consume, 1)
        yield_([])
```

### Specifying the Object FIFO Depth as an Array

As was mentioned in the beginning of this section, the AIE architecture is a spatial architecture that requires explicit data movement. As such, while the Object FIFO's conceptual design is that of an ordered buffer between two or more AIE tiles, in reality its conceptual depth is spread out over multiple resource pools that may be located at different levels of the memory hierarchy and on different tiles.

A more in-depth, yet still abstract, view of the Object FIFO's depth is that the producer and each consumer have their own working resource pool available in their local memory modules which they can use to send and receive data in relation to the data movement described by the Object FIFO. The Object FIFO primitive and its lowering typically allocate the depth of each of these pools such that the resulting behaviour matches that of the conceptual depth.

The user does however have the possibility to manually choose the depth of these pools. This feature is available because, while the Object FIFO primitive tries to offer a unified representation of the data movement across the AIE array, it also aims to provide performance programmers with the tools to control it more finely.

For example, in the code snippet below `of0` describes the data movement between producer A and consumer B:
```python
A = tile(1, 3)
B = tile(2, 4)
of0 = object_fifo("objfifo0", A, B, 3, T.memref(256, T.i32()))
```
The conceptual depth of the Object FIFO is `3`. The reasoning behind this choice of depth can be understood by looking at the acquire and release patterns of the two actors:
```python
@core(A)
def core_body():
    for _ in range_(9):
        elem0 = of0.acquire(ObjectFifoPort.Produce, 1)
        call(produce_func, [elem0])
        of0.release(ObjectFifoPort.Produce, 1)
        yield_([])

@core(B)
def core_body():
    for _ in range_(9):
        elems = of0.acquire(ObjectFifoPort.Consume, 2)
        call(consume_func, [elems[0], elems[1]])
        of0.release(ObjectFifoPort.Consume, 2)
        yield_([])
```
Each iteration:
* producer A acquires one object to produce into, calls the kernel function `produce_func` to store new data in it for B to consume, and releases the object,
* consumer B acquires two objects to consume, reads the data and applies kernel function `consume_func`, then releases both objects.

A conceptual depth of `2` would have sufficed for this system to function without deadlocking. However, with a depth of `3`, A and B can execute concurrently, i.e., while B consumes two objects and applies the kernel function, A has one object available into which it can produce at the same time.

The equivalent of this conceptual depth of `3` using an array of depths would be:
```python
of0 = object_fifo("objfifo0", A, B, [1, 2], T.memref(256, T.i32()))
```
where `1` is the number of resources available locally to producer A and `2` is the number available to consumer B.

> **NOTE:**  For a correct lowering, this feature should be used in situations where the producers and consumers of the Object FIFO are running on different tiles.

The feature of specifying the depths of the resource pools for different actors of the Object FIFO is used to support a specific dependency that can arise when working with multiple Object FIFOs and it is further explained in the ["Key Object FIFO Patterns" section](../section-2b/02_Broadcast/README.md#object-fifo-broadcast-pattern).

### Advanced Topic: Data Movement Accelerators

**The following topic is not required to understand the rest of this guide.**

This part of the guide introduces a few lower level concepts in the AIE hardware and takes a closer look at the individual resource pools on each tile and the reasoning behind their depths.

Every tile in the AIE array has its own dedicated Data Movement Accelerator (or "DMA"). The DMAs are responsible for moving data from the tile's memory module to the AXI stream interconnect, or from the stream to the memory module. In the case of compute tiles, both the compute core and the tile's DMA are able to access the tile's memory module. Because of this, there is a need for a **synchronization mechanism** that will allow the compute core and the DMA to signal to each other when data is available for the other party to read or write in order to avoid data corruption. This is very similar to the concept of the Object FIFO where producers and consumers must first acquire objects before they can access them, and release them when they are done so they may be acquired by the other party.

The figure below showcases a high-level view of a compute tile, where the compute core and the DMA are both reading and writing data to a location `buff` in the local memory module:

<img src="./../../assets/ComputeTile.png" height="250">

The intent of this high-level view showcases that the DMA is able to interact with memory buffers while the compute core is simultaneously accessing them. The DMA can send data from a buffer onto the AXI stream, and receive data from the stream to write into a buffer which the core is processing. Because this concurrency can lead to data races, a ping-pong buffer (also called double buffer) is often used instead of a single buffer. This is showcased in the figure below where the `buff` has been extended to a `buff_ping` and `buff_pong`:

<img src="./../../assets/ComputeTile_2.png" height="250">

> **NOTE:**  It is possible to directly configure the DMAs without the use of the Object FIFO primitive to setup data movement between tiles. This is described in [Section 2f](../section-2f/README.md).

## <u>Exercises</u>
1. In the previous [subsection](./README.md/#specifying-the-object-fifo-depth-as-an-array) it was explained that the conceptual depth of `3` for `of0` could be represented as an array of depths `[1, 2]`. With the advanced knowledge on the topic of DMAs, do you think those depths suffice for the compute cores on tiles A and B to run concurrently with their local DMAs? <img src="../../../mlir_tutorials/images/answer1.jpg" title="No. In the case of producer A, only a single object was allocated for the design which results in the compute core and the DMA having to wait while the other party respectively computes or moves the data. This is similar for consumer B, where the compute core acquires both allocated objects, leaving none for the DMA to interact with." height=25>

1. How would you update the depths? <img src="../../../mlir_tutorials/images/answer1.jpg" title="Producer A requires a ping-pong buffer to function concurrently with its DMA. Similarly, consumer B requires two additional objects that the DMA can write new data into while B computes. The updated depths are [2, 4]." height=25>

-----
[[Up](..)] [[Next - Section 2b](../section-2b/)]
