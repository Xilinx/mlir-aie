<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Implicit Copy Across Object FIFOs: Distribute & Join Patterns</ins>

### Object FIFO Implicit Copy

By design, an Object FIFO handles both the configuration of the data movement between producer and consumer(s), as well as the allocation of objects over the memory modules of Workers. Data consumed from one Object FIFO could be explicitly copied into another Object FIFO in the core code of a shared Worker between the two FIFOs. This is shown in the code snippet below where the Worker will copy data from `of_in` into `of_out`:
```python
of_in = ObjectFifo(line_type, name="in")
of_out = ObjectFifo(line_type, name="out")

def core_fn(of_in, of_out, copy_fn):
    elem_in = of_in.acquire(1)
    elem_out = of_out.acquire(1)
    copy_fn(elem_in, elem_out)
    of_in.release(1)
    of_out.release(1)

my_worker = Worker(core_fn, [of_in.cons(), of_out.prod(), copy_fn])
```

However, if the goal is to simply copy data from one Object FIFO to the other without modifying it, doing it in the manner described above results in allocating more objects than necessary, i.e., the data being copied to the second Object FIFO is already available in the first one. Additionally, Shim tiles and Mem tiles do not have a core on which the copy can be done explicitly.

Instead of an explicit copy, the Object FIFO API provides an implicit copy via the `forward()` function (defined in [objectfifo.py](../../../../python/iron/dataflow/objectfifo.py)), where an `ObjectFifoHandle` of type consumer is forwarded to the producer of a newly-constructed Object FIFO:
```python
def forward(
    self,
    placement: PlacementTile = AnyMemTile,
    obj_type: type[np.ndarray] | None = None,
    depth: int | None = None,
    name: str | None = None,
    dims_to_stream: list[Sequence[int]] | None = None,
    dims_from_stream: list[Sequence[int]] | None = None,
    plio: bool = False,
)
```
The `forward()` function creates a new Object FIFO to which the user can additionally specify the same inputs as to a regular Object FIFO. The `placement` tile is where the implicit copy will be done and is by default set to be a Mem tile.

With an implicit copy, the previous code can be written as:
```python
of_in = ObjectFifo(line_type, name="in")
of_out = of_in.cons().forward(obj_type=line_type, name="out")
```
Where a consumer `ObjectFifoHandle` to `of_in` is forwarded to `of_out` as its producer.

This functionality is also available at the explicitly placed level of abstraction. The Object FIFO API provides an implicit copy via an `object_fifo_link`, which can be initialized using its class constructor (defined in [aie.py](../../../../python/dialects/aie.py)):
```python
class object_fifo_link(ObjectFifoLinkOp):
    def __init__(
        self,
        fifoIns,
        fifoOuts,
        srcOffsets=[],
        dstOffsets=[],
    )
```
A link allows the user to specify a set of input Object FIFOs via the `fifoIns` input and a set of output ones via the `fifoOuts` input. Each Object FIFO may be specified either using its `name` or its Python object. Both inputs can be either a single Object FIFO or an array of them. It is required that there exists at least one shared tile between the consumer tiles of `fifoIns` and the producer tiles of `fifoOuts` for a link to be valid. This is because the implicit copy of data will be done using the Direct Memory Access channels (DMAs) of that tile.

Below is an example of a link created between two FIFOs `of_in` and `of_out`, where tile B is the shared tile between them:
```python
A = tile(1, 0)
B = tile(1, 1)
C = tile(1, 3)
of_in = object_fifo("in", A, B, 2, np.ndarray[(256,), np.dtype[np.int32]])
of_out = object_fifo("out", B, C, 2, np.ndarray[(256,), np.dtype[np.int32]])
object_fifo_link(of_in, of_out)
```

Depending on how many Object FIFOs are specified in `fifoIns` and `fifoOuts`, two different data patterns can be achieved: a Distribute or a Join. They are described in the two next subsections. Currently, it is not possible to do both patterns at once, i.e., if `fifoIns` is an array then `fifoOuts` can only be a single Object FIFO, and the other way around. At the highest level of abstraction these patterns are available as well.

A full design example that uses this features is available in Section 2f: [03_external_mem_to_core_L2](../../section-2f/03_external_mem_to_core_L2/).

### Distribute

Users can use the Object FIFO API to describe a distribute pattern where parts of data in every object from the producer are distributed to multiple consumers. This can be done with the `split()` function (defined in [objectfifo.py](../../../../python/iron/dataflow/objectfifo.py)):
```python
def split(
    self,
    offsets: list[int],
    placement: PlacementTile = AnyMemTile,
    depths: list[int] | None = None,
    obj_types: list[type[np.ndarray]] = None,
    names: list[str] | None = None,
    dims_to_stream: list[list[Sequence[int]]] | None = None,
    dims_from_stream: list[list[Sequence[int]]] | None = None,
    plio: bool = False,
) -> list[ObjectFifo]
```
The `split()` function creates multiple consumer Object FIFOs to which the user can additionally specify the same inputs as to a regular Object FIFO. The `offsets` are used to specify from which location in the producer Object FIFO's allocated memory to send data to each consumer Object FIFO.

Below you can see an example of an Object FIFO's consumer `ObjectFifoHandle` being split into `2` consumers, i.e. the number of cores that will be used in the design. The `split()` function is additionally given the offsets from which data will be sent to each consumer Object FIFO, the datatype of their objects, and their names.
```python
of0 = ObjectFifo(mem_tile_ty, name="objfifo0")
n_cores = 2
of_offsets = [
    (np.prod(np_ndarray_type_get_shape(mem_tile_ty)) // n_cores) * i
    for i in range(n_cores)
]
of0_fifos = of0.cons().split(
    of_offsets,
    obj_types=[aie_tile_ty] * n_cores,
    names=[f"objfifo{i}" for i in range(n_cores)],
)
```
Only `ObjectFifoHandle`s of type consumer can be split. The `obj_types` of the output FIFOs should be of a smaller size than the input one, and the sum of the sizes of the output FIFOs should equal the size of the `obj_type` of the input FIFO.

By using the link with one input Object FIFO and multiple output Object FIFOs, a user can describe a distribute pattern where parts of data in every object from the producer tile are distributed to each output FIFO. The `datatype` of the output FIFOs should be of a smaller size than the input one, and the sum of the sizes of the output FIFOs should equal the size of the `datatype` of the input FIFO.

Currently, the Object FIFO lowering uses the order in which the output FIFOs are specified in the `fifoOuts` to know which part of the input object should go to each output FIFO. To achieve the distribute, the lowering will use one output port of the shared tile to establish a connection per output FIFO, as in the figure below:

<img src="./../../../assets/Distribute.png" height="200">

The following code snippet describes the figure above. There are three Object FIFOs: `of0` has a producer tile A and a consumer tile B, while `of1` and `of2` have B as their producer tile and C and D respectively as their consumer tiles. The link specifies that data from `of0` is distributed to `of1` and `of2`. In this link, B is the shared tile where the implicit data copy will take place via B's DMAs. We can also note how `of1` and `of2`'s datatypes are half of `of0`'s, which means that the first half of objects in `of0` will go to `of1` and the second half to `of2`, based on their order in the link. This is explicitly set by specifying the `dstOffsets` option on the link.

```python
A = tile(1, 0)
B = tile(1, 1)
C = tile(1, 3)
D = tile(2, 3)
of0 = object_fifo("objfifo0", A, B, 2, np.ndarray[(256,), np.dtype[np.int32]])
of1 = object_fifo("objfifo1", B, C, 2, np.ndarray[(128,), np.dtype[np.int32]])
of2 = object_fifo("objfifo2", B, D, 2, np.ndarray[(128,), np.dtype[np.int32]])
object_fifo_link(of0, [of1, of2], [], [0, 128])
```

A full design example that uses this feature is available in Section 2f: [04_distribute_L2](../../section-2f/04_distribute_L2/).

### Join

The join pattern is the opposite of the distribute pattern where data received from multiple Object FIFOs is joined and sent to a single output Object FIFO. This can be done with the `join()` function (defined in [objectfifo.py](../../../../python/iron/dataflow/objectfifo.py)):
```python
def join(
    self,
    offsets: list[int],
    placement: PlacementTile = AnyMemTile,
    depths: list[int] | None = None,
    obj_types: list[type[np.ndarray]] = None,
    names: list[str] | None = None,
    dims_to_stream: list[list[Sequence[int] | None]] | None = None,
    dims_from_stream: list[list[Sequence[int] | None]] | None = None,
    plio: bool = False,
) -> list[ObjectFifo]
```
The `join()` function creates multiple producer Object FIFOs to which the user can additionally specify the same inputs as to a regular Object FIFO. The `offsets` are used to specify to which location in the producer Object FIFO's allocated memory to write data from each producer Object FIFO.

Below you can see an example of `2` Object FIFO's being created and joined in the producer `ObjectFifoHandle` of `of0`. The `join()` function is additionally given the offsets from which data will be written by each producer Object FIFO, the datatype of their objects, and their names.
```python
of0 = ObjectFifo(mem_tile_ty, name="objfifo0")
n_cores = 2
of_offsets = [
    (np.prod(np_ndarray_type_get_shape(mem_tile_ty)) // n_cores) * i
    for i in range(n_cores)
]
of0_fifos = of0.prod().join(
    of_offsets,
    obj_types=[aie_tile_ty] * n_cores,
    names=[f"objfifo{i}" for i in range(n_cores)],
)
```
Only `ObjectFifoHandle`s of type producer can be joined. The `obj_types` of the input FIFOs should be of a smaller size than the output one, and the sum of the sizes of the input FIFOs should be equal to the size of the `obj_type` of the output FIFO.

The join pattern with the link will have multiple input Object FIFOs and a single output Object FIFO. With this pattern the user can combine the smaller inputs from multiple sources into a single bigger output data movement. The `datatype` of the input FIFOs should be of a smaller size than the output one, and the sum of the sizes of the input FIFOs should be equal to the size of the `datatype` of the output FIFO.

Similarly, the order in `fifoIns` specifies which input object will make up which part of the larger objects of the output Object FIFO. To achieve the join, the lowering will use one input port of the shared tile to establish a connection per input FIFO, as in the figure below:

<img src="./../../../assets/Join.png" height="200">

The following code snippet describes the figure above. There are three Object FIFOs: `of0` has a producer tile B and a consumer tile A, while `of1` and `of2` have C and D respectively as their producer tiles and B as their consumer tile. The link specifies that data from `of1` and `of2` is joined into `of0`. In this link, B is the shared tile where the implicit data copy will take place via B's DMAs. We can also note how `of1` and `of2`'s datatypes are half of `of0`'s, which means that objects from `of1` will become the first half of objects in `of0` while objects in `of2` will become the second half, based on their order in the link.
```python
A = tile(1, 0)
B = tile(1, 1)
C = tile(1, 3)
D = tile(2, 3)
of0 = object_fifo("objfifo0", B, A, 2, np.ndarray[(256,), np.dtype[np.int32]])
of1 = object_fifo("objfifo1", C, B, 2, np.ndarray[(128,), np.dtype[np.int32]])
of2 = object_fifo("objfifo2", D, B, 2, np.ndarray[(128,), np.dtype[np.int32]])
object_fifo_link([of1, of2], of0, [0, 128], [])
```

A full design example that uses these features is available in Section 2f: [05_join_L2](../../section-2f/05_join_L2/).

-----
[[Prev](../02_Broadcast/)] [[Up](..)] [[Next](../04_Repeat/)]
