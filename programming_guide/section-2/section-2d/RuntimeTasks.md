<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 2g - Runtime Data Movement</ins>

* [Section 2 - Data Movement (Object FIFOs)](../../section-2/)
    * [Section 2a - Introduction](../section-2a/)
    * [Section 2b - Key Object FIFO Patterns](../section-2b/)
    * [Section 2c - Data Layout Transformations](../section-2c/)
    * [Section 2d - Programming for multiple cores](../section-2d/)
    * [Section 2e - Practical Examples](../section-2e/)
    * [Section 2f - Data Movement Without Object FIFOs](../section-2f/)
    * Section 2g - Runtime Data Movement

-----

IRON proposes a `Runtime` class with a `sequence()` function which can be programmed with `RuntimeTasks` that will launch Workers, fill and drain Object FIFOs with data from/to external memory. All IRON constructs introduced in this section are available [here](../../../python/iron/runtime/).

To create a Runtime sequence users can write:
```python
# To/from AIE-array runtime data movement
rt = Runtime()
with rt.sequence(data_ty_a, data_ty_b, data_ty_c) as (a, b, c):
    # runtime tasks
```
The arguments to this function describe buffers that will be available on the host side; the body of the function describes how those buffers are moved into the AIE-array.

#### **Runtime Tasks**

Runtime tasks are performed during runtime and they may be synchronous or asynchronous. Tasks can be added to the runtime sequence during the creation of the IRON design, and they can also be queued during runtime.

The `start()` operation is used to start one or multiple Workers that were declared in the IRON design. It is shown below and defined in [runtime.py](../../../python/iron/runtime/runtime.py):
```python
def start(self, *args: Worker)
```
If more than one Worker is given as input, they will be started in order.

The code snippet below shows how one `my_worker` Worker is started:
```python
rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (_, _, _):
    rt.start(my_worker)
```

To start multiple Workers with a single use of this operation users can write:
```python
workers = []
# create and append Workers to the "workers" array

rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (_, _, _):
    rt.start(*workers)
```

The `fill()` operation is used to fill an `in_fifo` `ObjectFifoHandle` of type producer with data from a `source` runtime buffer. It is shown below and defined in [runtime.py](../../../python/iron/runtime/runtime.py):
```python
def fill(
        self,
        in_fifo: ObjectFifoHandle,
        source: RuntimeData,
        tap: TensorAccessPattern | None = None,
        task_group: RuntimeTaskGroup | None = None,
        wait: bool = False,
        placement: PlacementTile = AnyShimTile,
    )
```
When the `wait` input is set to `True` this operation will be waited upon, i.e., a token will be produced when the operation is finished that a controller is waiting on. A `placement` Shim tile can also be explicitly specified, otherwise the compiler will choose one based on the placement algorithm. The `task_group` is explained further in this section.

The code snippet below shows how data from a source runtime buffer `a_in` is sent to the producer `ObjectFifoHandle` of `of_in`. This data could then be read via a consumer `ObjectFifoHandle` of the same Object FIFO.
```python
rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (a_in, _, _):
    rt.fill(of_in.prod(), a_in)
```

The `drain()` operation is used to fill an `out_fifo` `ObjectFifoHandle` of type consumer of data and write that data to a `dest` runtime buffer. It is shown below and defined in [runtime.py](../../../python/iron/runtime/runtime.py):
```python
def drain(
    self,
    out_fifo: ObjectFifoHandle,
    dest: RuntimeData,
    tap: TensorAccessPattern | None = None,
    task_group: RuntimeTaskGroup | None = None,
    wait: bool = False,
    placement: PlacementTile = AnyShimTile,
)
```
When the `wait` input is set to `True` this operation will be waited upon, i.e., a token will be produced when the operation is finished that a controller is waiting on. A `placement` Shim tile can also be explicitly specified, otherwise the compiler will choose one based on the placement algorithm. The `task_group` is explained further in this section.

The code snippet below shows how data from a consumer `ObjectFifoHandle` of `of_out` is drained into a destination runtime buffer `c_out`. Data could be produced into `of_out` via its producer `ObjectFifoHandle`. Additionally, the `wait` input of the `drain()` task is set meaning that this task will be waited on until completion, i.e., until the `c_out` runtime buffer had received enough data as described by the `data_ty`.
```python
rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (_, _, c_out):
    rt.drain(of_out.cons(), c_out, wait=True)
```

#### **Inline Operations into a Runtime Sequence**

In some cases it may be desirable to insert a Python function that generates arbitrary MLIR operations into the runtime sequence. One such example is when users want to set runtime parameters, which will be loaded into the local memory modules of the Workers at runtime, or to setup tracing in the design.

To inline operations into the runtime sequence, users can use the `inline_ops()` operation. It is shown below and defined in [runtime.py](../../../python/iron/runtime/runtime.py):
```python
def inline_ops(self, inline_func: Callable, inline_args: list)
```
The `inline_func` is the function to execute within an MLIR context and the `inline_args` are state the function needs to execute.

In the following code snippet, an array of `GlobalBuffers` is created where each of the buffers will hold a runtime parameter of type `16xi32`. A [`GlobalBuffer`](../../../python/iron/globalbuffer.py) is a memory region declared at the top-level of the IRON design that is available both to the Workers and to the runtime for operations. When `use_write_rtp` is set, runtime parameter specific operations will be generated within the runtime sequence at lower-levels of compiler abstraction.
```python
# Runtime parameters
rtps = []
for i in range(4):
    rtps.append(
        GlobalBuffer(
            np.ndarray[(16,), np.dtype[np.int32]],
            name=f"rtp{i}",
            use_write_rtp=True,
        )
    )
```
The actual values of the runtime parameters will be loaded into each of the buffers at runtime:
```python
rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (_, _, _):

    # Set runtime parameters
    def set_rtps(*args):
        for rtp in args:
            rtp[0] = 50
            rtp[1] = 255
            rtp[2] = 0

    rt.inline_ops(set_rtps, rtps)
```

#### **Runtime Task Groups**

It may be desirable to reconfigure the runtime sequence and reuse some of the resources from a previous configuration, especially given that some of these resources, like the BDs in a DMA task queue, are limited.

To facilitate this reconfiguration step, IRON introduces `RuntimeTaskGroup`s which can be created using the `task_group()` function as defined in [runtime.py](../../../python/iron/runtime/runtime.py).

`RuntimeTask`s can be added to a task group by specifying their `task_group` input. Tasks in the same group will be appended to the runtime sequence and executed in order. The `finish_task_group()` operation is used to mark the end of a task group, i.e., after this operation all of the tasks in the group will be waited on for completion after which they will be freed at the same time and the runtime sequence will be reconfigured by the next task group.

> **NOTE:**  Because of their ability to wait on runtime tasks until completion and free all the resources at the same time, task groups are well-placed to handle the asynchronous nature of runtime data movement tasks.

The runtime sequence in the code snippet below has two task groups. We can observe that the creation of the second task group happens at the end of execution of the first task group.
```python
rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (a_in, _, c_out):
    rt.start(*workers)

    tg = rt.task_group() # start first task group
    for groups in [0, 1]:
        rt.fill(of_in.prod(), a_in, task_group=tg)
        rt.drain(of_out.cons(), c_out, task_group=tg, wait=True)
        rt.finish_task_group(tg)
        tg = rt.task_group() # start second task group
    rt.finish_task_group(tg)
```

-----
[[Up](./README.md)]
