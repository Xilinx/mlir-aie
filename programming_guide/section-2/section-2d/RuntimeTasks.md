<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 2d - Runtime Data Movement</ins>

* [Section 2 - Data Movement (Object FIFOs)](../../section-2/)
    * [Section 2a - Introduction](../section-2a/)
    * [Section 2b - Key Object FIFO Patterns](../section-2b/)
    * [Section 2c - Data Layout Transformations](../section-2c/)
    * Section 2d - Runtime Data Movement
    * [Section 2e - Programming for multiple cores](../section-2e/)
    * [Section 2f - Practical Examples](../section-2f/)
    * [Section 2g - Data Movement Without Object FIFOs](../section-2g/)

-----

IRON describes runtime data movement with a `runtime_sequence` callable that you pass to the `Program`, along with its `arg_types`. The callable runs eagerly to describe the runtime data movement: it fills and drains Object FIFOs with data from/to external memory. The `Worker`s that perform the compute are passed to the `Program` (via `workers=[...]`) rather than started inside the sequence. All IRON constructs introduced in this section are available [here](../../../python/iron/runtime/).

To create a runtime sequence users can write:
```python
# To/from AIE-array runtime data movement
def runtime_sequence(a, b, c):
    # runtime tasks
    pass

Program(
    device,
    runtime_sequence,
    arg_types=[data_ty_a, data_ty_b, data_ty_c],
    workers=workers,
).resolve_program()
```
`Program(...)` takes the runtime sequence callable and a list of argument types via `arg_types`. The parameters of the callable (`a`, `b`, `c`) describe buffers that will be available on the host side; the body of the callable describes how those buffers are moved into the AIE-array.

#### **Runtime Tasks**

Runtime tasks are performed during runtime and they may be synchronous or asynchronous. The body of the sequence callable describes these tasks during the creation of the IRON design.

`Worker`s are not started from inside the sequence; instead they are passed to the `Program` via its `workers=[...]` argument. The code snippet below shows how one `Worker`, `my_worker`, is provided to the design:
```python
def runtime_sequence(a, b, c):
    pass

Program(
    device,
    runtime_sequence,
    arg_types=[data_ty, data_ty, data_ty],
    workers=[my_worker],
).resolve_program()
```

To run multiple `Worker`s, pass them all in the list:
```python
workers = []
# create and append Workers to the "workers" array

def runtime_sequence(a, b, c):
    pass

Program(
    device,
    runtime_sequence,
    arg_types=[data_ty, data_ty, data_ty],
    workers=workers,
).resolve_program()
```

The `fill()` operation is a method on a producer `ObjectFifoHandle`; it fills that fifo with data from a `source` runtime buffer. It is shown below and defined in [objectfifo.py](../../../python/iron/dataflow/objectfifo.py):
```python
def fill(
        self,
        source: RuntimeData,
        tap: TensorAccessPattern | None = None,
        offset=None,
        sizes=None,
        strides=None,
        group: TaskGroup | None = None,
        wait: bool = False,
        tile: Tile | None = None,
    )
```
When the `wait` input is set to `True` this operation will be waited upon, i.e., a token will be produced when the operation is finished that a controller is waiting on. A `tile` (Shim tile) can also be explicitly specified, otherwise the compiler will choose one. The `group` (a `TaskGroup`) is explained further in this section.

The code snippet below shows how data from a source runtime buffer `a_in` is sent to the producer `ObjectFifoHandle` of `of_in`. This data could then be read via a consumer `ObjectFifoHandle` of the same Object FIFO.
```python
def runtime_sequence(a_in, b, c):
    of_in.prod().fill(a_in)

# pass to Program(device, runtime_sequence, arg_types=[data_ty, data_ty, data_ty], ...)
```

The `drain()` operation is a method on a consumer `ObjectFifoHandle`; it drains that fifo of data and writes it to a `dest` runtime buffer. It is shown below and defined in [objectfifo.py](../../../python/iron/dataflow/objectfifo.py):
```python
def drain(
    self,
    dest: RuntimeData,
    tap: TensorAccessPattern | None = None,
    offset=None,
    sizes=None,
    strides=None,
    group: TaskGroup | None = None,
    wait: bool = False,
    tile: Tile | None = None,
)
```
When the `wait` input is set to `True` this operation will be waited upon, i.e., a token will be produced when the operation is finished that a controller is waiting on. A `tile` (Shim tile) can also be explicitly specified, otherwise the compiler will choose one. The `group` (a `TaskGroup`) is explained further in this section.

The code snippet below shows how data from a consumer `ObjectFifoHandle` of `of_out` is drained into a destination runtime buffer `c_out`. Data could be produced into `of_out` via its producer `ObjectFifoHandle`. Additionally, the `wait` input of the `drain()` task is set meaning that this task will be waited on until completion, i.e., until the `c_out` runtime buffer had received enough data as described by the `data_ty`.
```python
def runtime_sequence(a, b, c_out):
    of_out.cons().drain(c_out, wait=True)

# pass to Program(device, runtime_sequence, arg_types=[data_ty, data_ty, data_ty], ...)
```

#### **Inline Operations into the Runtime Sequence**

In some cases it may be desirable to insert a Python function that generates arbitrary MLIR operations into the runtime sequence. One such example is when users want to set runtime parameters, which will be loaded into the local memory modules of the Workers at runtime.

Because the `runtime_sequence` body runs eagerly inside the active MLIR context, you simply call such a function directly from the body — its operations are emitted in place. In the following code snippet, an array of `Buffer`s are created where each of the buffers will hold a runtime parameter of type `16xi32`. A [`Buffer`](../../../python/iron/buffer.py) is a memory region declared at the top-level of the IRON design that is available both to the `Worker`s and to the runtime for operations. When `use_write_rtp` is set, runtime parameter specific operations will be generated within the runtime sequence at lower-levels of compiler abstraction.
```python
# Runtime parameters
rtps = []
for i in range(4):
    rtps.append(
        Buffer(
            np.ndarray[(16,), np.dtype[np.int32]],
            name=f"rtp{i}",
            use_write_rtp=True,
        )
    )
```
The actual values of the runtime parameters will be loaded into each of the buffers at runtime:
```python
def runtime_sequence(a, b, c):

    # Set runtime parameters
    def set_rtps(*args):
        for rtp in args:
            rtp[0] = 50
            rtp[1] = 255
            rtp[2] = 0

    set_rtps(*rtps)

# pass to Program(device, runtime_sequence, arg_types=[data_ty, data_ty, data_ty], ...)
```
The propagation of data to these global buffers is not instantaneous and may lead to workers reading runtime parameters before they are available. To solve this, it is possible to instantiate `WorkerRuntimeBarrier`s defined in [worker.py](../../../python/iron/worker.py):
```python
class WorkerRuntimeBarrier:
    def __init__(self, initial_value: int = 0)
```

These barriers allow individual workers to synchronize with the runtime sequence at runtime:
```python
workerBarriers = []
for i in range(4):
    workerBarriers.append(WorkerRuntimeBarrier())

...

def core_fn(of_in, of_out, rtp, barrier):
    barrier.wait_for_value(1)
    runtime_parameter = rtp

...

def runtime_sequence(a, b, c):

    ...

    set_rtps(*rtps)

    for i in range(4):
        workerBarriers[i]._set_barrier_value(1)

# pass to Program(device, runtime_sequence, arg_types=[data_ty, data_ty, data_ty], ...)
```
Currently, a `WorkerRuntimeBarrier` may take any value between 0 and 63. This is due to the fact that these barriers leverage the lock mechansim of the architecture under-the-hood.

> **NOTE:**  Similar to the `Buffer` it is possible to create a single barrier and pass it as input to multiple workers. At lower stages of compiler abstraction this will result in a different lock being employed for each worker.

#### **Runtime Task Groups**

It may be desirable to reconfigure a runtime sequence and reuse some of the resources from a previous configuration, especially given that some of these resources, like the BDs in a DMA task queue, are limited.

To facilitate this reconfiguration step, IRON introduces `TaskGroup`s which are created by constructing a `TaskGroup()` inside the sequence callable, as defined in [taskgroup.py](../../../python/iron/runtime/taskgroup.py).

`fill`/`drain` tasks can be added to a task group by specifying their `group` input. Tasks in the same group will be appended to the runtime sequence and executed in order. The group's `finish()` method is used to mark the end of a task group. This call waits for tasks in the group annotated with `wait=True` to complete, and then frees _all_ resources used by the task.
If a `TaskGroup` is not explicitly specified for DMA tasks defined in a runtime sequence, then a single default task group (finished at end-of-sequence) is used.

> **NOTE:**  A call to  `tg.finish()` blocks the runtime sequence until all of the group's tasks annotated with `wait=True`  ("awaited tasks") have completed. After waiting, all resources of the task group -- including those _not_ annotated with `wait=True` ("unawaited tasks") -- will be freed and reused for subsequent tasks. 
> 
> To avoid race conditions, any unawaited tasks in the group should form a dependency of an awaited task.
> It is only safe to remove a `wait=True` if you can reason that another, awaited task in the same group can only complete if the awaited task also completed.
> For example, you may choose to set `wait=False` on an input fill if you can guarantee that a later (awaited) output drain depends on the input and completes only if the input fill completed as well.
>
> If you suspect a race condition, the safest (but possibly slower) solution is to annotated _all_ tasks (including inputs) with `wait=True`.

The runtime sequence in the code snippet below has two task groups. We can observe that the creation of the second task group happens at the end of execution of the first task group. (The `Worker`s are passed to the `Program` separately.)
```python
def runtime_sequence(a_in, b, c_out):
    tg = TaskGroup() # start first task group
    for _ in [0, 1]:
        of_in.prod().fill(a_in, group=tg)
        of_out.cons().drain(c_out, group=tg, wait=True)
        tg.finish()
        tg = TaskGroup() # start second task group
    tg.finish()

Program(
    device,
    runtime_sequence,
    arg_types=[data_ty, data_ty, data_ty],
    workers=workers,
).resolve_program()
```

-----
[[Up](./README.md)]
