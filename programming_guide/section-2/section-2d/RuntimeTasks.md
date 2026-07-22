<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Section 2d - Runtime Data Movement

* [Section 2 - Data Movement (ObjectFifos)](../../section-2/README.md)
    * [Section 2a - Introduction](../section-2a/README.md)
    * [Section 2b - Key ObjectFifo Patterns](../section-2b/README.md)
    * [Section 2c - Data Layout Transformations](../section-2c/README.md)
    * Section 2d - Runtime Data Movement
    * [Section 2e - Programming for multiple cores](../section-2e/README.md)
    * [Section 2f - Practical Examples](../section-2f/README.md)
    * [Section 2g - Data Movement Without ObjectFifos](../section-2g/README.md)

-----

IRON provides a `Runtime` class whose *sequence body* — a plain Python function — describes how host-side buffers are moved into and out of the AIE-array while one or more `Worker`s run. All IRON constructs introduced in this section are available [here](../../../python/iron/runtime/).

A `Runtime` is created from its sequence body and the inputs that body takes, mirroring how a `Worker` is created from a `core_fn` and its `fn_args`:
```python
# To/from AIE-array runtime data movement
def sequence(a, b, c):
    # runtime tasks
    ...

rt = Runtime(sequence, [data_ty_a, data_ty_b, data_ty_c])
```
The `inputs` list (`[data_ty_a, ...]`) describes the host-side buffers; the body receives one argument per input and describes how those buffers move into the AIE-array. The body runs later, inside the lowered `aie.runtime_sequence`, so it executes with a live MLIR context — meaning native `range_`/`if_` control flow and data-movement verbs work directly inside it.

#### **Passing ObjectFifos to the body: `fn_args`**

The body moves data by calling `fill`/`drain` on `ObjectFifoHandle`s. Those handles are passed to the `Runtime` via `fn_args` — exactly like a `Worker`'s `fn_args` — and are received as trailing parameters of the body, after the inputs:
```python
def sequence(a, b, c, in_h, out_h):
    in_h.fill(a)
    out_h.drain(c, wait=True)

rt = Runtime(
    sequence,
    [data_ty_a, data_ty_b, data_ty_c],
    fn_args=[of_in.prod(), of_out.cons()],
)
```
Passing the handles through `fn_args` (rather than capturing them by closure) lets the `Runtime` bind each ObjectFifo's shim endpoint up front, so the design resolves cleanly regardless of where the body appears.

#### **Runtime Tasks**

`Runtime` tasks are the data-movement operations performed at runtime; they may be synchronous or asynchronous.

`Worker`s are **not** started from inside the sequence — they are handed to the `Program` directly:
```python
Program(device, rt, workers=[my_worker]).resolve_program()
```

To run multiple `Worker`s, pass them all:
```python
workers = []
# create and append Workers to the "workers" array

Program(device, rt, workers=workers).resolve_program()
```

The `fill()` operation is a method on a *producer* `ObjectFifoHandle` that fills it with data from a `source` runtime buffer. It is defined in [objectfifo.py](../../../python/iron/dataflow/objectfifo.py):
```python
def fill(
        self,
        source,
        tap=None,
        wait: bool = False,
        group=None,
        ...
    )
```
When the `wait` input is set to `True` this operation will be waited upon, i.e., a token will be produced when the operation is finished that a controller is waiting on. The `group` is explained further in this section. The Shim tile is chosen by the compiler, or pinned via `prod(tile=...)` on the handle (see below).

The code snippet below shows how data from a source runtime buffer `a_in` is sent to the producer `ObjectFifoHandle` of `of_in`. This data could then be read via a consumer `ObjectFifoHandle` of the same ObjectFifo.
```python
def sequence(a_in, in_h):
    in_h.fill(a_in)

rt = Runtime(sequence, [data_ty], fn_args=[of_in.prod()])
```

The `drain()` operation is a method on a *consumer* `ObjectFifoHandle` that reads its data and writes it to a `dest` runtime buffer. It is defined in [objectfifo.py](../../../python/iron/dataflow/objectfifo.py):
```python
def drain(
    self,
    dest,
    tap=None,
    wait: bool = False,
    group=None,
    ...
)
```
When the `wait` input is set to `True` this operation will be waited upon, i.e., a token will be produced when the operation is finished that a controller is waiting on. The `group` is explained further in this section.

The code snippet below shows how data from a consumer `ObjectFifoHandle` of `of_out` is drained into a destination runtime buffer `c_out`. Data could be produced into `of_out` via its producer `ObjectFifoHandle`. Additionally, the `wait` input of the `drain()` task is set meaning that this task will be waited on until completion, i.e., until the `c_out` runtime buffer had received enough data as described by the `data_ty`.
```python
def sequence(c_out, out_h):
    out_h.drain(c_out, wait=True)

rt = Runtime(sequence, [data_ty], fn_args=[of_out.cons()])
```

To pin the Shim tile a handle's host-side DMA uses, pass `tile=` to `prod()`/`cons()` where the handle is created (not to `fill`/`drain`):
```python
rt = Runtime(sequence, [data_ty], fn_args=[of_in.prod(tile=Tile(0, 0))])
```

The `fill()`/`drain()` methods return a `Task` handle. For the common case you can ignore it, but it enables software-pipelined data movement: pass a `Task` as a `range_` `iter_arg` to carry an in-flight transfer across loop iterations, and call `.free()` / `.await_()` on it to manage its lifetime by hand (see [dmataskhandle.py](../../../python/iron/runtime/dmataskhandle.py)).

#### **Setting Runtime Parameters in the Body**

Because the sequence body runs inside a live MLIR context, you can write operations directly in it — there is no separate escape hatch. A common example is setting runtime parameters, which are loaded into the local memory modules of the Workers at runtime.

In the following code snippet, an array of `Buffer`s are created where each of the buffers will hold a runtime parameter of type `16xi32`. A [`Buffer`](../../../python/iron/buffer.py) is a memory region declared at the top-level of the IRON design that is available both to the `Worker`s and to the runtime for operations. When `use_write_rtp` is set, runtime parameter specific operations will be generated within the `Runtime`'s `sequence` at lower-levels of compiler abstraction.
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
The values are written to each buffer by indexing it directly in the body. The body runs after the `Worker`s (and the `Buffer`s they own) are placed, so these writes resolve correctly:
```python
def sequence(a, b, c):
    # Set runtime parameters
    for rtp in rtps:
        rtp[0] = 50
        rtp[1] = 255
        rtp[2] = 0

rt = Runtime(sequence, [data_ty, data_ty, data_ty])
```
The propagation of data to these global buffers is not instantaneous and may lead to workers reading runtime parameters before they are available. To solve this, it is possible to instantiate `WorkerRuntimeBarrier`s defined in [worker.py](../../../python/iron/worker.py):
```python
class WorkerRuntimeBarrier:
    def __init__(self, initial_value: int = 0)
```

These barriers allow individual workers to synchronize with the `Runtime`'s `sequence` at runtime. A barrier is set from the body with `barrier.set(value)`:
```python
workerBarriers = []
for i in range(4):
    workerBarriers.append(WorkerRuntimeBarrier())

...

def core_fn(of_in, of_out, rtp, barrier):
    barrier.wait_for_value(1)
    runtime_parameter = rtp

...

def sequence(a, b, c):
    # Set runtime parameters, then release the barriers.
    for rtp in rtps:
        rtp[0] = 50

    for i in range(4):
        workerBarriers[i].set(1)

rt = Runtime(sequence, [data_ty, data_ty, data_ty])
```
Currently, a `WorkerRuntimeBarrier` may take any value between 0 and 63. This is due to the fact that these barriers leverage the lock mechansim of the architecture under-the-hood.

> **NOTE:**  Similar to the `Buffer` it is possible to create a single barrier and pass it as input to multiple workers. At lower stages of compiler abstraction this will result in a different lock being employed for each worker.

#### **Runtime Task Groups**

It may be desirable to reconfigure a `Runtime`'s `sequence` and reuse some of the resources from a previous configuration, especially given that some of these resources, like the BDs in a DMA task queue, are limited.

To facilitate this reconfiguration step, IRON introduces `TaskGroup`s, created with the `TaskGroup()` constructor as defined in [taskgroup.py](../../../python/iron/runtime/taskgroup.py).

A task is added to a group by passing `group=` to `fill`/`drain`. Tasks in the same group are appended to the runtime sequence and executed in order. The `finish()` method marks the end of a task group: it waits for tasks in the group annotated with `wait=True` to complete, then frees _all_ resources used by the group.
If no group is specified for the DMA tasks in a body, a single default task group is used.

> **NOTE:**  A call to  `finish()` blocks the runtime sequence until all of the group's tasks annotated with `wait=True`  ("awaited tasks") have completed. After waiting, all resources of the task group -- including those _not_ annotated with `wait=True` ("unawaited tasks") -- will be freed and reused for subsequent tasks. 
> 
> To avoid race conditions, any unawaited tasks in the group should form a dependency of an awaited task.
> It is only safe to remove a `wait=True` if you can reason that another, awaited task in the same group can only complete if the awaited task also completed.
> For example, you may choose to set `wait=False` on an input fill if you can guarantee that a later (awaited) output drain depends on the input and completes only if the input fill completed as well.
>
> If you suspect a race condition, the safest (but possibly slower) solution is to annotated _all_ tasks (including inputs) with `wait=True`.

The body in the code snippet below has two task groups. We can observe that the creation of the second task group happens at the end of execution of the first task group.
```python
def sequence(a_in, b, c_out, in_h, out_h):
    tg = TaskGroup()  # start first task group
    for _ in [0, 1]:
        in_h.fill(a_in, group=tg)
        out_h.drain(c_out, group=tg, wait=True)
        tg.finish()
        tg = TaskGroup()  # start second task group
    tg.finish()

rt = Runtime(
    sequence,
    [data_ty, data_ty, data_ty],
    fn_args=[of_in.prod(), of_out.cons()],
)
```

-----
[Up](./README.md)
