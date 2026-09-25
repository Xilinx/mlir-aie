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

A `Runtime` is created from its sequence body and `fn_args`, mirroring how a `Worker` is created from a `core_fn` and its `fn_args`:
```python
# To/from AIE-array runtime data movement
def sequence(a, b, c):
    # runtime tasks
    ...

rt = Runtime(sequence, [data_ty_a, data_ty_b, data_ty_c])
```
Here every `fn_args` entry (`data_ty_a`, ...) is a **type**, which declares a host-side buffer: the body receives one argument per entry and describes how those buffers move into the AIE-array. The body runs later, inside the lowered `aie.runtime_sequence`, so it executes with a live MLIR context — meaning native `range_`/`if_` control flow and data-movement verbs work directly inside it.

#### **Passing ObjectFifos to the body: `fn_args`**

The body moves data by calling `fill`/`drain` on `ObjectFifoHandle`s. Those handles are passed to the `Runtime` as trailing `fn_args` entries — exactly like a `Worker`'s `fn_args` — and are received as trailing parameters of the body, after the type-declared buffers:
```python
def sequence(a, b, c, in_h, out_h):
    in_h.fill(a)
    out_h.drain(c, wait=True)

rt = Runtime(
    sequence,
    [data_ty_a, data_ty_b, data_ty_c, of_in.prod(), of_out.cons()],
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

rt = Runtime(sequence, [data_ty, of_in.prod()])
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

rt = Runtime(sequence, [data_ty, of_out.cons()])
```

To pin the Shim tile a handle's host-side DMA uses, pass `tile=` to `prod()`/`cons()` where the handle is created (not to `fill`/`drain`):
```python
rt = Runtime(sequence, [data_ty, of_in.prod(tile=Tile(0, 0))])
```

The `fill()`/`drain()` methods return a `Task` handle. Prefer the default managed transfers and `TaskGroup` for ordinary data movement; the runtime handles their waits and frees.

For software-pipelined data movement with manual lifetime control, issue the transfer with `managed=False` and do not pass `group=`. Use `range_` and `yield_` from `aie.iron.controlflow` to carry a `Task` through `iter_args` across loop iterations. Call `.await_()` only on transfers issued with `wait=True` (which requests a completion token), then call `.free()` when it is safe to reuse the descriptor. Awaiting alone does not free it. An unwaited transfer may be freed only after a dependent waited transfer proves it has completed. Do not manually free managed tasks: their task group already owns that responsibility. See [dmataskhandle.py](../../../python/iron/runtime/dmataskhandle.py).

In a sequence whose loops all have compile-time trip counts, freeing is optional: when a tile runs out of BDs, the compiler takes them back from started tasks it can prove finished (see [Running Out of Buffer Descriptors](./DMATasks.md#running-out-of-buffer-descriptors)). A pipelined sequence can then issue every transfer with `managed=False`, set `wait=True` only on the transfers the host must wait for, and `.await_()` just those. Issue a task after the transfers it depends on, e.g. a block's fills before its drain, since the compiler may have to wait for it to finish before issuing anything else.

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
A `WorkerRuntimeBarrier` may take any value between 0 and the device's `max_lock_value` (63 on current devices). This is due to the fact that these barriers leverage the lock mechanism of the architecture under-the-hood.

> **NOTE:**  Similar to the `Buffer` it is possible to create a single barrier and pass it as input to multiple workers. At lower stages of compiler abstraction this will result in a different lock being employed for each worker.

#### **Runtime Task Groups**

It may be desirable to reconfigure a `Runtime`'s `sequence` and reuse some of the resources from a previous configuration, especially given that some of these resources, like the BDs in a DMA task queue, are limited. The compiler already reuses the BDs of tasks it can prove finished, so a task group is not needed just to stay within the BD pool; use one to say when the sequence should wait.

To facilitate this reconfiguration step, IRON introduces `TaskGroup`s, created with the `TaskGroup()` constructor as defined in [taskgroup.py](../../../python/iron/runtime/taskgroup.py).

A task is added to a group by passing `group=` to `fill`/`drain`. Transfers are submitted in sequence order and may overlap. The `finish()` method marks the end of a task group: it waits for tasks in the group annotated with `wait=True` to complete, then frees _all_ resources used by the group.
If no group is specified for managed DMA tasks in a body, a single default task group is used and finished at the end of the sequence. By default, `Runtime` rejects mixing explicit groups with this default group; assign all managed transfers to explicit groups when using them.

> **NOTE:**  A call to  `finish()` blocks the runtime sequence until all of the group's tasks annotated with `wait=True`  ("awaited tasks") have completed. After waiting, all resources of the task group -- including those _not_ annotated with `wait=True` ("unawaited tasks") -- will be freed and reused for subsequent tasks. 
> 
> To avoid race conditions, any unawaited tasks in the group should form a dependency of an awaited task.
> It is only safe to remove a `wait=True` if you can reason that another, awaited task in the same group can only complete if the awaited task also completed.
> For example, you may choose to set `wait=False` on an input fill if you can guarantee that a later (awaited) output drain depends on the input and completes only if the input fill completed as well.
>
> If you suspect a race condition, the safest (but possibly slower) solution is to annotated _all_ tasks (including inputs) with `wait=True`.

The body in the code snippet below has two task groups. Each group is finished before the next iteration submits its transfers.
```python
def sequence(a_in, b, c_out, in_h, out_h):
    for _ in range(2):
        tg = TaskGroup()
        in_h.fill(a_in, group=tg)
        out_h.drain(c_out, group=tg, wait=True)
        tg.finish()

rt = Runtime(
    sequence,
    [data_ty, data_ty, data_ty, of_in.prod(), of_out.cons()],
)
```

## Dispatch-time scalars

`DispatchTime[T]` rebuilds the instruction stream for each call using a compiled
host builder. `T` must be a supported NumPy integer scalar type, such as
`np.int32` or `np.int64`; built-in `int`/`bool` and floating-point types are
rejected.

- **Call-time value:** overrides the signature default without recompiling.
  If omitted, the default is used; without a default, the value is required.
- **Explicit specialization:** `iron.jit(generator, count=3)` or
  `design.specialize(count=3)` fixes the value and includes it in the cache key.
  Calls cannot override it; use another specialization to change it.

Defaults do not specialize parameters. Tensor capacities and worker tiling
remain compile-time properties; callers must keep dispatch values within the
design's valid ranges and buffer capacities.

`DispatchTime` parameters must be keyword-only, even when defaulted or
explicitly specialized. Prefer tensors first, then dispatch scalars, then
compile-time configuration; group ordering is a convention, not a restriction:

```python
import numpy as np
import aie.iron as iron

@iron.jit
def copy(a: iron.In, b: iron.Out, *,
         count: iron.DispatchTime[np.int32] = 3,
         tile_size: iron.CompileTime[int] = 256):
    ...  # Build the design.

copy(a, b)                        # Dispatch with the default count=3.
copy(a, b, count=6)               # Same compiled design; a different dispatch.
copy.specialize(count=3)(a, b)    # Compile with count fixed to 3.
```

### Generator-side binding and scope

The generator receives an identity-bearing symbolic parameter for each unbound
`DispatchTime[T]`, or a typed NumPy constant for a specialized one. Forward each
symbolic parameter exactly once as a direct `Runtime` argument, **in any order**.
The callback receives the corresponding SSA values:

```python
@iron.jit
def design(*, bar: iron.DispatchTime[np.int32],
           baz: iron.DispatchTime[np.int32]):
    def seq(baz_value, bar_value):
        ...  # baz_value corresponds to baz, bar_value to bar.

    rt = iron.Runtime(seq, fn_args=[baz, bar])
    ...  # Build and resolve the Program with rt.
```

Aliases preserve identity. Missing or duplicate bindings, bare scalar-type
substitutes, and bindings to multiple sequences are rejected.

Use the **callback argument**, not the captured symbolic parameter, for runtime
arithmetic and MLIR control flow. Generation-time arithmetic, comparisons,
`if bar`, `range(bar)`, NumPy value/dtype conversion, and `Worker.fn_args` reject
symbolic parameters with `TypeError`. Shapes and worker configuration require
`CompileTime[T]` or explicit specialization; storing or forwarding a symbolic
parameter is valid.

### Compilation scope

The JIT requests device artifacts and a C++ transaction builder from the same
`aiecc` invocation and lowering pipeline. Python validates the scalar ABI and
compiles the host library. This requires a
[host C++17 compiler](../../iron_configuration.md#dispatch-time-scalar-compilation),
including with wheel installations; subsequent dispatches call the library
without compiling.

By default, artifacts live in the JIT cache. For explicit outputs, use
`compile(xclbin_path=..., pdi_path=...)` (`pdi_path` is optional). Retain the
builder library in the adjacent `<xclbin stem>.prj` directory with the device
artifacts; `design.compilable.get_dispatch_lib_path()` returns its path.

The Python bridge supports one runtime sequence and rejects remaining
`load_pdi` operations because its runtimes cannot supply those resources.
While any parameters remain dynamic, `inst_path`, `elf_path`, and
`full_elf=True` are unsupported: full ELF embeds static instructions, with no
per-call replacement API. Fully specialized designs retain normal full-ELF
support. Native hosts can request C++ builders, including reconfiguration
builders, directly from [`aiecc`](../../../tools/aiecc/README.md#parameterized-c-transaction-builders).

-----
[Up](./README.md)
