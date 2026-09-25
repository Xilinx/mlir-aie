<!-- Copyright (C) 2024-2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# IRON API (`aie.iron`)

IRON is the high-level Python interface for programming AMD Ryzen™ AI NPUs.
You describe a design as Python objects — tiles, workers, data movement, and a
host runtime — and IRON compiles it to an optimized `xclbin` + instruction
stream via the MLIR-AIE toolchain.

Structural design objects are **resolvable**: they lower to MLIR operations
when the design is compiled. This page also includes host-side utilities,
type markers, and runtime task handles. For the direct MLIR op wrappers that IRON lowers *to*, see the
[Dialect op wrappers](dialect_wrappers.md) page.

Import the public design abstractions from `aie.iron`
(e.g. `from aie.iron import Worker, ObjectFifo, Runtime`), or use
`import aie.iron as iron` for decorators and tensor factories. Supporting
types are also documented below; not all are re-exported at package level.

Prefer `ObjectFifo` for synchronized streaming, `Worker` for compute, and
`Runtime(sequence, fn_args)` for host-side transfers. Pass runtime buffer
types and the ObjectFifo handles used by the sequence through `fn_args`,
and pass workers to `Program(workers=...)`. Use `Flow`, `TileDma`, and `Lock`
when explicit routing or DMA control is the teaching goal; use raw dialect
operations only where those abstractions do not expose the required operation.

---

## Core design abstractions

The objects most designs are built from.

### Program

::: iron.program
    options:
      show_root_heading: false

### Worker

::: iron.worker
    options:
      show_root_heading: false

### ObjectFifo

::: iron.dataflow.objectfifo
    options:
      show_root_heading: false

### Runtime

The host-side orchestration entry point. Calls to producer-handle `fill` and
consumer-handle `drain` are declared in the sequence body passed to `Runtime(seq, fn_args)`;
Workers are passed to `Program(workers=...)`.

::: iron.runtime.runtime
    options:
      show_root_heading: false

### Buffer

Use a `Buffer` for local scratch storage shared by sequential kernel calls
in one Worker, as in the [edge-detection example](../../programming_examples/vision/edge_detect/).
Passing it in the Worker's `fn_args` associates it with that Worker's tile.
Unlike an ObjectFifo, a Buffer does not provide producer/consumer synchronization.

::: iron.buffer
    options:
      show_root_heading: false

### Kernels

`Kernel` binds a function symbol; `KernelObject` owns the shared link artifact.
Both are exported from `aie.iron`. Pass a `KernelObject("shared.o")` to several
`Kernel` constructors to bind symbols from one precompiled object;
`ObjectFile("shared.o", symbol_prefix=...)` is the same for a prebuilt object
whose symbols were renamed under a prefix. For C++ source,
`ExternalFunction` creates the owner, exposed as `fn.object_file`;
`fn.object_file.bind(symbol, arg_types)` binds another entry point to that owner,
applying its symbol prefix. External functions with the same
explicit output filename and identical source recipes also share ownership.
Conflicting recipes for one output filename are rejected.
Resolving a source-backed binding registers its artifact for compilation without
requiring the original `ExternalFunction` to remain alive. Independent operations
such as `kernels.zero(...)` own their own objects.

::: iron.kernel
    options:
      show_root_heading: false

### ScratchpadParameter

::: iron.scratchpad_parameter
    options:
      show_root_heading: false

---

## Compile-time & JIT

Decorators and markers for JIT-compiling a design and injecting compile-time
constants. These are re-exported into `iron` from `aie.utils`.

!!! note
    The JIT entry point and tensor factories below are thin re-exports from
    the compiled `aie.utils` package. Their full signatures and source are
    available in the running package; the summaries here describe the public
    contract.

| Symbol | Kind | Summary |
|--------|------|---------|
| `iron.jit` | decorator | Compile a design on a cache miss, then run it on the attached NPU. |
| `iron.CompilableDesign` | class | Bundle a design generator with its compile-time configuration. |
| `iron.CallableDesign` | class | A compiled, callable design produced from a `CompilableDesign`. |
| `iron.compileconfig` | decorator | Attach compile-time configuration to a design generator. |
| `iron.get_compile_arg` | function | Dynamically inject a compile-time argument (advanced). |
| `iron.In` / `iron.Out` / `iron.InOut` | markers | Type-annotation markers for design inputs/outputs. |
| `iron.CompileTime` | marker | Type-annotation marker for a compile-time constant argument. |
| `iron.DispatchTime` | marker | Integer scalar that can vary per call without recompiling the device program. |

For dispatch scalar defaults, specialization, and runtime binding, see
[Dispatch-time scalars](../programming_guide/section-2/section-2d/RuntimeTasks.md#dispatch-time-scalars)
in the runtime data-movement guide.

See the [Programming Guide](../programming_guide/README.md) for worked
examples of `@iron.jit`.

---

## Tensor factories

NumPy-like helpers that allocate NPU-accessible host tensors. Re-exported
into `iron` from `aie.utils`.

| Symbol | Summary |
|--------|---------|
| `iron.tensor` | Wrap existing data as an NPU-accessible tensor. |
| `iron.arange` | NPU-accessible analogue of `numpy.arange`. |
| `iron.zeros` / `iron.ones` / `iron.full` | Allocate a tensor filled with 0, 1, or a constant. |
| `iron.zeros_like` | Allocate a zero tensor matching another's shape/dtype. |
| `iron.rand` / `iron.randint` | Allocate a tensor of random floats / integers. |

## Device management

| Symbol | Summary |
|--------|---------|
| `iron.get_current_device` | Return the currently selected NPU device. |
| `iron.set_current_device` | Select the NPU device for subsequent allocations. |
| `iron.ensure_current_device` | Raise if no device is currently selected. |

The [`Device`][iron.device.device.Device] also reports the hardware limits a
design is sized against, such as `max_lock_value`, `max_repeat_count`,
`dma_task_queue_depth` and `get_num_bds(tile_type)`, so a design can read them
instead of hardcoding them.

::: iron.device.device
    options:
      show_root_heading: false
      members:
        - Device

---

## Data type helpers

::: iron.dtype
    options:
      show_root_heading: false

---

## Advanced primitives

Still part of the `aie.iron` API —
but reach for these only when the managed [`ObjectFifo`][iron.ObjectFifo]
abstraction is not enough and you need explicit control over routing, DMA
descriptors, and locks.

### Flow / PacketFlow

Circuit-switched ([`Flow`][iron.Flow]) and packet-switched
([`PacketFlow`][iron.PacketFlow]) stream connections, plus the
[`PacketDest`][iron.PacketDest] endpoint descriptor.  A `Flow` given no DMA
channels lets the compiler assign them; name the assigned channel with
`flow.endpoint(tile)`, which returns a [`FlowEndpoint`][iron.FlowEndpoint]
that DMA programs, `tile_dma_task` and `tile_dma_chain` take in place of a
channel index.  A `Flow` or `PacketFlow` with one shim end has `fill` /
`drain`; `PacketFlow.fill` stamps the route's packet ID on the input.

::: iron.dataflow.flow
    options:
      show_root_heading: false

### CascadeFlow

Directed cascade-stream connection between two adjacent Workers.

::: iron.dataflow.cascadeflow
    options:
      show_root_heading: false

### TileDma / DmaChannel / Bd

Explicit tile DMA programs: [`TileDma`][iron.TileDma],
[`DmaChannel`][iron.DmaChannel], buffer descriptors ([`Bd`][iron.Bd]), and the
[`Acquire`][iron.Acquire] / [`Release`][iron.Release] lock actions.

::: iron.dataflow.tile_dma
    options:
      show_root_heading: false

### Lock

::: iron.lock
    options:
      show_root_heading: false

### Runtime tasks

Lower-level runtime task types scheduled by the
[`Runtime`][iron.runtime.runtime.Runtime].

::: iron.runtime.task
    options:
      show_root_heading: false

::: iron.runtime.taskgroup
    options:
      show_root_heading: false

::: iron.runtime.dmatask
    options:
      show_root_heading: false

::: iron.runtime.tiledmatask
    options:
      show_root_heading: false

::: iron.runtime.dmataskhandle
    options:
      show_root_heading: false
