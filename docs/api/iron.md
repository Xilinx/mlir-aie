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

### Dispatch-time scalars

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
[host C++17 compiler](../programming_guide/iron_configuration.md#dispatch-time-scalar-compilation),
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
builders, directly from [`aiecc`](../aiecc/README.md#parameterized-c-transaction-builders).

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
[`PacketDest`][iron.PacketDest] endpoint descriptor.

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

::: iron.runtime.dmataskhandle
    options:
      show_root_heading: false
