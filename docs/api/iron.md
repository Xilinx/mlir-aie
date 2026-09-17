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
| `iron.jit` | decorator | JIT-compile a design and run it on the attached NPU (Triton-style). The first call compiles to an `xclbin` + instruction stream; later calls hit a cache. |
| `iron.CompilableDesign` | class | Bundle a design generator with its compile-time configuration. |
| `iron.CallableDesign` | class | A compiled, callable design produced from a `CompilableDesign`. |
| `iron.compileconfig` | decorator | Attach compile-time configuration to a design generator. |
| `iron.get_compile_arg` | function | Dynamically inject a compile-time argument (advanced). |
| `iron.In` / `iron.Out` / `iron.InOut` | markers | Type-annotation markers for design inputs/outputs. |
| `iron.CompileTime` | marker | Type-annotation marker for a compile-time constant argument. |
| `iron.DispatchTime` | marker | Type-annotation marker for a NumPy integer scalar argument. Changing its value between calls reuses the compiled design and rebuilds only the instruction stream. Explicit prebinding specializes it to a compile-time constant. |

For a generator with `M: iron.DispatchTime[np.int32]`, `iron.jit(generator)`
keeps `M` dynamic, including when the signature supplies a default.
If a call omits `M`, its signature default is used for that dispatch; without
a default, the caller must supply `M`. An explicit call-time value overrides
the default without recompiling. Using the default does not specialize `M`.
`iron.jit(generator, M=256)` or `design.specialize(M=256)` instead fixes `M`
for that specialization and includes the constant in its cache key. A call
cannot override a specialized parameter; create another specialization instead.
This lets the same generator express static and dynamic runtime sequences.
Tensor capacities and worker tiling remain compile-time configuration.

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

### Generator-side binding limitations

An unbound `DispatchTime[T]` parameter currently reaches the generator as the
NumPy scalar **type** `T`, not an identity-bearing symbolic value. Forward
each such parameter once to `Runtime(seq, fn_args=[...])`, in signature order
relative to the other unbound dispatch parameters. The runtime-sequence body
receives the corresponding SSA block arguments. Explicitly specialized
parameters instead reach the generator as typed NumPy constants.

!!! warning
    Binding is currently positional, not tracked through Python variable
    identity. With two `DispatchTime[np.int32]` parameters, forwarding them in
    reverse order silently swaps their values. The ABI check detects argument
    count and C-type mismatches, but cannot detect a same-type permutation or
    replacement.

    There is also no general check restricting an unbound parameter's use to
    the runtime sequence. Using it as an integer may raise a Python type error,
    but operations valid on a type object (such as a Python truth test or using
    it as a dtype) can succeed at generation time. Do not use unbound dispatch
    parameters for tensor shapes, worker configuration, or Python conditionals.
    Use `CompileTime[T]` or explicit specialization for those purposes.

### Compilation scope

Dynamic designs accept `compile(xclbin_path=...)` and an optional `pdi_path`.
Their dispatch library resides in the adjacent `<xclbin stem>.prj` directory;
use `CompilableDesign.get_dispatch_lib_path()` to locate it and retain it with
the xclbin. There is no static instruction stream, so `inst_path`, `elf_path`,
and `full_elf=True` are unsupported while any parameters remain dynamic.
The default compilation mode manages these artifacts in the JIT cache.

The JIT still invokes `aiecc` to build device artifacts. For the dispatch
builder, it then reads `input_with_addresses.mlir`, runs the shared
`aie-npu-dma-lowering` pipeline in-process, translates to C++, and compiles the
host library. This pipeline is the DMA-lowering stage, not a replacement for
`aiecc`'s runtime-sequence materialization, load-PDI expansion, PDI-ID assignment,
or full-ELF packaging.

The dynamic bridge requires exactly one runtime sequence. Dynamic
multi-device/reconfiguration flows are not supported by this integration;
the sequence-count check alone does not validate those flows. Ordinary static
designs, including designs with every dispatch parameter explicitly
specialized, use the existing `aiecc` path.

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
