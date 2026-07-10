<!-- Copyright (C) 2024-2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# High-level IRON API

IRON is the high-level Python interface for programming AMD Ryzen™ AI NPUs.
It exposes tile placement, data movement, and host runtime as Python objects
that compile to an optimized `xclbin` + instruction stream via the MLIR-AIE
toolchain.

All symbols documented here are importable directly from the `iron`
namespace (e.g. `from iron import Worker, ObjectFifo, Runtime`).

For hand-routed data movement below the [`ObjectFifo`][iron.ObjectFifo]
abstraction, see the [Lower-level primitives](iron_lowlevel.md) page.

---

## Core design abstractions

The objects you use to describe an NPU design.

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

The host-side orchestration entry point. Lower-level runtime task types
(`DMATask`, task groups) are documented on the
[Lower-level primitives](iron_lowlevel.md#runtime-tasks) page.

::: iron.runtime.runtime
    options:
      show_root_heading: false

### Buffer

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
