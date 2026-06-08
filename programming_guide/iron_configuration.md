<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# IRON Python Configurations

There are several options that exist to configure the IRON Python programming environment.

## Default IRON Tensor Class

This is a variable that controls the types of [```aie.utils.Tensor```](../python/utils/hostruntime/tensor_class.py)s that are produced by the utility functions ```tensor```, ```ones```, etc. Right now there are two tensor implementations: [```CPUOnlyTensor```](../python/utils/hostruntime/tensor_class.py) and [```XRTTensor```](../python/utils/hostruntime/xrtruntime/tensor.py).

By default, if ```pyxrt``` is available, the ```DEFAULT_TENSOR_CLASS``` is set to ```XRTTensor```. However, you can also manually set this value through the ```set_tensor_class()```, e.g.:
```python
>>> import numpy as np
>>> print(aie.utils.tensor.DEFAULT_TENSOR_CLASS.__name__)
XRTTensor
>>> type(iron.tensor((2, 2), np.int32))
<class 'aie.utils.xrtruntime.tensor.XRTTensor'>
>>> aie.utils.set_tensor_class(aie.utils.tensor.CPUOnlyTensor)
>>> print(aie.utils.tensor.DEFAULT_TENSOR_CLASS.__name__)
CPUOnlyTensor
>>> type(aie.utils.tensor((2, 2), np.int32))
<class 'aie.utils.tensor.CPUOnlyTensor'>
```

## Default IRON Device

If the IRON device is not set, many designs will fetch it on demand from the [`DefaultNPURuntime`](../python/utils/__init__.py) (a `CachedXRTRuntime` instance), which queries XRT for the attached NPU. You can override this by calling [`iron.set_current_device()`](../python/utils/hostruntime/__init__.py), which takes the new device and returns the previous one:
```python
>>> import aie.iron as iron
>>> iron.set_current_device(iron.device.NPU1())
<abc.NPU2 object at 0x722a659826c0>
>>> iron.get_current_device()
<abc.NPU1 object at 0x722a65903a10>
```

### Cross-compiling for a different NPU

The target architecture (`aie2` / `aie2p`) is part of the per-design
cache key (see [compilation_stages.md](./compilation_stages.md)
§Lowering — `_compute_artifact_hash` mixes `target_arch` in along with
peano + aiecc mtimes), so `set_current_device(...)` before `.compile()`
is enough to drive a cross-arch build.  Each arch lands in its own
cache subdirectory, no collision with the binary for whatever NPU is
physically attached:

```python
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.utils.hostruntime import set_current_device

# Same generator, two arches → two distinct cache dirs.
for dev_cls in (NPU1Col1, NPU2Col1):
    set_current_device(dev_cls())
    my_design.specialize(N=4096).compile()
```

Useful for building Strix binaries on Phoenix-only CI hosts (or vice
versa), and for shipping pre-built xclbins for multiple NPU generations
without needing each one attached at build time.

### Arch-aware kernel introspection (`mac_dims`)

Some `aie.iron.kernels` factories pick a different MMUL geometry per
arch — `kernels.mm(int16, int16)` is `(r, s, t) = (4, 4, 4)` on AIE2
(Phoenix) but `(4, 4, 8)` on AIE2P (Strix).  The chosen geometry is
exposed on the returned `ExternalFunction` as `.mac_dims`, so designs
can drive their DMA-layout transforms from the kernel itself instead
of hardcoding for one arch:

```python
mm = kernels.mm(dim_m=64, dim_k=64, dim_n=64,
                input_dtype=np.int16, output_dtype=np.int16)
r, s, t = mm.mac_dims
# r/s/t now reflect whichever arch is active per get_current_device().
```

The full per-arch table lives in
[`python/iron/kernels/linalg.py`](../python/iron/kernels/linalg.py)
(`_MM_MAC_DIMS`).  Combined with the cross-compile pattern above, the
same generator file produces a correct binary for each arch without
ever editing the source.

## IRON Cache Location

The IRON jit feature caches compiled objects in a directory defined by ```NPU_CACHE_HOME```. By default this value is the user's home directory.

## IRON XRT Runtime Cache Size

The `CachedXRTRuntime` caches XRT contexts to improve performance. The size of this cache can be configured using the `XRT_CONTEXT_CACHE_SIZE` environment variable. This is particularly useful in CI environments where multiple tests run in parallel and might exhaust the available NPU contexts.

```bash
export XRT_CONTEXT_CACHE_SIZE=1
```

## Diagnostic Output and Log Level

The `aie` library uses Python's standard `logging` module for all diagnostic output. Set
`AIE_LOG_LEVEL` to control verbosity. Valid values: `DEBUG`, `INFO`, `WARNING`
(default), `ERROR`, `CRITICAL`.

```bash
AIE_LOG_LEVEL=DEBUG python my_script.py    # show debug messages
AIE_LOG_LEVEL=INFO python my_script.py     # show info and above
AIE_LOG_LEVEL=ERROR python my_script.py    # errors only
```

For per-module control or routing to a file, use the `logging` API directly:

```python
import logging

logging.getLogger("aie").setLevel(logging.ERROR)

# Route aie logs to a file instead of the console
handler = logging.FileHandler("aie.log")
handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
logging.getLogger("aie").addHandler(handler)
logging.getLogger("aie").propagate = False  # don't also send to root logger
```

## Helper utilities cheat-sheet

The helpers most designs reach for, grouped by where they live.  Each
has a docstring; `help(obj)` or `print(obj.__doc__)` shows it.

### Host-side tensor + design construction (`aie.iron`)

| Helper | What it does |
|--------|--------------|
| `iron.{tensor, zeros, ones, full, rand, randint, arange, zeros_like}` | Host-tensor factories (default `device="npu"`, also accept `device="cpu"`); see also [`set_tensor_class`](#default-iron-tensor-class). |
| `iron.{In, Out, InOut, Compile}` | Type annotation markers for `@iron.jit` generator parameters (see [`compilation_stages.md`](./compilation_stages.md) §Appendix A). |
| `iron.{jit, CompilableDesign, CallableDesign}` | The JIT decorator + the two design wrapper classes (`CompilableDesign` is the recipe; `CallableDesign` is the ready-to-run wrapper). |
| `iron.ceildiv(a, b)` | Pure-integer ceiling division.  Same value on every code path; here so designs don't redefine it locally. |
| `iron.{set_current_device, get_current_device}` | Read/write the active `Device` (see [§Default IRON Device](#default-iron-device)). |
| `iron.kernels.*` | Pre-packaged kernel factories — `mm`, `conv2dk1`, `conv2dk3`, `passthrough`, eltwise, etc.  Each returns an `ExternalFunction` ready to bind in a `Worker`. |
| `iron.{Buffer, Lock, Flow, TileDma, DmaChannel, Bd, Acquire, Release}` | IRON-Python peers of `ObjectFifo` for designs that want to hand-wire DMA + sync (canonical example: `programming_examples/basic/chaining_channels/`). |
| `iron.algorithms.{transform_typed, transform_binary_typed, transform_parallel, for_each_typed}` | Element-wise dataflow templates — handle `Worker` / `ObjectFifo` / `Runtime` plumbing for one-arg / two-arg / multi-column / fill-and-drain patterns. |
| `iron.{compile_context, get_compile_arg}` | Dynamic compile-time arg injection.  See [§compile_context](#compile_context-for-nested-generator-helpers) below. |

### Argparse + runtime glue (`aie.iron.device`, `aie.utils`)

| Helper | What it does |
|--------|--------------|
| `aie.utils.hostruntime.argparse.device_from_args(args)` | Resolve a parsed argparse `Namespace` to a `Device` — collapses `from_name(args.dev, n_cols=...)` boilerplate.  `n_cols="auto"` reads `args.n_cols` if present, otherwise defaults to 1.  Lives next to the `add_*_args` family that produces the `Namespace` it consumes. |
| `aie.utils.DefaultNPURuntime` | Module-level `CachedXRTRuntime` instance; auto-detects NPU1 / NPU2 via XRT.  Used by `iron.tensor(..., device="npu")` and `@iron.jit` runtime binding. |
| `aie.utils.hostruntime.argparse.{add_compile_args, add_runtime_args}` | Add the standard `--xclbin-path`/`--insts-path` and `--xclbin`/`--instr`/`-k`/`--trace_size` flags to a parser. |
| `aie.utils.test.create_npu_kernel(opts)` | Build an `NPUKernel` (plus optional `TraceConfig`) from a parsed `argparse.Namespace`.  See `programming_examples/basic/vector_scalar_mul/test.py` for the canonical use pattern. |
| `aie.utils.benchmark.{run_iters, print_benchmark}` | Warmup + timed iterations + summary stats.  See [section-4/section-4a/README.md](./section-4/section-4a/README.md). |
| `aie.utils.verify.{nearly_equal, count_mismatches}` | Tolerance-aware compare for LUT / saturating / bf16 outputs that don't match exactly. |

### Compile-pipeline introspection (`aie.utils.compile`)

| Helper | What it does |
|--------|--------------|
| `aie.utils.compile.NPU_CACHE_HOME` | Cache root `Path`; defaults to `~/.npu/cache`, override with the `NPU_CACHE_HOME` env var. |
| `aie.utils.compile.jit._dma_size_parser.parse_dma_sizes(kernel_dir)` | Per-host-arg element counts read from the entry-point `aie.runtime_sequence` in `input_with_addresses.mlir`.  Backs the [tensor-arg validation](./compilation_stages.md) in stage 5. |
| `logging.getLogger("aie.utils.compile").setLevel(logging.DEBUG)` | Print every `clang++` / `aiecc` subprocess invocation.  See [compilation_stages.md §Watching the compile pipeline](./compilation_stages.md#watching-the-compile-pipeline). |

### `compile_context` for nested generator helpers

Most designs supply compile-time values via the explicit
`Compile[T]`-annotated generator signature.  Some patterns — composite
generators, helper functions reused across designs — want to inject
values *through* a helper that doesn't take them as explicit kwargs.
`compile_context` opens a per-thread context that `get_compile_arg`
can read out:

```python
from aie.iron import compile_context, get_compile_arg

def make_fifo_pair(line_ty):
    # Helper has no name_prefix kwarg, but can read one from context.
    prefix = get_compile_arg("prefix", default="")
    return (ObjectFifo(line_ty, name=f"{prefix}in"),
            ObjectFifo(line_ty, name=f"{prefix}out"))

with compile_context(prefix="layer1_"):
    of_in, of_out = make_fifo_pair(line_ty)        # layer1_in, layer1_out

of_in, of_out = make_fifo_pair(line_ty)            # in, out (default)
```

Contexts nest — inner values shadow outer ones for the duration of the
inner `with` block.  Implemented on top of `contextvars`, so it is
thread- and async-safe.  `CompilableDesign.compile()` uses the same
mechanism internally to surface the bound `Compile[T]` kwargs to the
generator body.

Prefer explicit `Compile[T]` parameters when you can; reserve
`compile_context` for the cases where threading the value through every
helper signature would obscure the design.
