<!---//===- kernels_library.md -----------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# IRON Kernel Library — `aie.iron.kernels`

`aie.iron.kernels` packages the AIE compute kernels under
[`aie_kernels/`](../aie_kernels/) as factory functions that return
ready-to-bind [`ExternalFunction`](../python/iron/kernel.py) objects.
Each factory bundles three things that designs would otherwise repeat
by hand:

* The source path (e.g. `aie_kernels/aie2/mm.cc`).
* The compile flags (e.g. `-DDIM_M=64 -DDIM_K=64 -DDIM_N=64 -DBIT_WIDTH=16`).
* The typed argument list (e.g. `[a_ty, b_ty, c_ty]`).

A factory call inside an `@iron.jit` design is sufficient to compile
the kernel `.o` into the JIT work directory automatically — there is
no separate Makefile rule to keep in sync.

## A first example: `passthrough`

The simplest factory call wires straight into a `Worker`:

```python
import numpy as np
import aie.iron as iron
from aie.iron import ObjectFifo, Worker, Runtime, Program, In, Out, CompileTime
import aie.iron.kernels as kernels

@iron.jit
def passthrough_design(a: In, b: Out, *, N: CompileTime[int]):
    line_ty = np.ndarray[(N,), np.dtype[np.uint8]]
    of_in = ObjectFifo(line_ty, name="in")
    of_out = ObjectFifo(line_ty, name="out")

    # One call → an ExternalFunction carrying source path + flags + arg list.
    pt = kernels.passthrough(tile_size=N, dtype=np.uint8)

    def core_fn(of_in, of_out, pt):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        pt(elem_in, elem_out, N)        # invocation matches pt.arg_types
        of_in.release(1)
        of_out.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_out.prod(), pt])
    rt = Runtime()
    with rt.sequence(line_ty, line_ty) as (a_in, b_out):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), b_out, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()
```

The kernel `.o` lands in the per-design cache directory alongside the
xclbin — see [`compilation_stages.md`](./compilation_stages.md)
§Per-design cache directory contents.  No Makefile rule was harmed.

## Sibling kernels share one `.o`: `kernels.mm(...).zero`

Some factories expose an extra binding for a companion symbol that
lives in the same `.cc`.  `kernels.mm(...)` is the canonical case: the
matmul `.cc` exports both `matmul_*` and `zero_*` symbols, and the
returned `ExternalFunction` carries a `.zero` attribute that binds the
zero-fill kernel against *the same* compiled `.o`:

```python
matmul = kernels.mm(
    dim_m=m, dim_k=k, dim_n=n,
    input_dtype=np.int16,
    output_dtype=np.int16,
)
zero_kernel = matmul.zero          # sibling binding, no extra compile
```

Without the `.zero` attribute the design would have to call a separate
`kernels.mm_zero(...)` factory that recompiled `mm.cc` a second time
for no functional benefit.  The same pattern applies to any factory
that documents a `.zero` (today: `mm` and `mv`).

`kernels.mm(...)` also exposes `.mac_dims` — the `(r, s, t)` MMUL
geometry the kernel was compiled with, which varies by arch and dtype.
Designs read it to drive their DMA layout transforms without
hardcoding for one NPU generation.  See
[`iron_configuration.md`](./iron_configuration.md)
§Arch-aware kernel introspection.

## Shared-buffer factory kwargs

Some convolution factories accept an opt-in kwarg that decouples the
*buffer* size from the *call* size, so a design that splits the work
across multiple workers can share one weight tape between them:

| Factory | Kwarg | Decouples |
|---------|-------|-----------|
| `kernels.conv2dk3(weight_output_channels=)` | weight buffer total OCs | from per-call `output_channels` slice (the `channel_offset` runtime arg selects this worker's slice) |
| `kernels.conv2dk1_skip_init(skip_input_channels=)` | skip-projection weights ICs | from main-conv `input_channels`; the two weight blocks are concatenated in one buffer |
| `kernels.bn_conv2dk1_relu_xy_pool_padded(weight_chunk_count=)` | per-call weight tape chunk | from full `input_channels * output_channels` tile (for cascade / output-split streaming) |
| `kernels.bn_fc_relu_ui16_pad(weight_chunk_count=)` | same as above for FC | (used by the MobileNet V3 classifier head) |

Default behavior is unchanged — leave the kwarg unset and the
buffer / call sizes match like always.

## Discovering what's available

`aie.iron.kernels` is organised by category, with a one-line summary
on each submodule's `__doc__`:

| Submodule | What's in it |
|-----------|--------------|
| [`kernels.eltwise`](../python/iron/kernels/eltwise.py)       | element-wise: passthrough, scale, add, mul, relu |
| [`kernels.reduce`](../python/iron/kernels/reduce.py)         | reductions: reduce_add, reduce_min, reduce_max, compute_max |
| [`kernels.activation`](../python/iron/kernels/activation.py) | activations: softmax, gelu, silu, swiglu, bf16_exp.  Companion numpy refs (`relu_ref`, `silu_ref`, `gelu_ref`, `bf16_exp_ref`, `softmax_ref`) live in the same module so host harnesses don't reimplement the math. |
| [`kernels.linalg`](../python/iron/kernels/linalg.py)         | linear algebra: mm (+ `.zero`, `.mac_dims`), mv (+ `.zero`), cascade_mm (+ `.{get_only,put_only,put_get,zero}`, `.mac_dims`) |
| [`kernels.conv`](../python/iron/kernels/conv.py)             | convolutions: conv2dk1/3/14, conv2dk1_skip(_init), bn_* bottleneck variants for MobileNet/ResNet |
| [`kernels.vision`](../python/iron/kernels/vision.py)         | vision: rgba2hue, rgba2gray, gray2rgba, threshold, bitwise_or/and, filter2d, add_weighted |

The submodule files are the authoritative catalog — each function has
a complete docstring covering its kwargs, dtype constraints, and the
raised `ValueError` envelope.  To list factory names + signatures
without leaving the REPL:

```python
import aie.iron.kernels as kernels
help(kernels)                 # package summary
help(kernels.conv)            # one submodule + all its factories
help(kernels.mm)              # one factory + full kwargs
print(kernels.mm.__doc__)     # raw access, no pager

# Tab-completion: in IPython / Jupyter,
kernels.<TAB>                 # category-level
kernels.conv.<TAB>            # factory-level
```

The factories themselves are short — typically a `_make_extern` call
plus dtype validation.  Reading the source of a factory you're about
to use is often faster than chasing through the docstring.

## When you outgrow the library

If no factory matches what your design needs, drop down to
[`ExternalFunction`](../python/iron/kernel.py) directly.  Every kernel
in `aie.iron.kernels` is built this way; the factories save typing,
they don't gate anything:

```python
from aie.iron import ExternalFunction

my_kernel = ExternalFunction(
    "my_kernel_symbol",
    source_file="path/to/my_kernel.cc",
    arg_types=[a_ty, b_ty, out_ty, np.int32],
    compile_flags=["-DTILE_SIZE=1024", "-DUSE_VECTORIZED"],
    include_dirs=[...],
)
```

The minimal pattern around `_make_extern` in
[`python/iron/kernels/eltwise.py`](../python/iron/kernels/eltwise.py)
(see `passthrough()` for the simplest case) is a good template.  If
the new kernel proves useful across designs, contributing it back as a
new factory under the right submodule is straightforward — match the
existing module's docstring + `ValueError` shape and the auto-listing
above picks it up.

## Related reading

* [`compilation_stages.md`](./compilation_stages.md) — how the factory
  call's source / flags / arg list flow through the JIT pipeline into
  the per-design cache directory.
* [`iron_configuration.md`](./iron_configuration.md) §Arch-aware
  kernel introspection — `.mac_dims` for portable matmul designs.
* [`section-4/section-4a/README.md`](./section-4/section-4a/README.md)
  §Verifying NPU output — `aie.utils.verify.{nearly_equal,
  count_mismatches}` for LUT-approximation kernels (most things under
  `kernels.activation`).
