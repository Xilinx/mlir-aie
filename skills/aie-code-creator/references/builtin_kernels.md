<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Built-in Kernels and Algorithm Templates

Before writing a line of C++, check whether mlir-aie already ships the kernel and the
dataflow around it. Two packages cover most common work:

- **`aie.iron.kernels`** — factory functions returning a ready-to-use `ExternalFunction`.
  Each one points at a maintained, vectorized `.cc` in `aie_kernels/` and compiles it for you.
- **`aie.iron.algorithms`** — whole-design templates (fifos, workers, split/join, runtime
  sequence) that take a kernel and a tensor type and return a finished MLIR module.

Together they turn a multi-core element-wise design into ~15 lines with no C++ at all.
Hand-writing a kernel is still the right answer when nothing here matches — see
[`kernel_intrinsics.md`](kernel_intrinsics.md) — but reaching for it first is how you end up
maintaining a slower copy of a kernel that already exists.

## Coverage: does a built-in already do this?

| You need | Reach for | Notes |
|----------|-----------|-------|
| copy / passthrough | `kernels.passthrough(tile_size, dtype)` | `dtype` ∈ `uint8`, `int16`, `int32` |
| multiply by a scalar | `kernels.scale(tile_size, dtype, vectorized=True)` | `dtype` ∈ `int16`, `int32`; see signature note below |
| element-wise `a+b` / `a*b` | `kernels.add(...)` / `kernels.mul(...)` | **bf16 only**, `tile_size` fixed at 1024 |
| ReLU | `kernels.relu(tile_size=1024)` | bf16 |
| softmax / gelu / silu / swiglu / exp | `kernels.softmax`, `.gelu`, `.silu`, `.swiglu`, `.bf16_exp` | bf16, LUT-based |
| sum / min / max over a tile | `kernels.reduce_add`, `.reduce_min`, `.reduce_max` | |
| running max | `kernels.compute_max(dtype)` | |
| matrix multiply `C += A*B` | `kernels.mm(dim_m, dim_k, dim_n, input_dtype, output_dtype)` | see MMUL geometry below |
| matrix–vector `c += A*b` | `kernels.mv(dim_m, dim_k, ...)` | int16 in → int32 out only |
| cascade-accumulated matmul | `kernels.cascade_mm(...)` | pairs with `CascadeFlow` |
| conv2d 1×1 / 3×3 / skip / batchnorm | `kernels.conv2dk1`, `.conv2dk3`, `.conv2dk1_skip`, `.bn_*` | |
| RGBA↔gray/hue, threshold, filter2d, bitwise | `kernels.rgba2gray`, `.threshold`, `.filter2d`, … | vision ops |

Everything above is imported as `from aie.iron import kernels`.

The reference NumPy implementations (`kernels.relu_ref`, `.silu_ref`, `.gelu_ref`,
`.bf16_exp_ref`, `.softmax_ref`) exist so you can build the golden output for a test without
re-deriving the LUT approximation the kernel actually uses — compare against the `_ref`
version, not against exact `np` math, or the tolerance will look mysteriously bad.

## Algorithm templates

These return a complete `mlir.ir.Module` — they build the ObjectFifos, Workers, split/join,
and `Runtime` sequence for you. All of them are designed to be the entire body of an
`@iron.jit` function.

| Template | Shape of computation |
|----------|----------------------|
| `transform(func, tensor_ty, *params, tile_size=16)` | unary element-wise, one core |
| `transform_parallel(func, tensor_ty, *params, tile_size=16, num_channels=1, pass_size_to_kernel=True)` | unary element-wise across every column |
| `transform_binary(func, tensor_ty, ...)` | two inputs → one output, one core |
| `transform_parallel_binary(func, tensor_ty, ...)` | two inputs → one output, multi-core |
| `reduce(func, input_ty, output_ty, trace_size=0)` | whole tensor → small output; hands the *whole* input to one kernel call |
| `for_each(func, tensor_ty, tile_size=16)` | in-place transform over a tiled tensor |
| `row_at_a_time`, `row_at_a_time_tiled`, `row_at_a_time_with_skip`, `sliding_3row` | conv-style row pipelines |

Import as `from aie.iron.algorithms import transform_parallel` (etc.).

### Kernel signatures worth checking before you wire one up

The `ExternalFunction` a factory returns carries a concrete `arg_types` list, and a mismatch
fails at MLIR verification with an argument-count error rather than at runtime. Two that
surprise people:

- **`kernels.scale`** is `(in_tile, out_tile, factor, n)` where `factor` is a **1-element
  buffer** (`np.ndarray[(1,), np.dtype[np.int32]]`), not a plain Python scalar. Feed it from a
  small `depth=1` ObjectFifo or a `Buffer` — the upside is the factor becomes settable per
  launch without a rebuild.
- **`kernels.add` / `kernels.mul` / `kernels.relu`** are bare `(in..., out)` with no trailing
  size, which is why they need `pass_size_to_kernel=False`. `kernels.passthrough` *does* take a
  trailing size and wants the default `True`.

When in doubt, ask the kernel. `arg_types` is a **method**, not a property — call it:

```python
>>> kernels.scale(1024, np.int32).arg_types()
[ndarray[(1024,), int32], ndarray[(1024,), int32], ndarray[(1,), int32], numpy.int32]
```

Unsupported dtypes fail loudly at construction rather than producing a wrong kernel, e.g.
`kernels.add(tile_size=1024, dtype=np.int8)` raises
`ValueError: add() dtype must be bfloat16 ... Only the bf16 variant is available in the
installed aie_kernels`. That's your signal to hand-write the kernel.

Two template arguments trip people up:

- **`pass_size_to_kernel`** appends `tile_size` as a trailing `int` argument to every kernel
  call. It defaults to `True`, which is right for kernels with a `(in, out, n)` signature —
  set it to `False` for bare `(in, out)` kernels like `kernels.add`/`kernels.mul`. A mismatch
  here surfaces as an MLIR verification error about argument count, not a runtime bug.
- **`num_channels=2`** drives both shim DMA channels per column (one worker per
  column×channel), which roughly doubles DDR throughput on bandwidth-bound element-wise work.
  It is not compatible with shared tensor `*params`.

## The `@iron.jit` signature: `In` / `Out` / `InOut` / `CompileTime[T]`

Modern designs declare their host-visible tensors and their compile-time specialization
directly in the function signature, rather than closing over module-level constants:

```python
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out, kernels
from aie.iron.algorithms import transform_parallel

@iron.jit
def my_relu(
    a_in: In,                                  # host → NPU tensor argument
    c_out: Out,                                # NPU → host tensor argument
    *,
    size: CompileTime[int] = 65536,            # baked into the compiled design
    tile: CompileTime[int] = 1024,
):
    return transform_parallel(
        kernels.relu(tile_size=tile),
        np.ndarray[(size,), np.dtype[bfloat16]],
        tile_size=tile,
        pass_size_to_kernel=False,
    )
```

- `In` / `Out` / `InOut` annotate the **tensor** parameters. They tell the JIT which
  direction each buffer moves so it can emit the right `fill`/`drain`.
- `CompileTime[T]` parameters are **specialization knobs**: each distinct value produces a
  distinct compiled design, and the value is available as a plain Python value inside the
  body (so you can use it in `np.ndarray[(size,), ...]`). Pass them at the call site as
  keyword arguments — `my_relu(a, c, size=8192)` — or freeze them at decoration time with
  `@iron.jit(size=8192)`.
- Keep `CompileTime` parameters **keyword-only** (after the `*`). An unannotated non-tensor
  parameter *with a default* is rejected at decoration time on purpose: there is no plumbing
  for runtime scalar arguments, so the default would be silently baked in and any per-call
  override ignored. The error tells you to annotate it.

`@iron.jit` also accepts config keys alongside compile-time values:
`use_cache` (default `True`), `source_files`, `object_files`, `compile_flags`,
`include_paths`, `aiecc_flags`, `trace_config`, `full_elf`. Anything that isn't one of those is matched
against a `CompileTime[T]` parameter name, and a typo raises `TypeError` at decoration time
rather than running with an unbound value.

## Worked example: multi-core bf16 element-wise add, no C++

Adapted from `programming_examples/ml/eltwise/eltwise.py`.

```python
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, Out, kernels
from aie.iron.algorithms import transform_parallel_binary
from aie.utils.verify import assert_pass


@iron.jit
def eltwise_add(
    a_in: In,
    b_in: In,
    c_out: Out,
    *,
    size: CompileTime[int] = 65536,
    num_channels: CompileTime[int] = 1,
):
    return transform_parallel_binary(
        kernels.add(tile_size=1024),
        np.ndarray[(size,), np.dtype[bfloat16]],
        tile_size=1024,
        num_channels=num_channels,
        pass_size_to_kernel=False,      # kernels.add is (in0, in1, out)
    )


if __name__ == "__main__":
    N = 65536
    rng = np.random.default_rng(0)
    a_np = rng.uniform(-1.0, 1.0, size=(N,)).astype(bfloat16)
    b_np = rng.uniform(-1.0, 1.0, size=(N,)).astype(bfloat16)

    a = iron.tensor(a_np, dtype=bfloat16, device="npu")
    b = iron.tensor(b_np, dtype=bfloat16, device="npu")
    c = iron.zeros_like(a)

    eltwise_add(a, b, c, size=N)

    expected = (a_np.astype(np.float32) + b_np.astype(np.float32)).astype(bfloat16)
    assert_pass(c.numpy(), expected, atol=0.00390625, fail_msg="eltwise_add mismatch")
```

That is the whole design — no `ObjectFifo`, no `Worker`, no `Runtime`, no `.cc` file.
Drop to the explicit `ObjectFifo`/`Worker`/`Runtime` form in
[`patterns.md`](patterns.md) when the topology stops being "same op over every tile":
asymmetric fan-out, multi-stage pipelines, cascades, custom placement, RTP-gated cores.

## MMUL geometry without guessing: `.mac_dims`

`kernels.mm(...)` (and `cascade_mm`) expose the micro-kernel geometry the freshly-compiled
kernel actually expects, so you can compute tile shapes and `dims_to_stream` layouts from
the kernel rather than from a table that may drift:

```python
mm_k = kernels.mm(dim_m=64, dim_k=64, dim_n=64,
                  input_dtype=bfloat16, output_dtype=np.float32)
r, s, t = mm_k.mac_dims          # per-arch (r, s, t) for this dtype combo
mm_zero = mm_k.zero              # companion zero-fill kernel for the accumulator
```

Prefer `.mac_dims` over hardcoding `(4, 8, 8)`: the value differs between AIE2 and AIE2P and
between dtype combinations, and `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` changes it again
on AIE2P. The static table in [`architecture.md`](architecture.md) is for reasoning about
divisibility constraints; `.mac_dims` is what you should actually feed into code.

Note that `cascade_mm`'s kernel is **scalar** on both AIE2 and AIE2P, so its `mac_dims` are
`(1, 1, 1)` and its L2→L1 buffers must stay plain row-major. Feeding it a tiled layout
produces silently wrong output, not an error.

## When to stop using the library

Write your own kernel (and your own topology) when:

- the dtype or tile size you need isn't in the supported set above — e.g. `kernels.add` is
  bf16-only with a fixed 1024 tile, so an int8 add needs a hand-written kernel;
- the op fuses several stages that the library exposes only separately, and the intermediate
  would otherwise round-trip through L1 for no reason;
- the access pattern is a stencil/sliding window or anything where the "one tile in, one tile
  out" assumption of `transform*` doesn't hold (check the `conv_pipeline` templates first);
- you are optimizing a kernel that profiling has already identified as the bottleneck — at
  which point see the `aie-kernel-opt` skill rather than starting from scratch.
