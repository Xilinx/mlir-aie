<!---//===- kernels_library.md -----------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

    def sequence(a_in, b_out, in_h, out_h):
        in_h.fill(a_in)
        out_h.drain(b_out, wait=True)

    rt = Runtime(sequence, [line_ty, line_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()
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
| [`kernels.eltwise`](../python/iron/kernels/eltwise.py)       | element-wise: passthrough, scale, add, mul, mul_add, relu |
| [`kernels.reduce`](../python/iron/kernels/reduce.py)         | reductions: reduce_add, reduce_min, reduce_max, compute_max |
| [`kernels.activation`](../python/iron/kernels/activation.py) | activations: softmax, tanh, sigmoid, gelu, silu, swiglu, leaky_relu, bf16_exp, exp2f_vec |
| [`kernels.datamovement`](../python/iron/kernels/datamovement.py) | data movement and conversion: axpy, convert_copy, expand, transpose |
| [`kernels.linalg`](../python/iron/kernels/linalg.py)         | linear algebra: mm (+ `.zero`, `.mac_dims`, `.stream_dims`), mv (int16 + `.zero`, bf16), cascade_mm (+ `.{get_only,put_only,put_get,zero}`, `.mac_dims`), mm_bfp (+ `.zero`), mm_bfp_shuffle, mha (+ the flash-attention siblings) |
| [`kernels.conv`](../python/iron/kernels/conv.py)             | convolutions: conv2dk1/3/14, conv2dk1_skip(_init), dwconv1d, bn_* bottleneck variants for MobileNet/ResNet |
| [`kernels.transformer`](../python/iron/kernels/transformer.py) | transformer blocks: rms_norm, layer_norm (bf16, f32, affine + cast), rope, mm_activation_epilogue |
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

Every module also exports a numpy reference per kernel (`add_ref`,
`reduce_max_ref`, `mm_ref`, `softmax_ref`, ...), and most factories attach
a `KernelContract` to the function they return:

```python
fn = kernels.reduce_max(dtype=np.int32, tile_size=1024)
fn.contract.roles        # ('in', 'out', 'count')
fn.contract.reference    # kernels.reduce_max_ref
fn.contract.tolerance.kind  # 'exact'
fn.contract.ops_per_call # 1024

mm = kernels.mm(input_dtype=np.int16, output_dtype=np.int32)
mm.contract.acc_dtype    # numpy.int64: accauto is acc64 for int16
mm.contract.reduction    # 64: products summed per output per call (dim_k)
mm.contract.overflow     # 'undefined': to_vector() in the core's default mode
mm.contract.rounding     # 'unspecified'
kernels.mm.dtypes        # every (input_dtype, output_dtype) the factory supports
```

The contract is what lets `aie.utils.kernel_harness` build, run and check
a kernel without a hand-written design:

```python
from aie.utils import kernel_harness as kh

verdict = kh.check(kernels.reduce_max, calls=16, dtype=np.int32)
assert verdict, verdict.detail
```

Tolerances are kernel-owned. Integer kernels and lossless copies are
bit-exact; LUT approximations declare the `rtol` their reference
documents; a factory that declares nothing is judged with
`Tolerance.default_for(out_dtype)`: exact for integers, `1e-4` for
float32, `1e-2` for float16 and the canonical `0.128` for bf16.

### What a signature cannot say

A contract also declares the dtype facts an `arg_types` list leaves out:

- `acc_dtype` and `reduction`: what the kernel accumulates in and over how
  many terms. `aie.utils.kernel_harness.input_limit` turns them into the
  largest integer input that cannot overflow the accumulator, which is
  where the harness and the benchmark registry draw their data.
- `overflow`: whether an integer result outside the output range wraps,
  saturates, or is undefined (a `to_vector()` in the core's default
  mode). The judge clips or wraps the reference to match, and under
  `undefined` refuses to grade an overflowing reference at all.
- `rounding`: how a narrowing rounds (`floor`, `nearest`, `nearest_even`,
  or `unspecified` for an `srs` in the core's default mode, which is why
  some kernels allow one LSB).
- `nonfinite` and `subnormals`: whether NaN / inf propagate and whether
  subnormal inputs are preserved, flushed (the judge then compares them
  as zero) or unspecified. The benchmark registry derives each kernel's
  edge-data cases from these, so widening what a kernel is fed is a
  contract change.
- `unsupported`: why the generic harness cannot run this kernel, when it
  cannot; the reference still says what the kernel computes.
- `.dtypes` on the factory: the dtype combinations it builds. The host
  contract test builds every entry.

## Standing in for a hand-built kernel (amd/IRON)

An operator that builds a kernel by hand needs the exported **symbol**,
the **source file** and the **compile flags** to line up. Every factory
publishes all three (`fn.name`, `fn.source_file`, `fn.compile_flags`), so
a factory call replaces a hand-written source-plus-symbol pair wherever
the two trees share the kernel.
`test_factories_reproduce_the_iron_operator_kernel_specs` in
`test/python/test_kernel_contracts.py` pins that table for the kernels
[amd/IRON](https://github.com/amd/iron)'s operators build: `saxpy`,
`gelu_bf16`, `silu_bf16`, `sigmoid_bf16`, `tanh_bf16`, `softmax_bf16`,
`eltwise_add_bf16_vector`, `eltwise_mul_bf16_vector`, `layer_norm`,
`rope`, `passThroughLine`, `expand_uint4_to_bfloat16`, `transpose_4x4`,
`matvec_vectorized_bf16_bf16` and the `matmul_*` family with its
`-DDIM_*`, dtype-`ONLY`, `-DB_COL_MAJ` / `-DC_COL_MAJ` and
`AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` flags. Renaming a symbol or
dropping a flag fails that test rather than silently breaking a
downstream build.

Three things make this work for more than one kernel per design:

- **`inout`.** A kernel that accumulates into its output declares that
  argument `inout` rather than `out`: `mm` computes `C += A * B`, reads C
  back, ships a `.zero` sibling, and a design zeroes the buffer before
  the first call. The reference still computes the whole product, so an
  `inout` output is excluded from `reference_indices` like an `out` one;
  `contract.accumulates` says which kind a kernel is. The int16 `mv` and
  `cascade_mm` accumulate the same way; the bf16 `mv` stores.
- **Whole-object symbol prefixing.** Each parameterisation of a kernel
  gets its own symbol prefix, and every symbol its object defines is
  prefixed, not just the declared one. A translation unit usually
  exports more (`mm.cc` emits the `zero_*` that `.zero` binds; `mha.cc`
  includes `mm.cc` and defines `matmul_*` names of its own), and leaving
  those bare made two parameterisations collide at link.
  `ExternalFunction.sibling(symbol, arg_types)` binds another symbol
  from the same object with the prefix applied; that is how `.zero`, the
  cascade trio and `mha`'s flash-attention siblings are built. Chess-built
  kernels are the exception: `llvm-objcopy` corrupts xchesscc objects, so
  they keep bare symbols and only one variant may appear in a design.
- **`host_args`.** `aie.utils.kernel_harness.host_args(fn, calls=, shape=)`
  returns one `HostArg` (direction, shape, dtype) per host buffer the
  design takes, in the layout the device expects: B transposed for a
  `b_col_maj` matmul, C transposed for `c_col_maj`, encoded bytes for a
  bfp16ebs8 operand, interleaved tiles where streamed tensors share one
  fifo, a reduction's DMA padding. `param` arguments are absent, since
  they are baked into the design. A caller that only needs to size
  buffers reads this instead of running the sampler.

Where the kernel *sources* have diverged, no factory can stand in:

| Kernel | Why a factory cannot substitute |
| --- | --- |
| `relu` | IRON's `relu.cc` exports `relu_bf16`; this tree's exports `bf16_relu`. |
| `rms_norm` | IRON's exports `rms_norm_bf16_vector` and `weighted_rms_norm`; this tree's exports `rms_norm`. |
| `mm` with `-DROUND_CONV_EVEN` | The flag exists only in IRON's `mm.cc`; passing it here would be a no-op, so the factory does not offer it. |

Two factories are drop-ins even though the harness cannot run them.
`kernels.mha()` compiles `aie_kernels/aie2p/mha.cc` once and binds its
ten symbols: the returned kernel is the `QK^T` matmul, with `.zero`,
`.matmul_rowmaj`, `.matmul_scalar`, `.partial_softmax`, `.matmul_pv`,
`.rescale_o` and `.init_scale_buffer` beside it (`passThroughLine` is
only declared there; take it from `passthrough(dtype=np.int32)`, as
IRON's MHA operator does). The bf16 `mv` builds the shared
`aie_kernels/generic/mv.cc` with its `(m, row_offset, A, b, c)`
signature, `DIM_K` and `VEC_SIZE`. Both contracts declare `unsupported`
(attention is a multi-core dataflow; the matvec design drives the int16
signature); what an operator needs is the object and the binding, and
both provide that.

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

### Adding a kernel

A new factory is complete when one line each in two places covers it:

1. **Contract.** Pass `contract=KernelContract(...)` to `_make_extern`
   with the argument roles (`in`, `out` or `inout`, `param`, `count`,
   `scalar`), a
   numpy reference exported as `<name>_ref`, `ops_per_call` for the
   benchmark's throughput series, and a `Tolerance` with its evidence in
   `note` — or none, to get the dtype default. Reductions set `out_valid`
   to the number of meaningful output elements. Say what the kernel
   accumulates in (`acc_dtype`, `reduction`), what an integer overflow
   does (`overflow`) and how a fixed-point shift rounds (`rounding`),
   from the C++ rather than from a guess; a factory with more than one
   dtype lists them in a `.dtypes` table.
2. **Case.** Add the shapes to time and the edge data the kernel must
   survive as one `Case(...)` in
   [`benchmarks/kernels/registry.py`](../benchmarks/kernels/registry.py),
   and one entry in
   [`test/python/npu/test_kernels_e2e.py`](../test/python/npu/test_kernels_e2e.py)
   for the per-PR device smoke test.

The host test [`test/python/test_kernel_contracts.py`](../test/python/test_kernel_contracts.py)
then checks the roles against the real `arg_types()`, the reference's
arity, and that the harness design lowers to MLIR; the nightly static
checker (`benchmarks/static`) reports per-loop II, zero-overhead-loop
status and missing-bank loads from Peano's remarks, which is where to
look when optimizing. See
[`benchmarks/kernels/README.md`](../benchmarks/kernels/README.md) for the
test tiers.

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
