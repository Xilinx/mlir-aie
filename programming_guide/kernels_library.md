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

* The source path (e.g. `aie_kernels/linalg/mm.cc`).
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

## Compose kernels explicitly

Use the same zero factory for any accumulator, independently of the operation
that follows it:

```python
matmul = kernels.mm(
    dim_m=m, dim_k=k, dim_n=n,
    input_dtype=np.int16,
    output_dtype=np.int16,
)
zero_kernel = kernels.zero(tile_size=m * n, dtype=np.int16)
```

Zeroing compiles only the zero kernel, not another copy of the matrix kernel.
The generic test builder uses this factory through each accumulating kernel's
declared initializer. Explicit designs choose when to zero, allowing repeated
accumulation into the same tile.

Output-only kernels need no dummy inputs: `zero_kernel.expected([])` gives one
tile, and `zero_kernel.judge(actual, reference, calls=n)` checks every repeated
tile against it. Streamed-input references must still describe all calls.

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
| [`kernels.zero`](../python/iron/kernels/zero.py) | target-vectorized zero fill, including partial vectors and packed BFP blocks |
| [`kernels.quant`](../python/iron/kernels/quant.py) | q4nx dequantization to GEMM-ordered bfp16ebs8 (AIE2P), with byte-exact verification |
| [`kernels.linalg`](../python/iron/kernels/linalg.py)         | linear algebra: mm, mv, cascade_mm, mm_bfp, mm_bfp_shuffle, mha |
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

Factories attach a `KernelContract` to the function they return. Use
`fn.expected(inputs)` for the configured reference calculation:

```python
fn = kernels.reduce_max(dtype=np.int32, tile_size=1024)
fn.contract.roles        # (In, Out, Param): marker classes, not strings
fn.contract.parameter_bindings  # ((2, 1024),): the fixed element count
fn.contract.reference    # kernels.reduce_max_ref
fn.contract.tolerance    # exact integer comparison
fn.contract.ops_per_call # 1024

mm = kernels.mm(input_dtype=np.int16, output_dtype=np.int32)
mm.contract.acc_dtype    # numpy.int64: accauto is acc64 for int16
mm.contract.reduction    # 64: products summed per output per call (dim_k)
kernels.mm.dtypes        # every (input_dtype, output_dtype) the factory supports
```

The contract is what lets `aie.iron.algorithms.kernel_design` build, run and check
a kernel without a hand-written design:

```python
from aie.iron.algorithms import kernel_design as kd

fn = kernels.reduce_max(dtype=np.int32)
design = kd.design(kernels.reduce_max, calls=16, dtype=np.int32)
inputs = kd.sample_inputs(fn, calls=16)
ins, out = kd.upload(
    inputs, kd.output_size(fn, calls=16), fn.output_dtype(np.int32), fn=fn, poison=True
)
design(*ins, out)
verdict = fn.judge(out.numpy().copy(), fn.expected(inputs), calls=16)
assert verdict, verdict.detail
```

`kd.design(..., guard=True)` also catches writes past an output. Each output
tile gets `kd.GUARD_BYTES` of `0x55` after it, which the kernel never sees.
Size the output with `kd.output_size(fn, calls=16, guard=True)` and split the
result with `kd.strip_guard(fn, out.numpy(), calls=16)`, which returns the
data and the number of changed guard bytes. bfp outputs are not guarded.

Every design validates independent tile calls, including matrix kernels.
`kd.design(kernels.mm, calls=16)` repeats sixteen tile products, initializing
each output before its call. The factory's dimensions size a tile; `calls`
does not change its reduction length. Whole-problem `shape=` is rejected:
algorithm integration tests own global iteration and accumulation.

`contract.layouts` declares a `TensorLayout` per argument: its logical tile
shape and reversible host storage codec. The same builder handles row-major,
blocked, transposed and block-floating-point tiles without recognizing a
kernel's name or kind. Matrix contracts use the independent-call `*_tile_ref`
references; whole-matrix `mm_ref` and `mv_ref` remain available to algorithms.

`In`, `Out`, and `InOut` reuse the JIT's tensor direction markers.
The kernel-only `Param` means a read-only value fixed across the builder's
independent calls. Its existing argument type determines whether it is a scalar
operand or a tensor initialized in a core buffer; there is no count-specific role.
Fixed values, including counts, belong in `parameter_bindings`; unbound scalar
values come from `design(..., scalars=...)`, and tensor values from `params=`.
These values are embedded in the compiled design: changing them recompiles it.
This describes the validation builder, not an inherent lifetime restriction on
the C++ argument; a hand-written design can supply a different operand per call
when the kernel permits it. Factories that specialize loop bounds compile their
element counts and convolution dimensions into the kernel object. Their legacy
count/dimension operands remain for ABI compatibility but do not resize the
operation: construct another factory configuration to change those bounds.
Other operands, such as scales and convolution boundary selectors, remain
runtime values. Compiling the raw sources without the corresponding `*_ELEMS`
or `CONV_*` defines retains their runtime bounds.
`Out` is written, while `InOut`
is read and written and requires a declared `initializers` entry. Its reference
describes the result from that initial state. Initialization occurs on every
independent call, not only the first.

Multiple outputs are ordered by `contract.out_indices`. Their reference,
output sizes and device dtypes are tuples; `output_dtype()` derives the latter
from declarations. `judge` returns one aggregate verdict, false if any output
fails, with per-output diagnostics. `upload` returns a tuple of output tensors,
passed as `design(*inputs, *outputs)`. The builder respects the target's available
output DMA channels.

Tolerances are kernel-owned. Integer kernels and lossless copies are
bit-exact; LUT approximations declare the `rtol` their reference
documents; a factory that declares nothing is judged with
`Tolerance.default_for(out_dtype)`: exact for integers, `1e-4` for
float32, `1e-2` for float16 and the canonical `0.128` for bf16.

### What a signature cannot say

A contract also declares the dtype facts an `arg_types` list leaves out:

- `acc_dtype` and `reduction`: what the kernel accumulates in and over how
  many terms. `fn.input_limit(dtype)` turns them into the largest integer
  input that cannot overflow the accumulator, which is where the builder
  and the case table draw their data.
- `setup`: a kernel to run once on the core first, when this one needs it;
  see [Rounding mode](#rounding-mode) below. The rounding setter is compiled
  as always-inline LLVM IR and merged into the core, avoiding an external
  function call; the setter therefore uses Peano, not Chess.
- `stack_bytes`: the core stack a Worker calling this kernel needs, when
  that is more than the target's default.
- `unsupported`: why the generic builder cannot run this kernel, when it
  cannot; the reference still says what the kernel computes.

What the kernel does on overflow, how a narrowing store rounds, and what
it does with NaN or subnormal inputs are *not* declared. The numpy
`reference` is the arithmetic model -- a saturating kernel's reference
clips, a denormal-flushing one flushes -- and `tolerance` is the slack
allowed against it. Declaring the same fact in two places let the two
drift.
- `.dtypes` on the factory: the dtype combinations it builds. The host
  contract test builds every entry.

## Standing in for a hand-built kernel

Code that builds a kernel by hand needs the exported **symbol**, the
**source file** and the **compile flags** to line up. Every factory
publishes all three (`fn.name`, `fn.source_file`, `fn.compile_flags`), so a
factory call can replace a hand-written source-plus-symbol pair.
`test_factories_reproduce_the_iron_operator_kernel_specs` in
`test/python/test_kernel_contracts.py` pins that table, so renaming a symbol
or dropping a flag fails a test here rather than silently breaking a build
elsewhere.

Three things make this work for more than one kernel per design:

- **`InOut`.** A kernel that accumulates into its output declares that
  argument `InOut` rather than `Out`: `mm` computes `C += A * B`, reads C
  back, and a design calls `kernels.zero(...)` before
  the first call. The reference still computes the whole product, so an
  `InOut` output is excluded from `reference_indices` like an `Out` one;
  `contract.accumulates` says which kind a kernel is. The int16 `mv` and
  `cascade_mm` accumulate the same way; the bf16 `mv` stores.
- **Whole-object symbol prefixing.** Each parameterization of a kernel
  gets its own symbol prefix, and every symbol its object defines is
  prefixed, not just the declared one. A translation unit usually
  exports more (`mha.cc`
  includes `mm.cc` and defines `matmul_*` names of its own), and leaving
  those bare made two parameterizations collide at link.
  `fn.object_file.bind(symbol, arg_types)` binds another symbol
  from the same object with the prefix applied. Chess-built
  kernels are the exception: `llvm-objcopy` corrupts xchesscc objects, so
  they keep bare symbols and only one variant may appear in a design.
- **`host_args`.** `kernel_design.host_args(fn, calls=)`
  reports the direction, shape, dtype and element count of each host buffer the
  design takes, in the layout the device expects: packed tiles from each
  argument's layout codec, encoded bytes for a bfp16ebs8 operand,
  interleaved tiles where streamed tensors share one
  fifo, a reduction's DMA padding. `Param` arguments are absent, since
  they are baked into the design. A caller that only needs to size
  buffers reads this instead of running the sampler.

`kernels.mha()` compiles `aie_kernels/linalg/mha.cc` once and binds its
selected entry point, the `QK^T` product: `mm.cc`'s bf16 tile matmul with
its index gate bound open, validated by the generic builder like `mm`. The
other symbols of the translation unit bind from the same object. The bf16
`mv` binds parameters `(m, row_offset)` and is validated the same way. For
the kernels that need more than one tile, see
[Kernels the generic builder cannot run](#kernels-the-generic-builder-cannot-run).

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
   with the argument roles (`In`, `Out`, `InOut`, or `Param`), a
   numpy reference exported as `<name>_ref`, `ops_per_call` for the
   performance checks' throughput series, and a `Tolerance` with its evidence in
   `note` — or none, to get the dtype default. Reductions set `out_valid`
   to the number of meaningful output elements. Say what the kernel
   accumulates in (`acc_dtype`, `reduction`), and model overflow and
   rounding in the reference from the C++ rather than from a guess.
   A factory with more than one dtype lists them in a `.dtypes` table.
   Bind fixed parameters explicitly with `parameter_bindings`, publish nontrivial
   storage with `layouts`, and initialize `InOut` tiles with `initializers`.
   Declare what the kernel's trace markers measure with `trace=`:
   `Trace.whole_call()` when one `event0()` before the work and one
   `event1()` after it bracket every call of the entry symbol, or
   `Trace.none(reason)` / `Trace.partial(reason)` when they do not. A
   kernel the performance checks time must be `whole_call`.
2. **Case.** Add one `Case(...)` to
   [`test/python/npu/kernel_cases.py`](../test/python/npu/kernel_cases.py):
   the shape to run and, with `smoke=True`, that it is the kernel's
   representative shape for the per-PR device test. The same table drives
   the nightly correctness sweep and the performance checks, so there is nothing
   else to register.

The host test [`test/python/test_kernel_contracts.py`](../test/python/test_kernel_contracts.py)
then checks the roles against the real `arg_types()`, the reference's
arity, that the generated design lowers to MLIR, and that `setup`
agrees with the source.
[`test/python/test_kernel_trace_markers.py`](../test/python/test_kernel_trace_markers.py)
compiles every build to optimized IR and checks the markers the entry
symbol reaches against `trace=`. Markers in a sibling kernel of the same
file, around an inner loop, or skipped by an early return do not count as
`whole_call`.

## Testing, performance and static checks

Every tier below reads the contract and the case table; none restates
what a kernel computes.

| Tier | What | Where | When |
| --- | --- | --- | --- |
| host | contract vs. factory; design lowers to MLIR | `test/python/test_kernel_contracts.py` | every PR (lit) |
| host, Peano | trace markers vs. the contract's `trace` | `test/python/test_kernel_trace_markers.py` | every PR (lit) |
| device, smoke | the `smoke` cases on random data | `test/python/npu/test_kernels_e2e.py` | every PR on the NPU runners |
| device, full | every case, every edge-data case, `--seeds` seeds | the same file, `-m extensive` | nightly, before anything is timed |
| host, static | Peano remarks per kernel build | `python -m aie.utils.compile.remarks` | on demand |

There is no compile-only tier: a design that will not build fails the tier
that runs it. The smoke run builds every factory's design on both
architectures each PR, and the nightly builds every case's.

```bash
pytest test/python/test_kernel_contracts.py                        # host
pytest test/python/npu/test_kernels_e2e.py -k eltwise              # NPU, smoke
pytest test/python/npu/test_kernels_e2e.py -m extensive --seeds 3  # NPU, everything
pytest test/python/npu/test_kernels_perf.py -m perf -k mul          # time one kernel
pytest test/python/npu/test_kernels_perf.py -m perf --perf-out perf.json
python -m aie.utils.compile.remarks --target aie2p --out static.json
```

To measure a kernel change against the code before it, point
`--baseline-sources` at a second checkout (any directory holding
`aie_kernels/` and `aie_runtime_lib/`). The static checks then compile both
and print each row that differs; the performance checks run each case from both,
back to back, on the same inputs:

```bash
mkdir ../base && git archive HEAD aie_kernels aie_runtime_lib | tar -x -C ../base
python -m aie.utils.compile.remarks --target aie2p --only '^gelu' \
    --out static.json --baseline-sources ../base
pytest test/python/npu/test_kernels_perf.py -m perf -k gelu \
    --baseline-sources ../base --perf-meta meta.json
```

### Data policy

Random data is bounded by `fn.input_limit(dtype)`, using the contract's
`acc_dtype` and per-call `reduction`. A whole-design test can supply its
full reduction length explicitly. An edge case exercises the datapath rather than overflowing
the accumulator. The output range does not bound it: what a kernel does
when a result leaves that range is its reference's to model, and clipping
inputs to it would leave a requantizing kernel's data near zero.

Which edge cases a kernel is fed is a property of the case, not of the
contract. Integer kernels get the extremes; matmul operands never carry
NaN; a kernel whose reference handles NaN, inf and subnormals says so by
listing them in its case's `data_cases`, and the test then proves it.

### Rounding mode

The core narrows accumulators (an `srs` shift, a bf16 store) in whatever
mode its rounding-mode register holds, and a fresh core boots in `floor`.
The contract's `setup` names a kernel to run once on the core before the
first call. A kernel whose source calls `aie::set_rounding` itself (the
conv kernels, `layer_norm`, `mha`, the aie2p `mm`) needs none, so its
`setup` is `None`. A kernel that relies on the design to have set the
mode names the setter: the bf16 kernels that store from an fp32
accumulator use `setup=conv_even`, the mode numpy's reference rounds in.
A design calls `fn.contract.setup()` once before the kernel; the builder
does the same, so every test runs each kernel in the mode its contract was
written for.

The two are alternatives, and a test enforces it against the sources: a
kernel whose `.cc` calls `aie::set_rounding` must not also name a
`setup`, and one that names a `setup` must actually narrow something.

This is the mechanism for the convention
[#3481](https://github.com/Xilinx/mlir-aie/issues/3481) asks for (a kernel
either owns its mode or assumes the caller set one), not the whole of it:
`rms_norm`, `layer_norm` and `mv` still save and restore the register
around their body, and nothing yet boots a core into `conv_even` by
default. Both remain to do under that issue.

### What the performance checks record

`test/python/npu/test_kernels_perf.py` measures a kernel only after it has
produced a correct result under its declared tolerance; a wrong result fails
the test, and a failed session writes no `--perf-out` file at all. Per case it records core
`cycles` and `cycles_per_kop`, `npu_us` from `aie.utils.benchmark`, and
the `xclbin`, `insts` and core-ELF sizes of the build it ran.

`cycles` is recorded only for a kernel whose contract declares
`Trace.whole_call()`. The trace holds one interval per call of the kernel
and one per call of each traced initializer (`zero` before `mm`), in the
order the harness calls them, and `kd.cycles_per_call` splits it by that
position. The row is the kernel's minimum. Every call does the same work,
so anything above the minimum is the core waiting. The median, the maximum,
each initializer's minimum and whether the trace buffer filled go in the
row's `range`. The trace buffer is sized to the number of intervals the
contract declares. Preflight reads the device and its power mode through the
host runtime (`HostRuntime.power_mode()`); the nightly workflow tries to
switch to `performance` first, but always records the active mode in the
results. A bit-exact `passthrough` smoke test inside a cycle band guards
the machine. Nightly data goes to `gh-pages:kernel-checks/<npu>/` and is
graphed at `https://xilinx.github.io/mlir-aie/kernel-checks/`, whose kernels
view lists every factory with the builds each NPU offers, how its cases fared that
night and their latest numbers (`utils/kernel_checks/catalogue.py` writes the
catalogue); nothing gates a pull request. A Peano-bump PR is compared against the
cached nightly baseline by `utils/kernel_checks/pr_report.py`, which keeps one PR
comment listing failing cases and `cycles` or core ELF size regressions of 2 % or more.

### Static checks

`aie.utils.compile.remarks` compiles every factory build (defaults plus
each `.dtypes` entry) exactly as the JIT does, with Peano's
optimization-record flags, and turns the records into per-kernel series:
each loop's II and whether it is a zero-overhead loop, program memory,
missing-bank loads and dropped `#pragma`s. It then reads the object: the
loop counts and program memory cover only the functions the entry symbol
reaches, which are the ones the core link keeps, and `libcalls` names the
runtime-library routines it calls (`__divsf3`, `__mulsf3`, `__floatsisf`:
on AIE2P, scalar float divide, multiply and int-to-float are software
routines). `kernel_stack_bytes` is the deepest call path's frames from the
entry, without those routines' own and without the core's `main`, which
aiecc's measured stack also counts; above the contract's `stack_bytes`
(else the target default) it prints a warning, since an overflow corrupts
the neighbouring memory silently. Each build prints its entry symbol and
source file, and `--meta` names its object (kept with `--keep DIR`). The record shapes and the
regression rules are documented on the module
([API](../api/kernels.md#static-checks)). These checks run on demand; there
is no static-check CI workflow. When invoked in GitHub Actions, the tool
emits a warning annotation for a dropped pragma and an error annotation
for a kernel that fails to compile. With
`MLIR_AIE_KERNEL_SOURCES` set to a checkout, the checkout's
`aie_kernels/` is compiled against an installed wheel. The separate
`nightlyKernelChecks.yml` workflow runs the hardware correctness and
performance checks nightly and on Peano-pin pull requests; it does not run these
static checks.

### Kernels the generic builder cannot run

The builder runs one kernel on one Worker. A cascade pair is two: the PUT
half of `cascade_mm` (`cascade_mm_put`) streams its product onto the cascade
and the GET half adds its own product and the cascade term, so both
contracts say `unsupported` and `test/python/npu/test_kernels_e2e.py`
builds the pair by hand and judges it against the two products.
`set_rounding` has a contract but no data output: it is a setup operation,
covered by the rounding-mode tests above.

The MobileNet bottleneck kernels (`bn_*`) are exported for the
[`mobilenet`](../programming_examples/ml/mobilenet) examples. The single-core
ones carry contracts with round-half-even integer references and run in the
hardware sweeps at MobileNet V3 layer shapes; `bn_conv2dk1_relu_xy_pool_padded`
accumulates into its output across calls, so its case zeroes that buffer
first. The cascade halves (`bn_conv2dk1_partial_*` and
`bn_conv2dk1_input_split_partial_*`) exist as one symbol per network block;
`test/python/npu/test_bn_cascade_pairs.py` builds each pair the way the
MobileNet cascade block calls it and judges it against a numpy model of
the whole conv, and the contract test lists them by name as not judged.

`mm_bfp_shuffle` validates the forward permutation through declared plain-BFP
input and blocked-BFP output codecs, comparing exactly the represented values.
The default equal-sized buffers are supported; custom unequal buffer extents
still require the enclosing design's runtime dimensions. Its direct-call ABI
continues to accept either shuffle direction.

`q4nx_dequant` validates one packed q4nx block on AIE2P. Its input contains
bf16 scales and minima followed by unsigned four-bit codes; its output is
the GEMM-ordered bfp16ebs8 byte stream. Both are exposed as byte buffers so
the harness checks exponents, mantissas and ordering exactly, including the
kernel's floor-rounded bf16 intermediate. The default block and two smaller
geometries participate in the compile and extensive hardware sweeps; the
default also runs as a hardware smoke test and performance check.

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
