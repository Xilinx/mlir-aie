<!-- Copyright (C) 2024-2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Python Kernel Library

Pre-built AIE kernel wrappers for common operations. These provide ready-to-use
`Worker`-compatible callables backed by optimized native AIE code. For the C++
kernel sources these wrap, see [C++ AIE kernels](aie_kernels.md).

## Element-wise operations

::: iron.kernels.eltwise
    options:
      show_root_heading: false

## Data movement

::: iron.kernels.datamovement
    options:
      show_root_heading: false

::: iron.kernels.zero
    options:
      show_root_heading: false

## Quantization

::: iron.kernels.quant
    options:
      show_root_heading: false

## Core state

::: iron.kernels.core
    options:
      show_root_heading: false

## Reduction

::: iron.kernels.reduce
    options:
      show_root_heading: false

## Linear algebra

::: iron.kernels.linalg
    options:
      show_root_heading: false

## Convolution

::: iron.kernels.conv
    options:
      show_root_heading: false

## Activation functions

::: iron.kernels.activation
    options:
      show_root_heading: false

## Normalization

`rms_norm`, `rms_norm_eps`, and `layer_norm` support both aie2 and aie2p.
They accept `tile_size` (default 1024) or its compatibility alias `cols`.
`rope` in Data movement has the same size API and supports both interleaved
and `two_halves=True` layouts. The transformer module re-exports the canonical
bf16 norm and RoPE factories and references; importing through either module
does not select a different implementation. Norm references accept `eps`.

::: iron.kernels.norm
    options:
      show_root_heading: false

## Transformer blocks

::: iron.kernels.transformer
    options:
      show_root_heading: false

## Vision

::: iron.kernels.vision
    options:
      show_root_heading: false

## Contracts and the generic design builder

Every factory the generic builder can build carries a `KernelContract` on the
returned `ExternalFunction` (`fn.contract`). The field-by-field account is in
[Kernel Library](../programming_guide/kernels_library.md#what-a-signature-cannot-say);
the reference below is generated from the dataclass, so it cannot drift from
it. Multi-dtype factories publish the combinations they support as
`factory.dtypes`, and the factories above export their references as `*_ref`
functions (`add_ref`, `reduce_max_ref`, `mm_ref`, ...), so host code never
reimplements the math.

::: iron.kernels._common
    options:
      show_root_heading: false
      members:
        - KernelContract
        - TensorLayout
        - Param
        - Trace

`aie.iron.algorithms.kernel_design` turns any contract-bearing factory into a
design of one Worker, built on the same single-core pipeline as
`transform`, `for_each` and `reduce`. What a kernel can answer about itself
-- its
reference result, its safe input range, which arguments are parameters,
how to judge a device output -- lives on
[`ExternalFunction`][iron.ExternalFunction] instead, so bringing up a
kernel needs no test harness. See
[Kernel Library](../programming_guide/kernels_library.md#adding-a-kernel) for
the add-a-kernel procedure and the test tiers built on it.

Import the builder with `from aie.iron.algorithms import kernel_design as kd`
and call `kd.design(...)`, or import `design` directly from `aie.iron.algorithms`.
The former `aie.utils.kernel_harness` module has been removed.

::: iron.algorithms.kernel_design
    options:
      show_root_heading: false
      members:
        - design
        - host_args
        - sample_inputs
        - host_layout
        - output_size
        - upload
        - cycles_per_call
        - CallCycles
        - traced_intervals
        - split_intervals


`test/python/npu/test_kernels_perf.py` times the library: correctness
first, then cycles, wall time and build size, gated by a device preflight
and a measurement-sanity test. It is an ordinary pytest module, so `-k`
selects cases and the session's exit status decides whether any numbers are
written. `--baseline-sources DIR` measures every selected case a second
time with its kernels from `DIR` and compares the two runs' raw output words
and cycles in `--perf-meta`. The timing helpers live here:

::: utils.benchmark
    options:
      show_root_heading: false
      members:
        - Stats
        - BenchmarkResult
        - Preflight
        - run_iters
        - preflight
        - provenance
        - kernel_tree_digest

## Static checks

These compiler-remark checks are available on demand through
`python -m aie.utils.compile.remarks`; there is no static-check CI workflow.
The trace-marker check (`test/python/test_kernel_trace_markers.py`) runs in
lit on every PR, through `trace_markers` below.

::: utils.compile.remarks
    options:
      show_root_heading: false
      members:
        - LoopInfo
        - StaticReport
        - parse_yaml
        - parse_stderr
        - report_rows
        - workflow_annotations
        - compile_command
        - analyze
        - kernel_builds
        - Linked
        - linked
        - parse_readobj
        - trace_markers
        - trace_shape
        - entry_symbol

## Host-side helpers

`aie.utils.bfp` is the host side of the block-floating-point kernels: the
bfp16ebs8 codec and the tile shuffle the `mm_bfp` DMA layout needs. It is
the Python counterpart of `programming_examples/ml/block_datatypes/helper.h`,
which the examples' C++ hosts use; a Python host encodes and checks with
this module.

::: utils.bfp
    options:
      show_root_heading: false
      members:
        - encode
        - decode
        - quantize
        - shuffle

::: utils.verify
    options:
      show_root_heading: false
      members:
        - Tolerance
        - Verdict
        - compare
        - bf16_ulp_distance
