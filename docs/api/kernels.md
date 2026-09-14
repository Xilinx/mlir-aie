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

## Transformer blocks

::: iron.kernels.transformer
    options:
      show_root_heading: false

## Vision

::: iron.kernels.vision
    options:
      show_root_heading: false

## Contracts and the test harness

Every factory that the generic harness can build carries a `KernelContract`
on the returned `ExternalFunction` (`fn.contract`): argument roles, a numpy
reference, the tolerance the kernel is held to, its work per call, what it
accumulates in and over how many terms, and its integer overflow and
fixed-point rounding behaviour. Multi-dtype factories publish the
combinations they support as `factory.dtypes`. The
factories above also export their references as `*_ref` functions
(`add_ref`, `reduce_max_ref`, `mm_ref`, ...), so host code never reimplements
the math.

::: iron.kernels._common
    options:
      show_root_heading: false
      members:
        - KernelContract
        - ROLES

`aie.utils.kernel_harness` turns any contract-bearing factory into a
single-Worker design. What a kernel can answer about itself -- its
reference result, its safe input range, which arguments are parameters,
how to judge a device output -- lives on
[`ExternalFunction`][iron.ExternalFunction] instead, so bringing up a
kernel needs no test harness. See
[Kernel Library](../programming_guide/kernels_library.md#adding-a-kernel) for
the add-a-kernel procedure and the test tiers built on it.

::: utils.kernel_harness
    options:
      show_root_heading: false
      members:
        - design
        - host_args
        - HostArg
        - sample_inputs
        - host_layout
        - output_size
        - upload
        - cycles_per_call

A `Case` names one kernel at one shape; the tests and the benchmark read
the same table (`test/python/npu/kernel_cases.py`).

::: utils.kernel_harness.cases
    options:
      show_root_heading: false
      members:
        - Case
        - data_policy
        - inputs_for
        - load_cases

`test/python/npu/test_kernels_bench.py` times the library: correctness
first, then cycles, wall time and build size, gated by a device preflight
and a measurement-sanity test. It is an ordinary pytest module, so `-k`
selects cases and the session's exit status decides whether any numbers are
written. What it measures with lives here:

::: utils.benchmark
    options:
      show_root_heading: false
      members:
        - Stats
        - BenchmarkResult
        - Preflight
        - run_iters
        - preflight
        - measure_compile
        - provenance

## Static checks

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
