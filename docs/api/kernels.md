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
        - OVERFLOW
        - ROUNDING
        - NONFINITE
        - SUBNORMALS

`aie.utils.kernel_harness` turns any contract-bearing factory into a
single-Worker design, runs it, and judges the result with
`aie.utils.verify.compare`. See
[Kernel Library](../programming_guide/kernels_library.md#adding-a-kernel) for
the add-a-kernel procedure.

::: utils.kernel_harness
    options:
      show_root_heading: false
      members:
        - design
        - host_args
        - HostArg
        - sample_inputs
        - expected
        - input_limit
        - upload
        - run
        - check
        - judge
        - cycles_per_call

`aie.utils.bfp` is the host side of the block-floating-point kernels: the
bfp16ebs8 codec and the tile shuffle the `mm_bfp` DMA layout needs, ported
bit for bit from the block_datatypes examples' `helper.h`.

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
