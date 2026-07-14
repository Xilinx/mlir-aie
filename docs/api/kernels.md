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

## Vision

::: iron.kernels.vision
    options:
      show_root_heading: false
