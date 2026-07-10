<!-- Copyright (C) 2024-2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Tensor Access Patterns (`taplib`)

`taplib` provides abstractions for describing how data is tiled and streamed
between memory and AIE compute tiles. A **Tensor Access Pattern** (TAP)
describes a multi-dimensional iteration over a buffer, generating the DMA
descriptor sequences that the NPU hardware executes.

## TensorAccessPattern

::: helpers.taplib.tap
    options:
      show_root_heading: false

## TensorAccessSequence

::: helpers.taplib.tas
    options:
      show_root_heading: false

## TensorTiler2D

::: helpers.taplib.tensortiler2d
    options:
      show_root_heading: false

## Utilities

::: helpers.taplib.utils
    options:
      show_root_heading: false
