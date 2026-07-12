<!-- Copyright (C) 2024-2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Tensor Access Patterns (`taplib`)

`taplib` provides abstractions for describing how data is tiled and streamed
between memory and AIE compute tiles. A **Tensor Access Pattern** (TAP)
describes a multi-dimensional iteration over a buffer, generating the DMA
descriptor sequences that the NPU hardware executes.

::: helpers.taplib.tap.TensorAccessPattern
    options:
      show_root_heading: true
      heading_level: 2

::: helpers.taplib.tas.TensorAccessSequence
    options:
      show_root_heading: true
      heading_level: 2

::: helpers.taplib.tensortiler2d.TensorTiler2D
    options:
      show_root_heading: true
      heading_level: 2

## Utilities

::: helpers.taplib.utils
    options:
      show_root_heading: false
