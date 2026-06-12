<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# Eltwise Unary (ReLU | SiLU | GELU)

This design implements a `bfloat16` element-wise unary op (ReLU, SiLU, or GELU), parallelized across both shim DMA channels per column (`num_channels=2`). The op is selected at compile time via the `op` parameter; the structural design and host harness are shared.

ReLU is exact; SiLU and GELU use LUT-backed kernels and have per-op verification tolerances.


## Source Files Overview

1. `eltwise_unary.py`: IRON design. `op` is a `CompileTime[str]` parameter that selects `kernels.relu`, `kernels.silu`, or `kernels.gelu`. Per-op reference and tolerance live in a small dispatch table.

1. `relu.cc` / `silu.cc` / `gelu.cc`: Vectorized AIE kernels pulled from the IRON kernel library. Sources under [`aie_kernels/aie2/`](../../../aie_kernels/aie2/).

1. `test.cpp`: C++ testbench that loads the compiled XCLBIN via the modern `xrt::elf` path, runs the kernel, and verifies output against a CPU reference. Pass `--op {relu,silu,gelu}` to match the compiled design.


## Usage

```shell
make op=relu && make run op=relu
make op=silu && make run op=silu
make op=gelu && make run op=gelu
```
