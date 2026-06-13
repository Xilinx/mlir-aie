<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# Row-wise Norm (RMS | Layer)

This design implements a `bfloat16` row-wise norm — either **RMSNorm** or **LayerNorm** — across an 8-core sequence. The op is selected at compile time via the `op` parameter; the structural design and host harness are shared. NPU2-only (the underlying kernels live under `aie_kernels/aie2p/`).

Per row:

* `op=rms` &nbsp;&nbsp;`out = (x * gamma) / sqrt(mean(x^2) + eps)`  &nbsp;&nbsp;(gamma=1, eps=1e-5)
* `op=layer` &nbsp;&nbsp;`out = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta`  &nbsp;&nbsp;(gamma=1, beta=0, eps=1e-5)


## Source Files Overview

1. `norm.py`: IRON design. `op` is a `CompileTime[str]` parameter that selects `rms_norm.cc` or `layer_norm.cc`. Per-op reference and tolerance live in a small dispatch table.

1. `rms_norm.cc` / `layer_norm.cc`: AIE2P kernels pulled from [`aie_kernels/aie2p/`](../../../aie_kernels/aie2p/).

1. `test.cpp`: C++ testbench that loads the compiled XCLBIN + `insts.bin` via `setup_and_run_aie`, computes the per-row reference, and reports pass/fail with a per-op tolerance. The op is selected via the `NORM_OP` env var (set by the Makefile's `run` target).


## Usage

```shell
make op=rms && make run op=rms
make op=layer && make run op=layer
```
