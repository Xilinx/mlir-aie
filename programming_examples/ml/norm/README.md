<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Row-wise Norm (RMS | Layer)

This design implements a row-wise norm — **RMSNorm** or **LayerNorm** — across an 8-core sequence. The op is selected at compile time via the `op` parameter; the structural design and host harness are shared. NPU2-only (the underlying kernels live under `aie_kernels/aie2p/`).

Per row:

* `op=rms` &nbsp;&nbsp;`out = (x * gamma) / sqrt(mean(x^2) + eps)`  &nbsp;&nbsp;(bf16; gamma=1, eps=1e-5)
* `op=layer` &nbsp;&nbsp;`out = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta`  &nbsp;&nbsp;(bf16; gamma=1, beta=0, eps=1e-5)
* `op=layer_f32` &nbsp;&nbsp;the same LayerNorm in f32 in/out, with a numerically stable centered two-pass variance. bf16 keeps the mean/std ratio small (a bf16 value near a large mean has an ulp wider than the std), so `E[x^2] - mean^2` is safe there; f32 can represent a large mean, where that form cancels, so the f32 path centers first.


## Source Files Overview

1. `norm.py`: IRON design. `op` is a `CompileTime[str]` parameter that selects the kernel (`rms_norm.cc` or `layer_norm.cc`) and the tensor dtype (bf16, or f32 for `layer_f32`). Per-op reference and tolerance live in a small dispatch table.

1. `rms_norm.cc` / `layer_norm.cc`: AIE2P kernels pulled from [`aie_kernels/aie2p/`](../../../aie_kernels/aie2p/).

1. `test.cpp`: C++ testbench for the bf16 ops (`rms`, `layer`). It loads the compiled XCLBIN + `insts.bin` via `setup_and_run_aie`, computes the per-row reference, and reports pass/fail with a per-op tolerance. The op is selected via the `NORM_OP` env var (set by the Makefile's `run` target). The f32 `layer_f32` op is verified through the standalone Python path instead (`norm.py`'s dispatch table, driven by `run_strix.lit`), against an f64 gold reference.


## Usage

### Standalone JIT verification

```shell
python3 norm.py --dev npu2 --op rms
python3 norm.py --dev npu2 --op layer
python3 norm.py --dev npu2 --op layer_f32 --embedding_dim 2048
```

`layer_f32` uses `embedding_dim=2048` because an f32 row is twice the bytes of a bf16 row and a 4096-wide f32 row does not fit the tile's local memory double-buffered.

### C++ Testbench

```shell
make op=rms && make run op=rms
make op=layer && make run op=layer
```
