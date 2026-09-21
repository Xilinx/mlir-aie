<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Row-wise Norm (RMS | Layer)

This design implements a row-wise norm (**RMSNorm** or **LayerNorm**) across an 8-core sequence. The op is selected at compile time via the `op` parameter; the structural design and host harness are shared for the same-dtype ops. NPU2-only (the underlying kernels live under `aie_kernels/aie2p/`).

Per row:

* `op=rms` &nbsp;&nbsp;`out = (x * gamma) / sqrt(mean(x^2) + eps)`  &nbsp;&nbsp;(bf16; gamma=1, eps=1e-5)
* `op=layer` &nbsp;&nbsp;`out = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta`  &nbsp;&nbsp;(bf16; gamma=1, beta=0, eps=1e-5)
* `op=layer_f32` &nbsp;&nbsp;the same LayerNorm in f32 in/out, with a numerically stable centered two-pass variance. bf16 keeps the mean/std ratio small (a bf16 value near a large mean has an ulp wider than the std), so `E[x^2] - mean^2` is safe there; f32 can represent a large mean, where that form cancels, so the f32 path centers first.
* `op=layer_affine_cast` &nbsp;&nbsp;the same centered LayerNorm, with a REAL per-column `gamma`/`beta` (not fixed to 1/0) and a `float32 -> bfloat16` narrowing cast fused into the same dispatch: `out = ((x - mean(x)) / sqrt(var(x) + eps)) * gamma + beta -> bfloat16`. `gamma`/`beta` are the same for every row and are packed into one `[2 * embedding_dim]` buffer (gamma then beta) that every core loads once, before its row loop. This is a seam op: a dataflow stage that needs f32 precision for the LayerNorm reduction but whose consumer is a bf16 matmul can produce its final, affine-applied, already-narrowed output in one on-chip dispatch.

`rms`/`layer`/`layer_f32` share one dtype in and out and are wired by the `norm` design below; `layer_affine_cast` has a different tensor shape (f32 in, bf16 out, plus the gamma/beta parameter tensor) and is wired by a second design, `norm_affine`, in the same file.

## Source Files Overview

1. `norm.py`: IRON design. `op` is a `CompileTime[str]` parameter that selects the kernel symbol and source (all four live in `layer_norm.cc` except `rms`, which is `rms_norm.cc`) and the tensor dtype. `norm` handles the three same-dtype ops (bf16 for `rms`/`layer`, f32 for `layer_f32`); `norm_affine` handles `layer_affine_cast`'s f32-in/bf16-out/gamma-beta-tensor shape, mirroring `ml/cast_f32_bf16`'s dtype-split `ObjectFifo`/`Worker`/`Runtime` wiring plus a third, per-core-constant parameter tensor (acquired once before the row loop, not per row). Per-op reference and tolerance live in a small dispatch table shared by both designs.

1. `rms_norm.cc` / `layer_norm.cc`: AIE2P kernels pulled from [`aie_kernels/aie2p/`](../../../aie_kernels/aie2p/). `layer_norm.cc` holds `layer_norm` (bf16), `layer_norm_f32` (gamma=1/beta=0 identity affine), and `layer_norm_affine_cast` (real affine + cast) as one templated core (`layer_norm_f32_impl<TIn, TOut, N, kAffine>`) instantiated three ways.

1. `test.cpp`: C++ testbench for the bf16 ops (`rms`, `layer`). It loads the compiled XCLBIN + `insts.bin` via `setup_and_run_aie`, computes the per-row reference, and reports pass/fail with a per-op tolerance. The op is selected via the `NORM_OP` env var (set by the Makefile's `run` target). The f32 ops (`layer_f32`, `layer_affine_cast`) are verified through the standalone Python path instead (`norm.py`'s dispatch table, driven by `run_strix.lit`), against an f64/f32 gold reference.


## Usage

### Standalone JIT verification

```shell
python3 norm.py --dev npu2 --op rms
python3 norm.py --dev npu2 --op layer
python3 norm.py --dev npu2 --op layer_f32 --embedding_dim 2048
python3 norm.py --dev npu2 --op layer_affine_cast --embedding_dim 1024
```

`layer_f32` and `layer_affine_cast` default to a smaller `embedding_dim` than `rms`/`layer` because an f32 row is twice the bytes of a bf16 row (plus, for `layer_affine_cast`, a `[2 * embedding_dim]` f32 gamma/beta buffer), and a 4096-wide row does not fit the tile's local memory double-buffered.

### C++ Testbench

```shell
make op=rms && make run op=rms
make op=layer && make run op=layer
```
