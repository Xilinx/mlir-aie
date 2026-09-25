<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Cast f32 -> bf16

This design implements an element-wise narrowing cast from `float32` to `bfloat16` across an 8-core split, round-to-nearest-even. It runs on NPU1 (aie2) and NPU2 (aie2p); the underlying kernel, `aie_kernels/datamovement/cast_f32_bf16.cc`, builds for both.

`out = bfloat16(in)`, round-to-nearest-even, over `n_vectors` vectors of `vector_size` elements each.

This is a seam op: an on-chip dataflow that produces f32 (a numerically-sensitive reduction, for instance) and one that consumes bf16 (a bf16 matmul) can be connected without the intermediate leaving the device to be requantized on the host. Rounding matters here: the AIE default float32->bfloat16 narrow truncates toward zero, which does not match a host pack that rounds to nearest even, so the two would disagree by up to 1 ULP per element without the explicit `aie::rounding_mode::conv_even`.

## Source Files Overview

1. `cast_f32_bf16.py`: IRON design. Structurally mirrors [`ml/norm`](../norm)'s 8-core split `@iron.jit` design; the input and output tiles have different dtypes here (f32 in, bf16 out), which the `transform_parallel`/`transform_parallel_binary` algorithm helpers do not support (they require uniform dtype across every tensor), so this uses the explicit `ObjectFifo`/`Worker`/`Runtime` wiring instead of one of those helpers.

1. `cast_f32_bf16.cc`: the kernel, pulled from [`aie_kernels/datamovement/`](../../../aie_kernels/datamovement/); the same source builds for aie2 and aie2p.

## Usage

```shell
python3 cast_f32_bf16.py --dev npu2
python3 cast_f32_bf16.py --dev npu
```
