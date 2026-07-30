<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Cast f32 -> bf16

This design implements a row-wise element-wise narrowing cast from `float32` to `bfloat16` across an 8-core sequence, round-to-nearest-even. NPU2-only (the underlying kernel lives under `aie_kernels/aie2p/`).

`out = bfloat16(in)`, round-to-nearest-even, per row over `embedding_dim`.

This is a seam op: an on-chip dataflow that produces f32 (a numerically-sensitive reduction, for instance) and one that consumes bf16 (a bf16 matmul) can be connected without the intermediate leaving the device to be requantized on the host. Rounding matters here: the AIE default float32->bfloat16 narrow truncates toward zero, which does not match a host pack that rounds to nearest even, so the two would disagree by up to 1 ULP per element without the explicit `aie::rounding_mode::conv_even`.

## Source Files Overview

1. `cast_f32_bf16.py`: IRON design. Structurally mirrors [`ml/norm`](../norm)'s 8-core row-split `@iron.jit` design; the input and output tiles have different dtypes here (f32 in, bf16 out), which the `transform_parallel`/`transform_parallel_binary` algorithm helpers do not support (they require uniform dtype across every tensor), so this uses the explicit `ObjectFifo`/`Worker`/`Runtime` wiring instead of one of those helpers.

1. `cast_f32_bf16.cc`: AIE2P kernel, pulled from [`aie_kernels/aie2p/`](../../../aie_kernels/aie2p/).

1. `test.cpp`: C++ testbench. Loads the compiled XCLBIN + `insts.bin` via `setup_and_run_aie`, and checks every output element against `test_utils::bfloat16_from_float` (round-to-nearest-even), which is the same rounding rule as the kernel, an exact bit-for-bit compare, not a tolerance.

## Usage

### Standalone JIT verification

```shell
python3 cast_f32_bf16.py --dev npu2
```

### C++ Testbench

```shell
make
make run
```
