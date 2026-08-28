<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Row-wise Bias Addition

This design takes two inputs, `in` and `bias`. 
`in` is a `M`&times;`N` matrix, and `bias` is a `1`&times;`N` row-vector.
The design performs a row-wise addition of `bias` to `in`. 
Conceptually, `bias` is broadcast into a `M`&times;`N` matrix by repeating it `M` times across rows, and then this matrix is added element-wise to `in`.

## Data Movement

The data movement and call into the kernel (see below) is described in `row_wise_bias_add.py`, a single `@iron.jit` design that compiles directly to NPU binaries via `--xclbin-path` / `--insts-path` and uses `ExternalFunction(source_file=…, compile_flags=[-DDIM_m=…, -DDIM_n=…])` so the `kernel.cc` build is part of the JIT flow.
A single AIE core is configured to process chunks of `m`&times;`n` of `in` and chunks of `n` of `bias` to produce `m`&times;`n` chunks of output.
Therefore, the output is tiled into `M/m`&times;`N/n` tiles, and the kernel function is called that number of times.
To avoid unnecessarily reloading the `bias` vector, we iterate through these tiles in a column-major fashion by calling the `TensorTiler2D.group_tiler`
with argument `tile_group_col_major=True`.

## Kernel

The vectorized kernel is implemented in `kernel.cc`.
The kernel uses vector intrinsics of size `t` to perform the additions.
The computation is designed such that the `bias` vector is not unnecessarily reloaded.
To achieve this, we first load a chunk of `t` elements of `bias`, then produce the results for the first `t` columns of `out` (this is the inner loop).
The outer loop iterates through chunks of `t` columns, loading the next `t` biases at the beginning of each iteration.

## Row-wise Affine Cast

`--op affine_cast` selects a second design in the same files: a per-column affine transform, `out = bfloat16(in*gamma + beta)`, narrowing the `float32` input to `bfloat16` on the way out.
`gamma` and `beta` are packed into one `1`&times;`2N` buffer, block-interleaved (`gamma` then `beta` per `n`-wide column block), since an AIE2 tile has only two input DMA channels and `in` already uses one.
The narrowing cast rounds to nearest even (`aie::rounding_mode::conv_even`), matching a host `float32`-\>`bfloat16` pack bit-for-bit; the AIE default truncates toward zero.

```shell
python3 row_wise_bias_add.py --op affine_cast --dev npu2
```
