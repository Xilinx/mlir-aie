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
To avoid unnecessarily reloading the `bias` vector, we iterate through these tiles in a column-major fashion by swapping the two grid dimensions of the tiling:
`TensorAccessPattern.full((M, N)).tile((m, n)).permute((1, 0, 2, 3))`.

## Kernel

The vectorized kernel is implemented in `kernel.cc`.
The kernel uses vector intrinsics of size `t` to perform the additions.
The computation is designed such that the `bias` vector is not unnecessarily reloaded.
To achieve this, we first load a chunk of `t` elements of `bias`, then produce the results for the first `t` columns of `out` (this is the inner loop).
The outer loop iterates through chunks of `t` columns, loading the next `t` biases at the beginning of each iteration.

## Row-wise Affine Cast

`--op affine_cast` selects a second design in `row_wise_bias_add.py`: a per-column affine transform, `out = bfloat16(in*gamma + beta)`, narrowing the `float32` input to `bfloat16` on the way out.
It walks the same column-major tile order, so each `gamma`/`beta` block is loaded once per column of tiles.
The kernel is the library's [`aie.iron.kernels.affine_cast`](../../../python/iron/kernels/datamovement.py) (source [`aie_kernels/datamovement/affine_cast_f32_bf16.cc`](../../../aie_kernels/datamovement/affine_cast_f32_bf16.cc)), so the regression suite in `test/python/npu/kernel_cases.py` checks it as well.

`gamma` and `beta` are the two rows of one `2`&times;`N` tensor, since an AIE2 tile has only two input DMA channels and `in` already uses one.
Tiling that tensor with `TensorAccessPattern.full((2, N)).tile((2, n))` delivers `gamma`'s `n`-wide column block followed by `beta`'s, which is the packing the kernel reads, so the host needs no reordering.

The narrowing store rounds to nearest even (`aie::rounding_mode::conv_even`), as a host `float32`-\>`bfloat16` conversion does; the AIE default truncates toward zero.
The `float32` multiply is emulated in `bfloat16` terms and can land one `float32` ulp off, so an output that sits on a `bfloat16` rounding tie may come out one `bfloat16` ulp away; the design checks against the kernel's declared tolerance.

```shell
python3 row_wise_bias_add.py --op affine_cast --dev npu2
```
