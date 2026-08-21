<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Depthwise Conv1d ('same', stride 1, bf16)

This design implements a depthwise (per-channel) 1D convolution, 'same' padding, stride 1, bf16, across an 8-core channel split. NPU2-only (the underlying kernel lives under `aie_kernels/aie2p/`).

Per channel of `seq_len`, with an optional per-channel bias:

```
out[t] = bias + sum_{p=0..kernel_size-1} w[p] * x[t + p],   t = 0 .. seq_len - 1
```

a cross-correlation (no kernel flip) over `x` zero-padded by `(kernel_size - 1) // 2` on each side, matching `torch.nn.Conv1d` / most framework "same" depthwise convs. Each channel has its own `kernel_size` taps (+ optional bias); `channels` must be a multiple of `n_cores` (default 8) and `seq_len` a multiple of 16.

## Source Files Overview

1. `dwconv1d.py`: IRON design. `n_cores` cores each process `channels // n_cores` channels; one ObjectFifo tile per channel row (matches `ml/norm`'s per-row structure). The caller passes `x` already 'same'-padded, plus a fixed 16-element tail slack the kernel's aligned loads need past the halo, see `_pad_input` and the kernel header comment for the exact layout. The weights tensor's row width (`w_row`) is always `kernel_size + 1`, even with `bias=False`: `kernel_size` is required odd, so `kernel_size` alone gives an odd bf16 row (`2 * kernel_size` bytes), which fails `aie.dma_bd`'s 4-byte transfer-length alignment; `kernel_size + 1` is always even. The extra column is unused, zero-filled padding when `bias=False`, the kernel never reads it.

1. `dwconv1d.cc`: AIE2P kernel, pulled from [`aie_kernels/aie2p/`](../../../aie_kernels/aie2p/). Vectorized 16-outputs/iteration: two aligned 256-bit loads build a 32-lane window and `aie::sliding_mul_ops<16, K, 1, 1, 1, bfloat16, bfloat16>` runs the K-tap correlation over it into one `accfloat` accumulator.

## Usage

```shell
python3 dwconv1d.py --dev npu2
python3 dwconv1d.py --dev npu2 -k 5 --no_bias
```

Override `channels`, `seq_len`, `kernel_size`, `n_cores`, or `bias` on the command line; see `--help`.
