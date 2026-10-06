<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Depthwise Conv1d ('same', stride 1, bf16)

This design implements a depthwise (per-channel) 1D convolution, 'same' padding, stride 1, bf16, across an `n_cores`-way channel split (8 by default). It runs on NPU1 (aie2) and NPU2 (aie2p); the underlying kernel lives under `aie_kernels/conv/` and builds for both. On NPU1 pass `-n 4`: each core streams `x` and `w` from the shim, and NPU1's four shim tiles have 8 such DMA channels, so the default 8 cores do not place.

Per channel of `seq_len`, with an optional per-channel bias:

```
out[t] = bias + sum_{p=0..kernel_size-1} w[p] * x[t + p],   t = 0 .. seq_len - 1
```

a cross-correlation (no kernel flip) over `x` zero-padded by `(kernel_size - 1) // 2` on each side, matching `torch.nn.Conv1d` / most framework "same" depthwise convs. Each channel has its own `kernel_size` taps (+ optional bias); `channels` must be a multiple of `n_cores` (default 8) and `seq_len` a multiple of 16.

## Source Files Overview

1. `dwconv1d.py`: IRON design. `n_cores` cores each process `channels // n_cores` channels; one ObjectFifo tile per channel row (matches `ml/norm`'s per-row structure). The caller passes `x` already 'same'-padded, plus a fixed 16-element tail slack the kernel's aligned loads need past the halo, see `_pad_input` and the kernel header comment for the exact layout. The weights tensor's row width (`w_row`) is always `kernel_size + 1`, even with `bias=False`: `kernel_size` is required odd, so `kernel_size` alone gives an odd bf16 row (`2 * kernel_size` bytes), which fails `aie.dma_bd`'s 4-byte transfer-length alignment; `kernel_size + 1` is always even. The extra column is unused, zero-filled padding when `bias=False`, the kernel never reads it.

1. `dwconv1d_channels_first.cc`: the kernel, pulled from [`aie_kernels/conv/`](../../../aie_kernels/conv/). Vectorized, 16 outputs per block. On aie2p, two aligned 256-bit loads build a 32-lane window, and two `aie::sliding_mul_ops<16, ...>` chains of about `K / 2` taps each run the correlation over it, joined by one accumulator add. aie2 has no bf16 sliding multiply; there one shift of a 48-sample window serves a tap for two blocks, and the halves of two taps' shifts fill the two halves of each `vmac.f`. It is the channels-first member of a pair; see [_Choosing a depthwise conv1d_](../../../aie_kernels/README.md#choosing-a-depthwise-conv1d) for when the channels-last form is the right one instead.

## Usage

```shell
python3 dwconv1d.py --dev npu2
python3 dwconv1d.py --dev npu2 -k 5 --no_bias
python3 dwconv1d.py --dev npu -n 4
```

Override `channels`, `seq_len`, `kernel_size`, `n_cores`, or `bias` on the command line; see `--help`.
