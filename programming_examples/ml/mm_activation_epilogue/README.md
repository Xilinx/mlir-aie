<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# mm_activation_epilogue: one resident program, three RTP-selected GEMM epilogues

This design implements a fused, `float32`-in/`float32`-out GEMM epilogue for
`aie2p` that applies one of `{identity, SiLU, GELU}` to a matmul's f32
accumulator tile. All three modes are compiled into ONE xclbin; which one
runs is selected per dispatch by a single runtime-parameter (RTP) word, not
by loading a different xclbin. NPU2-only (the underlying kernel lives under
`aie_kernels/aie2p/`).

## Why a runtime-selected epilogue

A GEMM microkernel's activation (none for a plain linear layer, SiLU for a
SwiGLU-style FFN, GELU for a classic transformer FFN) is usually fixed at
build time, one xclbin per activation. In a pipeline that calls several
differently-activated matmuls back to back, that means a full hardware
reconfiguration between them. This design instead compiles all three
epilogue bodies into one resident program and switches between them with an
`aiex.npu.rtp_write`-class dispatch: same xclbin, same `hw_context`, no
reload.

On NPU2 all three modes dispatch clean: identity is bit-exact against the
reference, and SiLU and GELU hold the `atol=0.05` gate over a `[-8, 8]` sweep.

## Source Files Overview

1. `mm_activation_epilogue.py`: IRON design. Two cores split a flat
   `size`-element `float32` input. Each core waits on a
   `WorkerRuntimeBarrier`, reads a per-core RTP `mode` word, and runs one
   pass over its tiles under that mode before releasing the barrier. The
   runtime sequence dispatches three such epochs -- mode 0, 1, 2 -- each
   draining into its own output tensor, so every mode is checked
   independently. Follows [`ml/scale_shift`](../scale_shift)'s RTP-parameter
   dispatch, extended to three phases and three outputs.

1. `mm_activation_epilogue.cc`: AIE2P kernel, from
   [`aie_kernels/aie2p/`](../../../aie_kernels/aie2p/). One `mode` argument
   (0/1/2) selects identity, SiLU, or GELU.

## Usage

```shell
python3 mm_activation_epilogue.py --dev npu2
```
