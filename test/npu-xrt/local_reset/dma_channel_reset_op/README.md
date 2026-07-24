<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# DMA channel reset (op-based)

The op-based analog of [`../dma`](../dma/README.md): same two-BD design (bad BD 0
`[900..907]`, good BD 1 `[100..107]`, both gated on `cons`), but the channel is
flushed with the merged `aiex.dma_channel_reset` op
([#3370](https://github.com/Xilinx/mlir-aie/pull/3370)) instead of the raw reset
writes. It reuses `../dma`'s host oracle directly (`%S/../dma/test.cpp`).

`aiex.dma_channel_reset(%t02, MM2S, 0)` lowers (in the default `aiecc` pipeline) to
the same mask-preserving reset pulse on `DMA_MM2S_0_Ctrl` (`0x1DE10`, bit 1) that
`../dma` issues directly -- two `aiex.npu.maskwrite32`s, with the address from
`AIETargetModel::getDmaControlAddress`. The op is reset-only, so the enqueue
(`aiex.npu.push_queue`) and lock arm remain around it.

## Behaviour

- **Correct protocol (this test):** enqueue bad -> `aiex.dma_channel_reset` ->
  enqueue good -> arm lock -> the collect matches `[100..107]`.
- **No reset (falsifies the test):** the bad BD sends first -> the collect is
  `[900..907]` -- a wrong-data failure, not a hang.
- **Reset without re-push:** nothing queued -> the collect never completes (hang).

## Reference

`aiex.dma_channel_reset` is defined in `include/aie/Dialect/AIEX/IR/AIEX.td` and
lowered by `lib/Dialect/AIEX/Transforms/AIELowerDmaChannelReset.cpp`; it mirrors
aie-rt's `XAie_DmaChannelReset` (`driver/src/dma/xaie_dma.c`). See
[`../dma`](../dma/README.md) and [`../README.md`](../README.md#references).
