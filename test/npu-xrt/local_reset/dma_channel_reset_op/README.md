<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# DMA channel reset (op-based)

The op-based analog of [`../dma`](../dma/README.md): identical design, but the
stalled MM2S channel is reset with the merged `aiex.dma_channel_reset`
runtime-sequence op instead of raw `aiex.npu.write32`s. This is the on-board test
for the op added in [#3370](https://github.com/Xilinx/mlir-aie/pull/3370) (which
shipped with only `aie-opt` FileCheck coverage). See [`../README.md`](../README.md)
for the shared design and how to run.

A run-forever MM2S channel on tile `(0, 2)` is stalled on its consumer-lock
acquire. `aiex.dma_channel_reset(%t02, MM2S, 0)` clears the channel's residual
queue state; the test then re-pushes BD 0 (`aiex.npu.push_queue`) and re-arms the
lock so the channel runs again and the collect returns the resident buffer -- the
same oracle as `../dma`, whose `test.cpp` this test compiles directly
(`%S/../dma/test.cpp`) rather than duplicating.

## Same protocol as `../dma`

`aiex.dma_channel_reset` lowers to a **mask-preserving reset pulse** on the same
register `../dma` writes -- the MM2S channel-0 control register (tile-local
`0x1DE10`), reset bit 1 -- emitted as two `aiex.npu.maskwrite32`s (assert bit 1
with mask `0x2`, then clear it). The address comes from
`AIETargetModel::getDmaControlAddress`, so it is correct for every tile class and
direction. Masking to bit 1 preserves the other CTRL fields, so the op is
**reset-only**: it does not disable/enable the channel and does not re-push a BD
or re-arm the lock. `../dma` does those with raw writes
(`disable -> reset -> deassert -> enable`); here the op replaces the reset core of
that sequence and the re-push + lock re-arm remain. Both drive the same reset bit
and mirror aie-rt's `XAie_DmaChannelReset`; this test is the on-board confirmation
that the op's mask-preserving pulse re-arms the channel exactly as the raw
sequence does.

## Behaviour

- **Correct protocol (this test):** `aiex.dma_channel_reset` + re-push BD +
  re-arm lock -> the channel sends the resident buffer -> collect matches.
- **Reset without re-push / re-arm:** the flushed channel has nothing queued (or
  the lock is never granted), so the send never starts and the collect never
  completes -- a hang by design (reproduce by hand by removing the `push_queue` /
  lock write from `aie.mlir`).

## Reference

`aiex.dma_channel_reset` is defined in `include/aie/Dialect/AIEX/IR/AIEX.td` and
lowered by `lib/Dialect/AIEX/Transforms/AIELowerDmaChannelReset.cpp` (pass
`--aie-lower-dma-channel-reset`, run in the default `aiecc` pipeline). It mirrors
the public aie-rt driver (<https://github.com/Xilinx/aie-rt>, vendored at
`third_party/aie-rt/`): `XAIE2PGBL_MEMORY_MODULE_DMA_MM2S_0_CTRL` in
`driver/src/global/xaie2pgbl_params.h`, and `XAie_DmaChannelReset` in
`driver/src/dma/xaie_dma.c`. See [`../README.md`](../README.md#references) for the
full table.
