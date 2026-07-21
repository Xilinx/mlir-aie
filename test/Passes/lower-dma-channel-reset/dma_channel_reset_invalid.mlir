//===- dma_channel_reset_invalid.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -split-input-file -verify-diagnostics --aie-lower-dma-channel-reset %s

// A channel index beyond the tile's DMA channel count is rejected by the
// verifier before it can reach getDmaControlAddress and emit a write to an
// unintended register. A core tile has 2 S2MM channels, so channel 2 is out of
// range.
module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      // expected-error @+1 {{channel 2 out of range for this tile and direction}}
      aiex.dma_channel_reset(0, 3, S2MM, 2)
    }
  }
}

// -----

// A shim NOC tile is rejected: only core and mem tiles have a per-channel DMA
// reset bit. On the shim NOC DMA control register bit 1 is PAUSE_MEM, not RESET
// (aie-rt's Aie2PShimDmaChProp sets Reset.Mask = 0), so there is nothing valid
// to lower to. This matches aie-rt's XAie_DmaChannelReset, which errors on
// SHIMNOC/SHIMPL tiles.
module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      // expected-error @+1 {{has no DMA channel reset (only core and mem tiles do; shim NOC DMA has no reset bit)}}
      aiex.dma_channel_reset(0, 0, S2MM, 0)
    }
  }
}
