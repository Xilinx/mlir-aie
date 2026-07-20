//===- dma_channel_reset_invalid.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt -split-input-file --aie-lower-dma-channel-reset %s 2>&1 | FileCheck %s

// A channel index beyond the tile's DMA channel count is rejected by the
// verifier before it can reach getDmaControlAddress and emit a write to an
// unintended register. A core tile has 2 S2MM channels, so channel 2 is out of
// range.
module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      // CHECK: channel 2 out of range for this tile and direction
      aiex.dma_channel_reset(0, 3, S2MM, 2)
    }
  }
}
