//===- dma_channel_reset_for_invalid.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -split-input-file -verify-diagnostics %s

// A symbol that resolves to something that is neither an aie.objectfifo nor its
// re-arm binding is rejected (here a shim_dma_allocation).
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    aie.shim_dma_allocation @notfifo (%t0, MM2S, 0)
    aie.runtime_sequence() {
      // expected-error @+1 {{'notfifo' must reference an aie.objectfifo}}
      aiex.dma_channel_reset_for(@notfifo)
    }
  }
}

// -----

// The resident re-arm is aie2p-specific (a DMA channel has no enable bit, so it
// is restarted by a START_QUEUE push); AIE1 is rejected.
module {
  aie.device(xcvc1902) {
    %t22 = aie.tile(2, 2)
    %t23 = aie.tile(2, 3)
    aie.objectfifo @of (%t22, {%t23}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence() {
      // expected-error @+1 {{is not supported on AIE1 devices}}
      aiex.dma_channel_reset_for(@of)
    }
  }
}
