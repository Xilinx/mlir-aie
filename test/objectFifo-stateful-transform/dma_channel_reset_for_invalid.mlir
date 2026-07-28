//===- dma_channel_reset_for_invalid.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -split-input-file -verify-diagnostics --aie-objectFifo-stateful-transform %s

// A dma_channel_reset_for on a fifo with no resident core/mem state to re-arm
// (here a shared-memory fifo with synchronization disabled: no DMA channel and
// no locks) is diagnosed at the fault site, instead of leaving a dead binding or
// a dangling @fifo_shim_alloc symbol reference for a later pass to trip over.
module {
  aie.device(npu2) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.objectfifo @e (%t02, {%t03}, 2 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence() {
      // expected-error @+1 {{objectFIFO 'e' has no resident core/mem DMA channels or locks to re-arm}}
      aiex.dma_channel_reset_for(@e)
    }
  }
}
