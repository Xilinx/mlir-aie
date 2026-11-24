//===- good-tct-pingpong.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-unroll-runtime-sequence-loops --canonicalize \
// RUN:         --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// A rolled ping-pong pushes the next transfer before awaiting the previous
// one, so two are in flight at a time, and awaits the last after the loop. An
// await consumes the oldest token on its channel, so each await here is for
// the transfer it names, and the sequence ends with every token consumed.
// After unrolling, the channel sees 4 issue_token pushes and 4 awaits.

// CHECK-LABEL: @pingpong
// CHECK-NOT:   scf.for
// CHECK-COUNT-4: aiex.dma_await_task
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @pingpong(%arg0: memref<8xi16>) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16> offset = 0 len = 8)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%init)
    %last = scf.for %i = %c1 to %c4 step %c1 iter_args(%prev = %init) -> (index) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16> offset = 0 len = 8)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%prev)
      aiex.dma_free_task(%prev)
      scf.yield %t : index
    }
    aiex.dma_await_task(%last)
    aiex.dma_free_task(%last)
  }
}
