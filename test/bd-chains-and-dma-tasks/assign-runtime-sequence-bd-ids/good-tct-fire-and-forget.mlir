//===- good-tct-fire-and-forget.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-unroll-runtime-sequence-loops --canonicalize \
// RUN:         --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// Anti-false-positive: a rolled ping-pong pushes N tokens on a channel and
// awaits only ONCE (the last), letting the channel FIFO cover the rest. This is
// legal -- over-production is always safe -- and must NOT be flagged as an
// imbalance. After unrolling, the channel sees 4 issue_token pushes and 1 pop,
// so the token count never goes negative.

// CHECK-LABEL: @fire_and_forget
// CHECK-NOT:   scf.for
// CHECK:       aiex.dma_await_task
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @fire_and_forget(%arg0: memref<8xi16>) {
    %c0 = arith.constant 0 : index
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
      aiex.dma_free_task(%prev)
      scf.yield %t : index
    }
    // One await covers all prior pushes on this channel via FIFO ordering.
    aiex.dma_await_task(%last)
    aiex.dma_free_task(%last)
  }
}
