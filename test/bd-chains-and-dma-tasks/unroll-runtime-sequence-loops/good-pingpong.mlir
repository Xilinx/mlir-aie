//===- good-pingpong.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-unroll-runtime-sequence-loops \
// RUN:         --aie-assign-runtime-sequence-bd-ids \
// RUN:         --split-input-file %s | FileCheck %s

// Depth-1 ping-pong over a constant 4-iteration loop. Unrolling runs first,
// expanding the loop into straight-line configures (prologue + 3 body copies,
// each freeing the previous). BD-ID allocation then colors them by ordinary
// liveness reuse: the free of the previous configure returns its id before the
// next configure claims one, so the ids alternate 0, 1, 0, 1. No scf.for and no
// rotating-window attribute should remain.

// CHECK-LABEL: @pingpong_depth1
// CHECK-NOT:   scf.for
// CHECK:       aie.dma_bd{{.*}}{bd_id = 0 : i32}
// CHECK:       aie.dma_bd{{.*}}{bd_id = 1 : i32}
// CHECK:       aie.dma_bd{{.*}}{bd_id = 0 : i32}
// CHECK:       aie.dma_bd{{.*}}{bd_id = 1 : i32}
// CHECK-NOT:   bd_id_window
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @pingpong_depth1(%arg0: memref<1024xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%init)
    %last = scf.for %i = %c1 to %c4 step %c1 iter_args(%prev = %init) -> (index) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%prev)
      scf.yield %t : index
    }
    aiex.dma_await_task(%last)
    aiex.dma_free_task(%last)
  }
}



// -----

// Straight-line BDs are unaffected by the unroll pass and allocate as usual.

// CHECK-LABEL: @straight_line_unaffected
// CHECK-NOT:   scf.for
// CHECK:       aie.dma_bd{{.*}}{bd_id = 0 : i32}
// CHECK-NOT:   bd_id_window
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @straight_line_unaffected(%arg0: memref<1024xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 1024)
      aie.end
    }
    aiex.dma_start_task(%t)
    aiex.dma_await_task(%t)
  }
}
