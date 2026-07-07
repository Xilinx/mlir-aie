//===- good-conv-halo-asymmetric.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-unroll-runtime-sequence-loops \
// RUN:         --aie-assign-runtime-sequence-bd-ids \
// RUN:         %s | FileCheck %s

// A boundary-vs-interior asymmetry, e.g. a convolution whose first tile has no
// halo (a single-BD load) and whose interior tiles carry a halo (a two-BD
// chain). Expressed as a one-behind ping-pong where the prologue chain (C=1)
// differs in length from the body chain (C=2). With unroll-first allocation
// this needs no special handling: after unrolling, every configure is
// independent and colored by liveness, so the differing chain lengths simply
// take the ids they need. (The old rotation-window allocator rejected this as a
// chain-length mismatch; it now just works.)

// CHECK-LABEL: @conv_halo
// CHECK-NOT:   scf.for
// Prologue: single BD gets id 0.
// CHECK:       aie.dma_bd{{.*}}{bd_id = 0 : i32}
// First interior tile: two-BD chain. id 0 is still held (freed after this
// configure), so the chain takes 1 and 2.
// CHECK:       aie.dma_bd{{.*}}{bd_id = 1 : i32}
// CHECK:       aie.dma_bd{{.*}}{bd_id = 2 : i32}
// CHECK-NOT:   bd_id_window
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @conv_halo(%arg0: memref<1024xi32>) {
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    // Prologue: boundary tile, no halo -> single BD (C=1).
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32>, 0, 256)
      aie.end
    }
    aiex.dma_start_task(%init)
    // Interior tiles: main body + halo strip -> two-BD chain (C=2), one-behind
    // free of the previous tile's task.
    %last = scf.for %i = %c1 to %c3 step %c1 iter_args(%prev = %init) -> (index) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32>, 0, 128)
        aie.next_bd ^halo
      ^halo:
        aie.dma_bd(%arg0 : memref<1024xi32>, 128, 128)
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
