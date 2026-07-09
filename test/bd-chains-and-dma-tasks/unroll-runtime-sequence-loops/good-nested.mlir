//===- good-nested.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-unroll-runtime-sequence-loops --canonicalize \
// RUN:         --aie-assign-runtime-sequence-bd-ids \
// RUN:         --split-input-file %s | FileCheck %s

// The static path is unroll -> canonicalize -> allocate. The unroll pass must
// reach constant-trip scf.for loops wherever they appear -- including nested in
// another loop or inside a (still-unfolded) constant scf.if arm -- because it
// runs BEFORE canonicalize folds the constant scf.if. If it did not descend
// into the arm, the inner loop would survive un-unrolled, the if would then
// fold, and the allocator would reject a sequence that is actually fully static.
// After unroll + canonicalize every case here is straight-line and allocates by
// ordinary liveness reuse.


// -----

// Constant loop nested inside another constant loop: both fully unroll; each
// task is freed within its iteration, so id 0 is reused throughout.
// CHECK-LABEL: @for_in_for
// CHECK-NOT:   scf.for
// CHECK:       aie.dma_bd{{.*}}{bd_id = 0 : i32}
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @for_in_for(%arg0: memref<1024xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      scf.for %j = %c0 to %c2 step %c1 {
        %c0_i32 = arith.constant 0 : i32
        %c256_i32 = arith.constant 256 : i32
        %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<1024xi32> offset = %c0_i32 len = %c256_i32)
          aie.end
        }
        aiex.dma_start_task(%t)
        aiex.dma_await_task(%t)
      }
    }
  }
}


// -----

// Constant loop nested inside a constant-predicate scf.if: unroll expands the
// loop (before the if is folded), canonicalize folds the if, and the result is
// straight-line -- no scf op reaches the allocator.
// CHECK-LABEL: @for_inside_const_if
// CHECK-NOT:   scf.for
// CHECK-NOT:   scf.if
// CHECK:       aie.dma_bd{{.*}}{bd_id = 0 : i32}
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @for_inside_const_if(%arg0: memref<1024xi32>) {
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.if %true {
      scf.for %j = %c0 to %c2 step %c1 {
        %c0_i32 = arith.constant 0 : i32
        %c256_i32 = arith.constant 256 : i32
        %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<1024xi32> offset = %c0_i32 len = %c256_i32)
          aie.end
        }
        aiex.dma_start_task(%t)
        aiex.dma_await_task(%t)
      }
    }
  }
}


// -----

// A constant-trip rolled ping-pong nested inside a constant-predicate scf.if:
// the for unrolls one level deep, the if folds, and the ids alternate 0, 1 just
// like the top-level ping-pong (see good-pingpong.mlir).
// CHECK-LABEL: @pingpong_inside_const_if
// CHECK-NOT:   scf.for
// CHECK-NOT:   scf.if
// CHECK:       aie.dma_bd{{.*}}{bd_id = 0 : i32}
// CHECK:       aie.dma_bd{{.*}}{bd_id = 1 : i32}
// CHECK:       aie.dma_bd{{.*}}{bd_id = 0 : i32}
// CHECK:       aie.dma_bd{{.*}}{bd_id = 1 : i32}
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @pingpong_inside_const_if(%arg0: memref<1024xi32>) {
    %true = arith.constant true
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.if %true {
      %c0_i32 = arith.constant 0 : i32
      %c256_i32 = arith.constant 256 : i32
      %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = %c0_i32 len = %c256_i32)
        aie.end
      }
      aiex.dma_start_task(%init)
      %last = scf.for %i = %c1 to %c4 step %c1 iter_args(%prev = %init) -> (index) {
        %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<1024xi32> offset = %c0_i32 len = %c256_i32)
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
}
