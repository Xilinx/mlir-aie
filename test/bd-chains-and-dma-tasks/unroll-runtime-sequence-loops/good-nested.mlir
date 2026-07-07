//===- good-nested.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-unroll-runtime-sequence-loops \
// RUN:         --aie-assign-runtime-sequence-bd-ids \
// RUN:         --split-input-file %s | FileCheck %s

// Constant-trip loops are unrolled wherever they appear in the sequence, not
// only as immediate children of the sequence body: loops nested inside other
// loops and inside scf.if arms are reached too. After unrolling, ordinary
// liveness-based allocation colors the resulting straight-line configures.

// -----

// Constant loop nested inside another constant loop: both fully unroll (2x2 = 4
// same-iteration configures, each freed within its body copy => id 0 reused).
// CHECK-LABEL: @for_in_for
// CHECK-NOT:   scf.for
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @for_in_for(%arg0: memref<1024xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      scf.for %j = %c0 to %c2 step %c1 {
        // CHECK: aie.dma_bd{{.*}}{bd_id = 0 : i32}
        %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<1024xi32>, 0, 256)
          aie.end
        }
        aiex.dma_start_task(%t)
        aiex.dma_await_task(%t)
      }
    }
  }
}

// -----

// Constant loop that CONTAINS an scf.if: the loop unrolls, and the scf.if
// bodies are preserved (an scf.if is never unrolled).
// CHECK-LABEL: @for_containing_if
// CHECK-NOT:   scf.for
// CHECK-COUNT-2: scf.if
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @for_containing_if(%arg0: memref<1024xi32>, %c: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      scf.if %c {
        // CHECK: aie.dma_bd{{.*}}{bd_id = 0 : i32}
        %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<1024xi32>, 0, 256)
          aie.end
        }
        aiex.dma_start_task(%t)
        aiex.dma_await_task(%t)
      }
    }
  }
}

// -----

// Constant loop nested INSIDE an scf.if arm: the loop still unrolls (it is not
// an immediate child of the sequence body). The scf.if is preserved.
// CHECK-LABEL: @for_inside_if
// CHECK-NOT:   scf.for
// CHECK:       scf.if
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @for_inside_if(%arg0: memref<1024xi32>, %c: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.if %c {
      scf.for %j = %c0 to %c2 step %c1 {
        // CHECK: aie.dma_bd{{.*}}{bd_id = 0 : i32}
        %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<1024xi32>, 0, 256)
          aie.end
        }
        aiex.dma_start_task(%t)
        aiex.dma_await_task(%t)
      }
    }
  }
}

// -----

// A constant-trip rolled ping-pong nested inside an scf.if arm. This is the
// same shape as the top-level good-pingpong.mlir case, but one level deeper.
// It must unroll and allocate identically (ids alternate 0, 1, 0, 1), NOT be
// misclassified as a runtime-bound loop and rejected for the dynamic path.
// CHECK-LABEL: @pingpong_inside_if
// CHECK-NOT:   scf.for
// CHECK:       aie.dma_bd{{.*}}{bd_id = 0 : i32}
// CHECK:       aie.dma_bd{{.*}}{bd_id = 1 : i32}
// CHECK:       aie.dma_bd{{.*}}{bd_id = 0 : i32}
// CHECK:       aie.dma_bd{{.*}}{bd_id = 1 : i32}
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @pingpong_inside_if(%arg0: memref<1024xi32>, %c: i1) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.if %c {
      %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32>, 0, 256)
        aie.end
      }
      aiex.dma_start_task(%init)
      %last = scf.for %i = %c1 to %c4 step %c1 iter_args(%prev = %init) -> (index) {
        %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<1024xi32>, 0, 256)
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
