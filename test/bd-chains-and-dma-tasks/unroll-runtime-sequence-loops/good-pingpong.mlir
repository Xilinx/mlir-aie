//===- good-pingpong.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids \
// RUN:         --aie-unroll-runtime-sequence-loops \
// RUN:         --split-input-file %s | FileCheck %s

// Depth-1 ping-pong over a constant 4-iteration loop.  The allocator assigns
// window [0, 1]; the unroll pass unrolls 3 body copies (plus prologue = 4
// total) and resolves them to bd_id 0, 1, 0, 1 in order.  No scf.for should
// remain after the pass.

// CHECK-LABEL: @pingpong_depth1
// CHECK-NOT:   scf.for
// CHECK:       aie.dma_bd{{.*}}{bd_id = 0 : i32}
// CHECK-NOT:   bd_id_window
// CHECK:       aie.dma_bd{{.*}}{bd_id = 1 : i32}
// CHECK-NOT:   bd_id_window
// CHECK:       aie.dma_bd{{.*}}{bd_id = 0 : i32}
// CHECK-NOT:   bd_id_window
// CHECK:       aie.dma_bd{{.*}}{bd_id = 1 : i32}
// CHECK-NOT:   bd_id_window
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @pingpong_depth1(%arg0: memref<1024xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256 sizes = [] strides = [])
      aie.end
    }
    aiex.dma_start_task(%init)
    %last = scf.for %i = %c1 to %c4 step %c1 iter_args(%prev = %init) -> (index) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256 sizes = [] strides = [])
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

// Straight-line BDs (no window) are unchanged by the unroll pass.

// CHECK-LABEL: @straight_line_unaffected
// CHECK-NOT:   scf.for
// CHECK:       aie.dma_bd{{.*}}{bd_id = 0 : i32}
// CHECK-NOT:   bd_id_window
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @straight_line_unaffected(%arg0: memref<1024xi32>) {
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 1024 sizes = [] strides = [])
      aie.end
    }
    aiex.dma_start_task(%t)
    aiex.dma_await_task(%t)
  }
}

// -----

// Runtime-bound loop: non-constant upper bound means the loop is NOT unrolled
// and the bd_id_window survives (to be handled by the dynamic EmitC path).

// CHECK-LABEL: @runtime_bound_preserved
// CHECK:       scf.for
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @runtime_bound_preserved(%arg0: memref<1024xi32>,
                                                %n: index) {
    %c1 = arith.constant 1 : index
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256 sizes = [] strides = [])
      aie.end
    }
    aiex.dma_start_task(%init)
    %last = scf.for %i = %c1 to %n step %c1 iter_args(%prev = %init) -> (index) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256 sizes = [] strides = [])
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
