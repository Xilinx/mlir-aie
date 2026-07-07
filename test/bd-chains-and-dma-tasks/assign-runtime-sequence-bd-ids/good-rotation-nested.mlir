//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids --split-input-file %s | FileCheck %s

// Nested loops with the ping-pong fully closed within each outer iteration. The
// inner rotation window (W=2) is reserved and released per outer iteration, so
// the outer loop reuses ids [0, 1] every iteration rather than accumulating.

// CHECK-LABEL: @rotation_nested
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @rotation_nested(%arg0: memref<8xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %o = %c0 to %c4 step %c1 {
      // CHECK: aie.dma_bd(%arg0 : memref<8xi16> offset = 0 len = 8 sizes = [] strides = []) {bd_id = 0 : i32, bd_id_window = array<i32: 0, 1>}
      %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16> offset = 0 len = 8 sizes = [] strides = [])
        aie.end
      }
      aiex.dma_start_task(%init)
      %last = scf.for %i = %c1 to %c4 step %c1 iter_args(%prev = %init) -> (index) {
        // CHECK: aie.dma_bd(%arg0 : memref<8xi16> offset = 0 len = 8 sizes = [] strides = []) {bd_id = 0 : i32, bd_id_window = array<i32: 0, 1>}
        %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<8xi16> offset = 0 len = 8 sizes = [] strides = [])
          aie.end
        }
        aiex.dma_start_task(%t)
        aiex.dma_free_task(%prev)
        scf.yield %t : index
      }
      aiex.dma_free_task(%last)
    }
  }
}
