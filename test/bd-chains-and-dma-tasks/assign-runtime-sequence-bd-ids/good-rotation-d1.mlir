//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids --split-input-file %s | FileCheck %s

// Rolled ping-pong (depth D=1): the loop body frees the previous iteration's
// task via an iter_arg, so the BD id is held across one loop back-edge and must
// rotate through a window of W = D+1 = 2 physical ids. Both the prologue task
// and the body task rotate through the SAME window, so each carries bd_id = 0
// (window base) and bd_id_window = [0, 1].

// CHECK-LABEL: @rotation_d1
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @rotation_d1(%arg0: memref<8xi16>) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
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
