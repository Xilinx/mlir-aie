//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids --split-input-file %s | FileCheck %s

// Rolled ping-pong (D=1) of a multi-dma_bd chain (C=2). The whole 2-BD chain
// rotates as a unit, so each of the C=2 descriptors needs its own W=2 window:
// the first dma_bd of the chain rotates through [0, 1] and the second through
// [2, 3] (total C*W = 4 ids reserved). The prologue chain and the body chain
// rotate through the same per-position windows.

// CHECK-LABEL: @rotation_chain_c2
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @rotation_chain_c2(%arg0: memref<8xi16>) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // CHECK: aie.dma_bd(%arg0 : memref<8xi16>, 0, 4) {bd_id = 0 : i32, bd_id_window = array<i32: 0, 1>
    // CHECK: aie.dma_bd(%arg0 : memref<8xi16>, 4, 4) {bd_id = 2 : i32, bd_id_window = array<i32: 2, 3>
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 4)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<8xi16>, 4, 4)
      aie.end
    }
    aiex.dma_start_task(%init)
    %last = scf.for %i = %c1 to %c4 step %c1 iter_args(%prev = %init) -> (index) {
      // CHECK: aie.dma_bd(%arg0 : memref<8xi16>, 0, 4) {bd_id = 0 : i32, bd_id_window = array<i32: 0, 1>
      // CHECK: aie.dma_bd(%arg0 : memref<8xi16>, 4, 4) {bd_id = 2 : i32, bd_id_window = array<i32: 2, 3>
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 4)
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%arg0 : memref<8xi16>, 4, 4)
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%prev)
      scf.yield %t : index
    }
    aiex.dma_free_task(%last)
  }
}
