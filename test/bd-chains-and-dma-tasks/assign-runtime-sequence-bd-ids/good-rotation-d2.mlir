//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids --split-input-file %s | FileCheck %s

// Free-two-iterations-ago (depth D=2): two iter_args carry the two most-recent
// tasks and the oldest is freed each iteration, so the BD is held across two
// loop back-edges and rotates through a window of W = D+1 = 3 ids. All three
// participating configures (the two prologue tasks and the body task) rotate
// through the same window [0, 1, 2].

// CHECK-LABEL: @rotation_d2
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @rotation_d2(%arg0: memref<8xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // CHECK: aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 0 : i32, bd_id_window = array<i32: 0, 1, 2>, bd_id_window_group = {{[0-9]+}} : i32}
    %i0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
      aie.end
    }
    // CHECK: aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 0 : i32, bd_id_window = array<i32: 0, 1, 2>, bd_id_window_group = {{[0-9]+}} : i32}
    %i1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
      aie.end
    }
    %r:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%p2 = %i0, %p1 = %i1) -> (index, index) {
      // CHECK: aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 0 : i32, bd_id_window = array<i32: 0, 1, 2>, bd_id_window_group = {{[0-9]+}} : i32}
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%p2)
      scf.yield %p1, %t : index, index
    }
    aiex.dma_free_task(%r#0)
    aiex.dma_free_task(%r#1)
  }
}
