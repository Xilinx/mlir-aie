//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids --split-input-file %s | FileCheck %s

// Control-flow-aware BD-ID allocation. Tasks inside scf.for / scf.if are
// allocated against a real per-tile pool with same-iteration reuse and arm
// exclusivity.

// scf.for, freed within the body => BD id reused every iteration.
// CHECK-LABEL: @for_body_reuse
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @for_body_reuse(%arg0: memref<8xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      // CHECK: aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 0 : i32}
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// scf.if: both arms freed within the arm => mutually exclusive, share BD id 0.
// CHECK-LABEL: @if_arm_share
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @if_arm_share(%arg0: memref<8xi16>, %c: i1) {
    scf.if %c {
      // CHECK: aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 0 : i32}
      %y = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      aiex.dma_start_task(%y)
      aiex.dma_free_task(%y)
    } else {
      // CHECK: aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 0 : i32}
      %z = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      aiex.dma_start_task(%z)
      aiex.dma_free_task(%z)
    }
  }
}

// -----

// A task live across the if while an in-arm task is configured => distinct ids
// (0 for the live-across task, 1 for the in-arm task).
// CHECK-LABEL: @if_live_across
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @if_live_across(%arg0: memref<8xi16>, %c: i1) {
    // CHECK: aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 0 : i32}
    %x = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
      aie.end
    }
    aiex.dma_start_task(%x)
    scf.if %c {
      // CHECK: aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 1 : i32}
      %y = aiex.dma_configure_task(%tile_0_0, MM2S, 1) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      aiex.dma_start_task(%y)
      aiex.dma_free_task(%y)
    }
    aiex.dma_free_task(%x)
  }
}
