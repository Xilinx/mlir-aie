//===- rearm_binding_populate.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -split-input-file --aie-assign-bd-ids %s | FileCheck %s

// Once BD ids are assigned, --aie-assign-bd-ids folds each objectFIFO re-arm
// binding's head BD id + (biased) repeat count onto the binding, so the later
// aiex.dma_channel_reset_for lowering reads them straight off the op instead of
// re-scanning the emitted aie.mem chain.

// Head BD id 5 (user-assigned, kept) and repeat_count 3 off the resident
// dma_start are folded onto @of_rearm.
// CHECK-LABEL: @populate
// CHECK: aie.objectfifo_rearm_binding @of_rearm channels(%{{.*}} : index) locks(%{{.*}}, %{{.*}} : index, index) {channel_dirs = array<i32: 0>, channel_indices = array<i32: 0>, head_bd_ids = array<i32: 5>, lock_inits = array<i32: 1, 0>, repeat_counts = array<i32: 3>}
module @populate {
  aie.device(npu2) {
    %t03 = aie.tile(0, 3)
    %buf = aie.buffer(%t03) : memref<64xi32>
    %pl = aie.lock(%t03, 0) {init = 1 : i32}
    %cl = aie.lock(%t03, 1) {init = 0 : i32}
    %mem = aie.mem(%t03) {
      %s = aie.dma_start(S2MM, 0, ^bd, ^end, repeat_count = 3)
    ^bd:
      %c1 = arith.constant 1 : i32
      aie.use_lock(%pl, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%cl, Release, %c1)
      aie.next_bd ^bd
    ^end:
      aie.end
    }
    aie.objectfifo_rearm_binding @of_rearm channels(%t03 : index) locks(%pl, %cl : index, index) {channel_dirs = array<i32: 0>, channel_indices = array<i32: 0>, lock_inits = array<i32: 1, 0>}
    aie.runtime_sequence() {
      aiex.dma_channel_reset_for(@of_rearm)
    }
  }
}

// -----

// A hand-authored binding that already carries head_bd_ids / repeat_counts is
// left untouched (it may back no objectFIFO chain at all).
// CHECK-LABEL: @already_populated
// CHECK: head_bd_ids = array<i32: 9>
// CHECK-SAME: repeat_counts = array<i32: 1>
module @already_populated {
  aie.device(npu2) {
    %t03 = aie.tile(0, 3)
    %pl = aie.lock(%t03, 0) {init = 1 : i32}
    aie.objectfifo_rearm_binding @of_rearm channels(%t03 : index) locks(%pl : index) {channel_dirs = array<i32: 0>, channel_indices = array<i32: 0>, lock_inits = array<i32: 1>, head_bd_ids = array<i32: 9>, repeat_counts = array<i32: 1>}
  }
}

// -----

// A binding endpoint with no resident DMA channel cannot be resolved, so it is
// left unpopulated (the reset_for lowering later diagnoses the gap).
// CHECK-LABEL: @unresolvable
// CHECK-NOT: head_bd_ids
module @unresolvable {
  aie.device(npu2) {
    %t03 = aie.tile(0, 3)
    %pl = aie.lock(%t03, 0) {init = 1 : i32}
    aie.objectfifo_rearm_binding @of_rearm channels(%t03 : index) locks(%pl : index) {channel_dirs = array<i32: 0>, channel_indices = array<i32: 0>, lock_inits = array<i32: 1>}
  }
}
