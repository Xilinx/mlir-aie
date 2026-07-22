//===- objectfifo_rearm_binding.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file %s | FileCheck %s

// aie.objectfifo_rearm_binding round-trips: it carries the non-shim DMA
// endpoints of an objectFIFO (channel_tiles paired with channel_dirs/indices)
// and its producer/consumer locks (locks paired with lock_inits), so a resident
// re-arm stays resolvable after the objectFIFO op is erased.

// CHECK-LABEL: aie.device(npu2)
// CHECK: %[[CT:.*]] = aie.tile(0, 3)
// CHECK: %[[MT:.*]] = aie.tile(0, 1)
// CHECK: %[[PL:.*]] = aie.lock(%[[CT]], 0)
// CHECK: %[[CL:.*]] = aie.lock(%[[CT]], 1)
// CHECK: aie.objectfifo_rearm_binding @of_rearm channels(%[[CT]], %[[MT]] : index, index) locks(%[[PL]], %[[CL]] : index, index) {channel_dirs = array<i32: 0, 1>, channel_indices = array<i32: 0, 0>, lock_inits = array<i32: 2, 0>}
module {
  aie.device(npu2) {
    %ct = aie.tile(0, 3)
    %mt = aie.tile(0, 1)
    %pl = aie.lock(%ct, 0) {init = 2 : i32}
    %cl = aie.lock(%ct, 1) {init = 0 : i32}
    aie.objectfifo_rearm_binding @of_rearm channels(%ct, %mt : index, index) locks(%pl, %cl : index, index) {channel_dirs = array<i32: 0, 1>, channel_indices = array<i32: 0, 0>, lock_inits = array<i32: 2, 0>}
  }
}

// -----

// A degenerate binding with no channels and no locks (a shim-only fifo has
// nothing on-tile to re-arm) is legal and round-trips.
// CHECK-LABEL: aie.device(npu1)
// CHECK: aie.objectfifo_rearm_binding @empty_rearm channels() locks() {channel_dirs = array<i32>, channel_indices = array<i32>, lock_inits = array<i32>}
module {
  aie.device(npu1) {
    aie.objectfifo_rearm_binding @empty_rearm channels() locks() {channel_dirs = array<i32>, channel_indices = array<i32>, lock_inits = array<i32>}
  }
}
