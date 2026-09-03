//===- bank_aware_reserved_data_zero_size.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A zero-sized buffer occupies no bytes, so it must not fragment the free
// region it sits inside. largestFreeRun, which this pass's reserved_data_size
// check shares with AIETargetLdScript, sees the space above the stack as one
// 64512-byte run, and not as two runs split at the zero-sized buffer's address.

// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck %s

// CHECK-LABEL: module @zero_size_buffer_does_not_fragment_reserved_data
// CHECK: %mid = aie.buffer(%tile_0_2) {address = 30016 : i32, mem_bank = 1 : i32, sym_name = "mid"} : memref<0xi32>
module @zero_size_buffer_does_not_fragment_reserved_data {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    // Pinned strictly inside bank 1, away from either end of the tile, so a
    // sweep that (incorrectly) treats a zero-length interval as a split
    // point reports two disjoint runs (28992 and 35520 bytes) instead of one
    // 64512-byte run.
    %mid = aie.buffer(%tile_0_2) {address = 30016 : i32, sym_name = "mid"} : memref<0xi32>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, reserved_data_size = 64512 : i32}
  }
}
