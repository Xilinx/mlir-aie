//===- buffer_clear.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -split-input-file --aie-lower-buffer-clear %s | FileCheck %s --implicit-check-not=aiex.buffer_clear

// Each aiex.buffer_clear lowers to one aiex.npu.blockwrite of `length` zero
// words at `address`, a tile-local offset. Calls sharing a `length` share one
// zero-data global; the dedup collapses only that global, not the per-call
// instruction, so each npu.blockwrite still carries its own words at runtime.
module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_1 = aie.tile(0, 1)

    // CHECK: memref.global "private" constant @blockwrite_data_0 : memref<8xi32> = dense<0>
    // CHECK: memref.global "private" constant @blockwrite_data_1 : memref<4xi32> = dense<0>
    aie.runtime_sequence() {
      // Core tile (0, 2), 8 words at offset 0.
      // CHECK: %[[G0:.*]] = memref.get_global @blockwrite_data_0 : memref<8xi32>
      // CHECK: aiex.npu.blockwrite(%[[G0]]) {address = 0 : ui32, column = 0 : i32, row = 2 : i32} : memref<8xi32>
      aiex.buffer_clear(%tile_0_2, 0, 8)

      // Same length, different offset: reuses @blockwrite_data_0.
      // CHECK: %[[G1:.*]] = memref.get_global @blockwrite_data_0 : memref<8xi32>
      // CHECK: aiex.npu.blockwrite(%[[G1]]) {address = 32 : ui32, column = 0 : i32, row = 2 : i32} : memref<8xi32>
      aiex.buffer_clear(%tile_0_2, 32, 8)

      // Mem tile (0, 1), 4 words: a different length gets its own global.
      // CHECK: %[[G2:.*]] = memref.get_global @blockwrite_data_1 : memref<4xi32>
      // CHECK: aiex.npu.blockwrite(%[[G2]]) {address = 4 : ui32, column = 0 : i32, row = 1 : i32} : memref<4xi32>
      aiex.buffer_clear(%tile_0_1, 4, 4)
    }
  }
}

// -----

// npu1 and npu2 inherit the same memory sizes from AIE2TargetModel, and a
// tile-local blockwrite is device-independent across the family.
module {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    aie.runtime_sequence() {
      // CHECK: memref.global "private" constant @blockwrite_data_0 : memref<2xi32> = dense<0>
      // CHECK: %[[G:.*]] = memref.get_global @blockwrite_data_0 : memref<2xi32>
      // CHECK: aiex.npu.blockwrite(%[[G]]) {address = 0 : ui32, column = 0 : i32, row = 2 : i32} : memref<2xi32>
      aiex.buffer_clear(%tile_0_2, 0, 2)
    }
  }
}
