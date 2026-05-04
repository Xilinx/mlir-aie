//===- test_place_buffers.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

// Buffer fits comfortably in CoreTile L1 (npu1 has 64KB) — placer is a
// pass-through.
// CHECK-LABEL: @buffer_fits_pinned
module @buffer_fits_pinned {
  aie.device(npu1) {
    // CHECK-DAG: %[[T:.*]] = aie.tile(2, 3)
    %t = aie.logical_tile<CoreTile>(2, 3)
    %b = aie.buffer(%t) : memref<128xi64>
    aie.core(%t) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Unconstrained CoreTile with a buffer that fits — greedy placement at (0,2).
// CHECK-LABEL: @buffer_fits_unconstrained
module @buffer_fits_unconstrained {
  aie.device(npu1) {
    // CHECK-DAG: %[[T:.*]] = aie.tile(0, 2)
    %t = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.buffer(%t) : memref<1024xi32>
    aie.core(%t) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Multiple buffers on one tile sum against capacity (3 * 8KB + 1KB stack).
// CHECK-LABEL: @multi_buffer_sum_fits
module @multi_buffer_sum_fits {
  aie.device(npu1) {
    // CHECK-DAG: %[[T:.*]] = aie.tile(2, 3)
    %t = aie.logical_tile<CoreTile>(2, 3)
    %b0 = aie.buffer(%t) : memref<2048xi32>
    %b1 = aie.buffer(%t) : memref<2048xi32>
    %b2 = aie.buffer(%t) : memref<2048xi32>
    aie.core(%t) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Buffers on physical aie.tile are skipped (validated by AIEAssignBuffers).
// CHECK-LABEL: @physical_tile_buffer_skipped
module @physical_tile_buffer_skipped {
  aie.device(npu1) {
    // CHECK-DAG: %[[PT:.*]] = aie.tile(2, 3)
    %pt = aie.tile(2, 3)
    %big = aie.buffer(%pt) : memref<65536xi32>
    // CHECK-DAG: %[[LT:.*]] = aie.tile(2, 4)
    %lt = aie.logical_tile<CoreTile>(2, 4)
    %small = aie.buffer(%lt) : memref<128xi32>
    aie.core(%lt) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// MemTile per-LogicalTileOp upper bound (npu1 MemTile = 512KB).
// CHECK-LABEL: @memtile_buffer_fits
module @memtile_buffer_fits {
  aie.device(npu1) {
    // CHECK-DAG: %[[MT:.*]] = aie.tile(0, 1)
    %mt = aie.logical_tile<MemTile>(0, 1)
    %b = aie.buffer(%mt) : memref<1024xi32>
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// AIE2P (npu2) target-model dispatch.
// CHECK-LABEL: @buffer_fits_npu2
module @buffer_fits_npu2 {
  aie.device(npu2) {
    // CHECK-DAG: %[[T:.*]] = aie.tile(0, 2)
    %t = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.buffer(%t) : memref<1024xi32>
    aie.core(%t) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}
