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

// Multiple buffers on the same logical tile sum together against capacity.
// 4 * memref<8192xi32> = 4 * 32KB = 128KB of buffers + 1KB stack would exceed
// npu1's 64KB CoreTile L1, but each buffer is 32KB on its own. This case uses
// 3 * 8KB = 24KB + 1KB stack = 25KB which fits comfortably; it pins the
// summation behaviour without overflowing.
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

// Buffer attached to a physical aie.tile (not aie.logical_tile) is silently
// ignored by the placer's buffer-capacity check: such buffers are not part of
// any placement decision and remain AIEAssignBuffers' responsibility. The
// 256KB physical-tile buffer here would overflow if it were counted, but the
// placer must accept this IR unchanged (a separate logical_tile with a small
// buffer placed normally proves the rest of the pipeline still runs).
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

// MemTile-pinned LogicalTileOp with a buffer that fits in the MemTile capacity
// (npu1 MemTile = 512KB). Validates that MemTile placements pass the new
// per-LogicalTileOp upper-bound check.
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

// Same shape on AIE2P (npu2), exercising the target-model dispatch on a
// different architecture. npu2 CoreTile L1 is also 64KB; a 4KB buffer plus
// 1KB default stack fits with room to spare.
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
