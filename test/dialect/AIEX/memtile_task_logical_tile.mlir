//===- memtile_task_logical_tile.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file %s -o /dev/null
// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

// A pinned logical task tile shares the physical resources' location.
// CHECK-LABEL: aie.device(npu1_2col)
// CHECK: %[[TILE:.*]] = aie.tile(0, 1)
// CHECK-NOT: aie.logical_tile
// CHECK: aie.buffer(%[[TILE]])
// CHECK: aie.lock(%[[TILE]],
// CHECK: aiex.dma_configure_task(%[[TILE]], MM2S, 4)
// CHECK: aie.use_lock
// CHECK: aie.dma_bd
aie.device(npu1_2col) {
  %tile = aie.tile(0, 1)
  %logical = aie.logical_tile<MemTile>(0, 1)
  %buf = aie.buffer(%tile) : memref<16xi32>
  %lock = aie.lock(%tile, 0) {init = 1 : i32}
  aie.runtime_sequence() {
    %task = aiex.dma_configure_task(%logical, MM2S, 4) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%lock, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.end
    }
  }
}

// -----

// The resource tiles may also be logical while the task tile is physical.
// CHECK-LABEL: aie.device(npu1_2col)
// CHECK: %[[TILE:.*]] = aie.tile(0, 1)
// CHECK-NOT: aie.logical_tile
// CHECK: aie.buffer(%[[TILE]])
// CHECK: aie.lock(%[[TILE]],
// CHECK: aiex.dma_configure_task(%[[TILE]], MM2S, 5)
// CHECK: aie.use_lock
// CHECK: aie.dma_bd
aie.device(npu1_2col) {
  %tile = aie.tile(0, 1)
  %logical = aie.logical_tile<MemTile>(0, 1)
  %buf = aie.buffer(%logical) : memref<16xi32>
  %lock = aie.lock(%logical, 0) {init = 1 : i32}
  aie.runtime_sequence() {
    %task = aiex.dma_configure_task(%tile, MM2S, 5) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%lock, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.end
    }
  }
}
