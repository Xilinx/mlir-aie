//===- memtile_task_unplaced_tile.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file %s -o /dev/null

// Resource locality is checked after placement resolves their coordinates.
aie.device(npu1_2col) {
  %tile = aie.tile(0, 1)
  %logical = aie.logical_tile<MemTile>(?, ?)
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

// -----

aie.device(npu1_2col) {
  %tile = aie.tile(0, 1)
  %logical = aie.logical_tile<MemTile>(0, ?)
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

// -----

aie.device(npu1_2col) {
  %tile = aie.tile(0, 1)
  %logical = aie.logical_tile<MemTile>(?, 1)
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

// -----

// The task and its resources may share an unplaced tile.
aie.device(npu1_2col) {
  %logical = aie.logical_tile<MemTile>(?, ?)
  %buf = aie.buffer(%logical) : memref<16xi32>
  %lock = aie.lock(%logical, 0) {init = 1 : i32}
  aie.runtime_sequence() {
    %task = aiex.dma_configure_task(%logical, MM2S, 4) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%lock, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.end
    }
  }
}
