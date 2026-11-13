//===- bad_memtile_task_local_channels.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// On AIE2-family memtiles, DMA channels 4 and 5 reach only the local memory
// banks and locks; channels 0-3 also reach the west and east neighbours. A
// runtime task follows the same rule as aie.memtile_dma.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu1_2col) {
  %t01 = aie.tile(0, 1)
  %t11 = aie.tile(1, 1)
  %buf = aie.buffer(%t01) {sym_name = "buf"} : memref<16xi32>
  // expected-note@+1 {{lock}}
  %east = aie.lock(%t11, 0) {init = 1 : i32}
  aie.runtime_sequence() {
    %t = aiex.dma_configure_task(%t01, MM2S, 4) {
      %c1 = arith.constant 1 : i32
      // expected-error@+1 {{'aie.use_lock' op is in a task on memtile DMA channel 4 and attempts to access a non-local lock; only channels below 4 reach neighbouring tiles}}
      aie.use_lock(%east, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
      aie.end
    }
  }
}

// -----

aie.device(npu1_2col) {
  %t01 = aie.tile(0, 1)
  %t11 = aie.tile(1, 1)
  // expected-note@+1 {{buffer}}
  %west = aie.buffer(%t01) {sym_name = "west"} : memref<16xi32>
  aie.runtime_sequence() {
    %t = aiex.dma_configure_task(%t11, S2MM, 5) {
      // expected-error@+1 {{'aie.dma_bd' op is in a task on memtile DMA channel 5 and attempts to access a non-local buffer; only channels below 4 reach neighbouring tiles}}
      aie.dma_bd(%west : memref<16xi32> offset = 0 len = 16) {bd_id = 24 : i32}
      aie.end
    }
  }
}

// -----

// Channel 3 may use the neighbour's lock and buffer; channel 4 may use local
// ones.
aie.device(npu1_2col) {
  %t01 = aie.tile(0, 1)
  %t11 = aie.tile(1, 1)
  %local = aie.buffer(%t01) {sym_name = "local"} : memref<16xi32>
  %east_buf = aie.buffer(%t11) {sym_name = "east_buf"} : memref<16xi32>
  %local_lock = aie.lock(%t01, 0) {init = 1 : i32}
  %east_lock = aie.lock(%t11, 0) {init = 1 : i32}
  aie.runtime_sequence() {
    %a = aiex.dma_configure_task(%t01, MM2S, 3) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%east_lock, AcquireGreaterEqual, %c1)
      aie.dma_bd(%east_buf : memref<16xi32> offset = 0 len = 16) {bd_id = 25 : i32}
      aie.end
    }
    %b = aiex.dma_configure_task(%t01, MM2S, 4) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%local_lock, AcquireGreaterEqual, %c1)
      aie.dma_bd(%local : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
      aie.end
    }
  }
}

// -----

// A constant is not a lock operation.
aie.device(npu1_2col) {
  %tile = aie.tile(0, 1)
  %buf = aie.buffer(%tile) : memref<16xi32>
  aie.runtime_sequence() {
    %task = aiex.dma_configure_task(%tile, MM2S, 4) {
      %lock = arith.constant 0 : index
      %c1 = arith.constant 1 : i32
      // expected-error@+1 {{'aie.use_lock' op requires a lock defined by aie.lock in a task on memtile DMA channel 4}}
      aie.use_lock(%lock, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.end
    }
  }
}

// -----

// A block argument has no defining operation.
aie.device(npu1_2col) {
  %tile = aie.tile(0, 1)
  %buf = aie.buffer(%tile) : memref<16xi32>
  aie.runtime_sequence(%lock : index) {
    %task = aiex.dma_configure_task(%tile, MM2S, 5) {
      %c1 = arith.constant 1 : i32
      // expected-error@+1 {{'aie.use_lock' op requires a lock defined by aie.lock in a task on memtile DMA channel 5}}
      aie.use_lock(%lock, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.end
    }
  }
}

// -----

// Pinned logical tiles at different coordinates still use non-local resources.
aie.device(npu1_2col) {
  %tile = aie.tile(0, 1)
  %logical = aie.logical_tile<MemTile>(1, 1)
  // expected-note@+1 {{buffer}}
  %buf = aie.buffer(%tile) : memref<16xi32>
  // expected-note@+1 {{lock}}
  %lock = aie.lock(%tile, 0) {init = 1 : i32}
  aie.runtime_sequence() {
    %task = aiex.dma_configure_task(%logical, MM2S, 4) {
      %c1 = arith.constant 1 : i32
      // expected-error@+1 {{'aie.use_lock' op is in a task on memtile DMA channel 4 and attempts to access a non-local lock; only channels below 4 reach neighbouring tiles}}
      aie.use_lock(%lock, AcquireGreaterEqual, %c1)
      // expected-error@+1 {{'aie.dma_bd' op is in a task on memtile DMA channel 4 and attempts to access a non-local buffer; only channels below 4 reach neighbouring tiles}}
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.end
    }
  }
}
