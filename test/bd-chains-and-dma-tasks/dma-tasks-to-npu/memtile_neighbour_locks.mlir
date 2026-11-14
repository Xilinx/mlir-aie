//===- memtile_neighbour_locks.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A runtime memtile BD can acquire a lock of one tile and release a lock of
// another. Each lock ID is offset by the base index of the tile that owns the
// lock (west +0, local +64, east +128), not by the release lock's base.

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// Memtile (0,1): acquire local lock 0, release east lock 1.
// CHECK: aiex.npu.writebd
// CHECK-SAME: lock_acq_enable = 1
// CHECK-SAME: lock_acq_id = 64
// CHECK-SAME: lock_rel_id = 129
// CHECK-SAME: row = 1
// Memtile (0,1): acquire east lock 2, release local lock 3.
// CHECK: aiex.npu.writebd
// CHECK-SAME: lock_acq_enable = 1
// CHECK-SAME: lock_acq_id = 130
// CHECK-SAME: lock_rel_id = 67
// CHECK-SAME: row = 1
// Memtile (1,1): acquire west lock 3, release local lock 1.
// CHECK: aiex.npu.writebd
// CHECK-SAME: column = 1
// CHECK-SAME: lock_acq_enable = 1
// CHECK-SAME: lock_acq_id = 3
// CHECK-SAME: lock_rel_id = 65
// CHECK-SAME: row = 1
aie.device(npu1_2col) {
  %t01 = aie.tile(0, 1)
  %t11 = aie.tile(1, 1)
  %buf = aie.buffer(%t01) { address = 0 : i32 } : memref<16xi32>
  %buf1 = aie.buffer(%t11) { address = 0 : i32 } : memref<16xi32>
  %lock_0_1_id0 = aie.lock(%t01, 0) { init = 1 : i32 }
  %lock_0_1_id3 = aie.lock(%t01, 3) { init = 0 : i32 }
  %lock_1_1_id1 = aie.lock(%t11, 1) { init = 0 : i32 }
  %lock_1_1_id2 = aie.lock(%t11, 2) { init = 1 : i32 }
  aie.runtime_sequence(%arg0: memref<16xi32>) {
    %t0 = aiex.dma_configure_task(%t01, MM2S, 0) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%lock_0_1_id0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) { bd_id = 0 : i32 }
      aie.use_lock(%lock_1_1_id1, Release, %c1)
      aie.end
    }
    %t1 = aiex.dma_configure_task(%t01, S2MM, 0) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%lock_1_1_id2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) { bd_id = 1 : i32 }
      aie.use_lock(%lock_0_1_id3, Release, %c1)
      aie.end
    }
    %t2 = aiex.dma_configure_task(%t11, MM2S, 0) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%lock_0_1_id3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf1 : memref<16xi32> offset = 0 len = 16) { bd_id = 0 : i32 }
      aie.use_lock(%lock_1_1_id1, Release, %c1)
      aie.end
    }
  }
}
