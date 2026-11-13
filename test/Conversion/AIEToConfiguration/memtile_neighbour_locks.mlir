//===- memtile_neighbour_locks.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A memtile BD can acquire a lock of one tile and release a lock of another.
// Each lock ID is offset by the base index of the tile that owns the lock
// (west +0, local +64, east +128), not by the base of the last lock seen.

// RUN: aie-opt --convert-aie-to-transaction %s | FileCheck %s

// BD word 7 on an AIE2 memtile: rel_val[30:24] rel_id[23:16] acq_en[15]
// acq_val[14:8] acq_id[7:0].
// Memtile (0,1) BD 0: acquire local lock 0 (64), release east lock 1 (129):
//   0x8181FF40 = -2122186944.
// Memtile (0,1) BD 1: acquire east lock 2 (130), release local lock 3 (67):
//   0x8143FF82 = -2126250110.
// CHECK: memref.global {{.*}} = dense<[16, 131072, 0, 0, 0, 0, 0, -2122186944, 16, 131072, 0, 0, 0, 0, 0, -2126250110]>
// Memtile (1,1) BD 0: acquire west lock 3 (3), release local lock 1 (65):
//   0x8141FF03 = -2126381309.
// CHECK: memref.global {{.*}} = dense<[16, 131072, 0, 0, 0, 0, 0, -2126381309]>

aie.device(npu1_2col) {
  %t01 = aie.tile(0, 1)
  %t11 = aie.tile(1, 1)
  %buf = aie.buffer(%t01) { address = 0 : i32 } : memref<16xi32>
  %buf1 = aie.buffer(%t11) { address = 0 : i32 } : memref<16xi32>
  %lock_0_1_id0 = aie.lock(%t01, 0) { init = 1 : i32 }
  %lock_0_1_id3 = aie.lock(%t01, 3) { init = 0 : i32 }
  %lock_1_1_id1 = aie.lock(%t11, 1) { init = 0 : i32 }
  %lock_1_1_id2 = aie.lock(%t11, 2) { init = 1 : i32 }
  // Memtile (0,1): local acquire with east release, and the reverse.
  aie.memtile_dma(%t01) {
    aie.dma_start(MM2S, 0, ^bd0, ^dma1)
  ^dma1:
    aie.dma_start(S2MM, 0, ^bd1, ^end)
  ^bd0:
    %c1 = arith.constant 1 : i32
    aie.use_lock(%lock_0_1_id0, AcquireGreaterEqual, %c1)
    aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) { bd_id = 0 : i32 }
    aie.use_lock(%lock_1_1_id1, Release, %c1)
    aie.next_bd ^end
  ^bd1:
    %c2 = arith.constant 1 : i32
    aie.use_lock(%lock_1_1_id2, AcquireGreaterEqual, %c2)
    aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) { bd_id = 1 : i32 }
    aie.use_lock(%lock_0_1_id3, Release, %c2)
    aie.next_bd ^end
  ^end:
    aie.end
  }
  // Memtile (1,1): west acquire with local release.
  aie.memtile_dma(%t11) {
    aie.dma_start(MM2S, 0, ^bd0, ^end)
  ^bd0:
    %c1 = arith.constant 1 : i32
    aie.use_lock(%lock_0_1_id3, AcquireGreaterEqual, %c1)
    aie.dma_bd(%buf1 : memref<16xi32> offset = 0 len = 16) { bd_id = 0 : i32 }
    aie.use_lock(%lock_1_1_id1, Release, %c1)
    aie.next_bd ^end
  ^end:
    aie.end
  }
}
