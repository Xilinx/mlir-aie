//===- dma_channel_reset_for.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-lower-dma-channel-reset-for %s | FileCheck %s

// aiex.dma_channel_reset_for expands, via the fifo's re-arm binding, into the
// resident re-arm of each non-shim channel:
//   1. aiex.dma_channel_reset (drain the channel queue),
//   2. aiex.set_lock per producer/consumer lock (re-arm to the fifo init),
//   3. aiex.npu.push_queue START_QUEUE re-push (required on aie2p: a DMA channel
//      has no enable bit, so the only way to restart it is a queue push).
// The re-push is emitted as aiex.npu.push_queue with bd_id + repeat read off the
// resident aie.dma_start; aie-dma-to-npu lowers it to the START_QUEUE write.

// Core tile (0,3), S2MM channel 0. Head BD id 5, repeat_count 3 (already the N-1
// biased value on the dma_start), no token (not a shim S2MM).
// CHECK-LABEL: @core_tile
// CHECK: aiex.dma_channel_reset(%[[T:.*]], S2MM, 0)
// CHECK: aiex.set_lock(%{{.*}}, 1)
// CHECK: aiex.set_lock(%{{.*}}, 0)
// The bd_id/repeat constant order is not part of the contract; match them
// order-independently.
// CHECK-DAG: %[[BD:.*]] = arith.constant 5 : i32
// CHECK-DAG: %[[RC:.*]] = arith.constant 3 : i32
// CHECK: aiex.npu.push_queue(0, 3, S2MM : 0) bd_id %[[BD]] repeat %[[RC]] {issue_token = false}
// CHECK-NOT: aiex.dma_channel_reset_for
// The now-unreferenced binding is dropped.
// CHECK-NOT: aie.objectfifo_rearm_binding
module @core_tile {
  aie.device(npu2) {
    %t03 = aie.tile(0, 3)
    %buf = aie.buffer(%t03) : memref<64xi32>
    %pl = aie.lock(%t03, 0) {init = 1 : i32}
    %cl = aie.lock(%t03, 1) {init = 0 : i32}
    %mem = aie.mem(%t03) {
      %s = aie.dma_start(S2MM, 0, ^bd, ^end, repeat_count = 3)
    ^bd:
      %c1 = arith.constant 1 : i32
      aie.use_lock(%pl, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%cl, Release, %c1)
      aie.next_bd ^bd
    ^end:
      aie.end
    }
    aie.objectfifo_rearm_binding @of_rearm channels(%t03 : index) locks(%pl, %cl : index, index) {channel_dirs = array<i32: 0>, channel_indices = array<i32: 0>, lock_inits = array<i32: 1, 0>}
    aie.runtime_sequence() {
      aiex.dma_channel_reset_for(@of_rearm)
    }
  }
}

// -----

// Mem tile (0,1), S2MM channel 0. The mem tile START_BD_ID field is 6 bits (48
// BDs) not 4, so head BD id 24 rides through push_queue whole; aie-dma-to-npu
// (tile-class masked) then keeps it 24, not truncated to 24 & 0xF = 8.
// CHECK-LABEL: @mem_tile
// CHECK: aiex.dma_channel_reset(%{{.*}}, S2MM, 0)
// CHECK: %[[BD:.*]] = arith.constant 24 : i32
// CHECK: aiex.npu.push_queue(0, 1, S2MM : 0) bd_id %[[BD]] repeat %{{.*}} {issue_token = false}
module @mem_tile {
  aie.device(npu2) {
    %mt = aie.tile(0, 1)
    %buf = aie.buffer(%mt) : memref<64xi32>
    %pl = aie.lock(%mt, 0) {init = 2 : i32}
    %cl = aie.lock(%mt, 1) {init = 0 : i32}
    %mem = aie.memtile_dma(%mt) {
      %s = aie.dma_start(S2MM, 0, ^bd, ^end, repeat_count = 0)
    ^bd:
      %c1 = arith.constant 1 : i32
      aie.use_lock(%pl, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%cl, Release, %c1)
      aie.next_bd ^bd
    ^end:
      aie.end
    }
    aie.objectfifo_rearm_binding @r channels(%mt : index) locks(%pl, %cl : index, index) {channel_dirs = array<i32: 0>, channel_indices = array<i32: 0>, lock_inits = array<i32: 2, 0>}
    aie.runtime_sequence() {
      aiex.dma_channel_reset_for(@r)
    }
  }
}

// -----

// Same core-tile op on npu1 (AIE2): head BD id 2, emitted as a push_queue.
// CHECK-LABEL: @npu1_core
// CHECK: %[[BD:.*]] = arith.constant 2 : i32
// CHECK: aiex.npu.push_queue(0, 3, S2MM : 0) bd_id %[[BD]] repeat %{{.*}} {issue_token = false}
module @npu1_core {
  aie.device(npu1) {
    %t03 = aie.tile(0, 3)
    %buf = aie.buffer(%t03) : memref<64xi32>
    %pl = aie.lock(%t03, 0) {init = 1 : i32}
    %cl = aie.lock(%t03, 1) {init = 0 : i32}
    %mem = aie.mem(%t03) {
      %s = aie.dma_start(S2MM, 0, ^bd, ^end)
    ^bd:
      %c1 = arith.constant 1 : i32
      aie.use_lock(%pl, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%cl, Release, %c1)
      aie.next_bd ^bd
    ^end:
      aie.end
    }
    aie.objectfifo_rearm_binding @of_rearm channels(%t03 : index) locks(%pl, %cl : index, index) {channel_dirs = array<i32: 0>, channel_indices = array<i32: 0>, lock_inits = array<i32: 1, 0>}
    aie.runtime_sequence() {
      aiex.dma_channel_reset_for(@of_rearm)
    }
  }
}
