// RUN: aie-opt --split-input-file --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-objectfifo-allocate %s -o %t
// RUN: aie-opt --split-input-file --aie-objectfifo-allocate %t -o %t2
// RUN: diff %t %t2
// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --aie-objectfifo-lower-dmas %s -o /dev/null

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A pinned local-only channel forces the small pool home, not a blanket ban
// on spilling. The larger pool can still be split between two memory tiles.
module @pinned_local {
  aie.device(npu2) {
    %home = aie.tile(0, 1)
    %reserved = aie.buffer(%home) : memref<196608xi8>
    aie.objectfifo.pool @large(%home) {depth = 2 : i32} : memref<147456xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 147456 : i32}
    }
    aie.objectfifo.pool @small(%home) {depth = 2 : i32} : memref<32768xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 32768 : i32}
    }
    aie.objectfifo.dma_endpoint @large_dma(%home) fills @large
    aie.objectfifo.dma_endpoint @small_dma(%home) fills @small {channelIndex = 5 : i32}
  }
}
// CHECK-LABEL: module @pinned_local
// CHECK-DAG: %[[H:.*]] = aie.tile(0, 1)
// CHECK-DAG: %[[N:.*]] = aie.tile(1, 1)
// CHECK-DAG: aie.buffer(%[[H]]) {sym_name = "small_buff_0"}
// CHECK-DAG: aie.buffer(%[[H]]) {sym_name = "small_buff_1"}
// CHECK-DAG: aie.buffer(%[[H]]) {sym_name = "large_buff_0"}
// CHECK-DAG: aie.buffer(%[[N]]) {sym_name = "large_buff_1"}
// CHECK: @large_dma(%[[H]]) fills @large {channelIndex = 0 : i32}
// CHECK: @small_dma(%[[H]]) fills @small {channelIndex = 5 : i32}

// -----

// Pool metadata does not determine reachability: these buffers are local to
// all six readers, but remote to the writer. Generated locks must follow them.
module @shared_existing {
  aie.device(npu2) {
    %reader = aie.tile(0, 1)
    %writer = aie.tile(1, 1)
    %b = aie.buffer(%reader) {sym_name = "existing"} : memref<16xi32>
    aie.objectfifo.pool @shared(%writer) {
      depth = 1 : i32, buffers = [@existing]
    } : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @r0(%reader) drains @shared
    aie.objectfifo.dma_endpoint @r1(%reader) drains @shared
    aie.objectfifo.dma_endpoint @r2(%reader) drains @shared
    aie.objectfifo.dma_endpoint @r3(%reader) drains @shared
    aie.objectfifo.dma_endpoint @r4(%reader) drains @shared
    aie.objectfifo.dma_endpoint @r5(%reader) drains @shared
    aie.objectfifo.dma_endpoint @w(%writer) fills @shared
  }
}
// CHECK-LABEL: module @shared_existing
// CHECK-NOT: shared_buff
// CHECK: @r4({{.*}}) drains @shared {channelIndex = 4 : i32}
// CHECK: @r5({{.*}}) drains @shared {channelIndex = 5 : i32}
// CHECK: @w({{.*}}) fills @shared {channelIndex = 0 : i32}

// -----

// The emptier neighbor is not reachable by the left endpoint. All users,
// rather than just the pool's nominal tile, constrain the spill destination.
module @shared_spill {
  aie.device(npu2) {
    %left = aie.tile(0, 1)
    %home = aie.tile(1, 1)
    %right = aie.tile(2, 1)
    %left_reserved = aie.buffer(%left) : memref<196608xi8>
    %home_reserved = aie.buffer(%home) : memref<524288xi8>
    aie.objectfifo.pool @shared(%home) {depth = 2 : i32} : memref<32768xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 32768 : i32}
    }
    aie.objectfifo.dma_endpoint @reader(%left) drains @shared {channelIndex = 5 : i32}
    aie.objectfifo.dma_endpoint @writer(%home) fills @shared
  }
}
// CHECK-LABEL: module @shared_spill
// CHECK: %[[LEFT:.*]] = aie.tile(0, 1)
// CHECK-DAG: aie.buffer(%[[LEFT]]) {sym_name = "shared_buff_0"}
// CHECK-DAG: aie.buffer(%[[LEFT]]) {sym_name = "shared_buff_1"}
// CHECK: @reader(%[[LEFT]]) drains @shared {channelIndex = 5 : i32}

// -----

// A preallocated split pool is remote from both endpoints. Its existing
// objects are neither moved nor replicated to satisfy the other endpoint.
module @split_existing {
  aie.device(npu2) {
    %left = aie.tile(0, 1)
    %right = aie.tile(1, 1)
    %a = aie.buffer(%left) {sym_name = "a"} : memref<16xi32>
    %b = aie.buffer(%right) {sym_name = "b"} : memref<16xi32>
    aie.objectfifo.pool @split(%left) {
      depth = 2 : i32, buffers = [@a, @b]
    } : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @reader(%left) drains @split
    aie.objectfifo.dma_endpoint @writer(%right) fills @split
  }
}
// CHECK-LABEL: module @split_existing
// CHECK-NOT: split_buff
// CHECK: @split({{.*}}) {buffers = [@a, @b], depth = 2 : i32}
// CHECK: @reader({{.*}}) drains @split {channelIndex = 0 : i32}
// CHECK: @writer({{.*}}) fills @split {channelIndex = 0 : i32}

// -----

// Pinned low channels consume the neighbor-capable budget even though their
// own buffers are local. Repair MM2S demand as well as S2MM demand.
module @reserved_channels {
  aie.device(npu2) {
    %home = aie.tile(0, 1)
    %reserved = aie.buffer(%home) : memref<196608xi8>
    %fixed = aie.buffer(%home) {sym_name = "fixed"} : memref<16xi32>
    aie.objectfifo.pool @fixed_pool(%home) {
      depth = 1 : i32, buffers = [@fixed]
    } : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @r0(%home) drains @fixed_pool {channelIndex = 0 : i32}
    aie.objectfifo.dma_endpoint @r1(%home) drains @fixed_pool {channelIndex = 1 : i32}
    aie.objectfifo.dma_endpoint @r2(%home) drains @fixed_pool {channelIndex = 2 : i32}
    aie.objectfifo.pool @large(%home) {depth = 2 : i32} : memref<147456xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 147456 : i32}
    }
    aie.objectfifo.pool @a(%home) {depth = 2 : i32} : memref<32768xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 32768 : i32}
    }
    aie.objectfifo.pool @b(%home) {depth = 2 : i32} : memref<32768xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 32768 : i32}
    }
    aie.objectfifo.dma_endpoint @a_dma(%home) drains @a
    aie.objectfifo.dma_endpoint @b_dma(%home) drains @b
  }
}
// CHECK-LABEL: module @reserved_channels
// CHECK: @r0({{.*}}) drains @fixed_pool {channelIndex = 0 : i32}
// CHECK: @r1({{.*}}) drains @fixed_pool {channelIndex = 1 : i32}
// CHECK: @r2({{.*}}) drains @fixed_pool {channelIndex = 2 : i32}
// CHECK: @a_dma({{.*}}) drains @a {channelIndex = 3 : i32}
// CHECK: @b_dma({{.*}}) drains @b {channelIndex = 4 : i32}

// -----

// A local-only channel on another user can reserve the pool's objects and
// generated locks there. The nominal pool tile is not a physical owner.
module @shared_local_channel {
  aie.device(npu2) {
    %home = aie.tile(0, 1)
    %reader = aie.tile(1, 1)
    aie.objectfifo.pool @shared(%home) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @reader(%reader) drains @shared {channelIndex = 5 : i32}
    aie.objectfifo.dma_endpoint @writer(%home) fills @shared
  }
}
// CHECK-LABEL: module @shared_local_channel
// CHECK-DAG: %[[LOCKS:.*]] = aie.tile(0, 1)
// CHECK-DAG: %[[READER:.*]] = aie.tile(1, 1)
// CHECK-DAG: aie.lock(%[[READER]]) {{.*}}sym_name = "shared_prod_lock_0"
// CHECK-DAG: aie.buffer(%[[READER]]) {sym_name = "shared_buff_0"}
// CHECK-DAG: aie.buffer(%[[READER]]) {sym_name = "shared_buff_1"}
// CHECK: @reader(%[[READER]]) drains @shared {channelIndex = 5 : i32}
// CHECK: @writer(%[[LOCKS]]) fills @shared {channelIndex = 0 : i32}

// -----

// Unresolved logical coordinates do not prove that sharing is impossible.
// Keep the buffers at their declared tile and defer affinity to placement.
module @unplaced_shared {
  aie.device(npu2) {
    %home = aie.logical_tile<MemTile>(?, ?)
    %reader = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo.pool @shared(%home) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @reader(%reader) drains @shared
  }
}
// CHECK-LABEL: module @unplaced_shared
// CHECK: %[[UNPLACED:.*]] = aie.logical_tile<MemTile>(?, ?)
// CHECK: aie.buffer(%[[UNPLACED]]) {sym_name = "shared_buff_0"}
// CHECK: @reader({{.*}}) drains @shared {channelIndex = 0 : i32}

// -----

// A distinct unresolved buffer tile must reserve a neighbor-capable channel
// before local endpoints can consume that restricted range.
module @partly_placed {
  aie.device(npu2) {
    %home = aie.tile(0, 1)
    %other = aie.logical_tile<MemTile>(1, ?)
    %a = aie.buffer(%home) {sym_name = "a"} : memref<16xi32>
    %b = aie.buffer(%other) {sym_name = "b"} : memref<16xi32>
    aie.objectfifo.pool @local(%home) {
      depth = 1 : i32, buffers = [@a]
    } : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.pool @remote(%home) {
      depth = 1 : i32, buffers = [@b]
    } : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @local0(%home) drains @local
    aie.objectfifo.dma_endpoint @local1(%home) drains @local
    aie.objectfifo.dma_endpoint @local2(%home) drains @local
    aie.objectfifo.dma_endpoint @local3(%home) drains @local
    aie.objectfifo.dma_endpoint @remote_dma(%home) drains @remote
  }
}
// CHECK-LABEL: module @partly_placed
// CHECK: @local0({{.*}}) drains @local {channelIndex = 1 : i32}
// CHECK: @local3({{.*}}) drains @local {channelIndex = 4 : i32}
// CHECK: @remote_dma({{.*}}) drains @remote {channelIndex = 0 : i32}
