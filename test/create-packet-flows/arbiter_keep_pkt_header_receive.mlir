//===- arbiter_keep_pkt_header_receive.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 | FileCheck %s
// RUN: sed 's/keep_pkt_header = false/keep_pkt_header = true/' %s | not aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" 2>&1 | FileCheck %s --check-prefix=KEPT

// Cores (0..5, 2) each send one 64-byte packet into their own S2MM channel of
// memtile (0,1), whose locks take exactly 64 bytes before the join on MM2S 0
// drains them. Dropping the header, every packet fits, nothing stalls, and the
// seven packet master ports at (0,1) may share its six arbiters. Keeping it,
// each receiver gets 68 bytes and fills, so any two joined flows can deadlock
// on a shared arbiter, and so can each of them with the join's own output.

// CHECK-NOT:   warning
// CHECK-NOT:   error
// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK-COUNT-7: aie.masterset

// KEPT: error: Unable to find a legal routing: at tile (0, 1), no two of
// KEPT-SAME: can share an arbiter, and each takes one there whatever the routing, but the switchbox has 6 free.

module {
  aie.device(npu2) {
    %m  = aie.tile(0, 1)
    %t0 = aie.tile(0, 2)
    %t1 = aie.tile(1, 2)
    %t2 = aie.tile(2, 2)
    %t3 = aie.tile(3, 2)
    %t4 = aie.tile(4, 2)
    %t5 = aie.tile(5, 2)
    %t6 = aie.tile(0, 3)

    aie.packet_flow(0) { aie.packet_source<%t0, DMA : 0> aie.packet_dest<%m, DMA : 0> } {keep_pkt_header = false}
    aie.packet_flow(1) { aie.packet_source<%t1, DMA : 0> aie.packet_dest<%m, DMA : 1> } {keep_pkt_header = false}
    aie.packet_flow(2) { aie.packet_source<%t2, DMA : 0> aie.packet_dest<%m, DMA : 2> } {keep_pkt_header = false}
    aie.packet_flow(3) { aie.packet_source<%t3, DMA : 0> aie.packet_dest<%m, DMA : 3> } {keep_pkt_header = false}
    aie.packet_flow(4) { aie.packet_source<%t4, DMA : 0> aie.packet_dest<%m, DMA : 4> } {keep_pkt_header = false}
    aie.packet_flow(5) { aie.packet_source<%t5, DMA : 0> aie.packet_dest<%m, DMA : 5> } {keep_pkt_header = false}
    aie.packet_flow(6) { aie.packet_source<%m, DMA : 0> aie.packet_dest<%t6, DMA : 1> }

    %b0 = aie.buffer(%t0) : memref<16xi32>
    aie.mem(%t0) {
      %0 = aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%b0 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 0, pkt_type = 0>}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %b1 = aie.buffer(%t1) : memref<16xi32>
    aie.mem(%t1) {
      %0 = aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%b1 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 1, pkt_type = 0>}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %b2 = aie.buffer(%t2) : memref<16xi32>
    aie.mem(%t2) {
      %0 = aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%b2 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 2, pkt_type = 0>}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %b3 = aie.buffer(%t3) : memref<16xi32>
    aie.mem(%t3) {
      %0 = aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%b3 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 3, pkt_type = 0>}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %b4 = aie.buffer(%t4) : memref<16xi32>
    aie.mem(%t4) {
      %0 = aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%b4 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 4, pkt_type = 0>}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %b5 = aie.buffer(%t5) : memref<16xi32>
    aie.mem(%t5) {
      %0 = aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%b5 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 5, pkt_type = 0>}
      aie.next_bd ^end
    ^end:
      aie.end
    }

    %mb0 = aie.buffer(%m) : memref<17xi32>
    %mb1 = aie.buffer(%m) : memref<17xi32>
    %mb2 = aie.buffer(%m) : memref<17xi32>
    %mb3 = aie.buffer(%m) : memref<17xi32>
    %mb4 = aie.buffer(%m) : memref<17xi32>
    %mb5 = aie.buffer(%m) : memref<17xi32>
    %p0 = aie.lock(%m, 0) {init = 1 : i32}
    %c0 = aie.lock(%m, 1) {init = 0 : i32}
    %p1 = aie.lock(%m, 2) {init = 1 : i32}
    %c1 = aie.lock(%m, 3) {init = 0 : i32}
    %p2 = aie.lock(%m, 4) {init = 1 : i32}
    %c2 = aie.lock(%m, 5) {init = 0 : i32}
    %p3 = aie.lock(%m, 6) {init = 1 : i32}
    %c3 = aie.lock(%m, 7) {init = 0 : i32}
    %p4 = aie.lock(%m, 8) {init = 1 : i32}
    %c4 = aie.lock(%m, 9) {init = 0 : i32}
    %p5 = aie.lock(%m, 10) {init = 1 : i32}
    %c5 = aie.lock(%m, 11) {init = 0 : i32}
    aie.memtile_dma(%m) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^r0, ^s1)
    ^r0:
      aie.use_lock(%p0, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb0 : memref<17xi32> offset = 0 len = 16)
      aie.use_lock(%c0, Release, %one)
      aie.next_bd ^r0
    ^s1:
      %1 = aie.dma_start(S2MM, 1, ^r1, ^s2)
    ^r1:
      aie.use_lock(%p1, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb1 : memref<17xi32> offset = 0 len = 16)
      aie.use_lock(%c1, Release, %one)
      aie.next_bd ^r1
    ^s2:
      %2 = aie.dma_start(S2MM, 2, ^r2, ^s3)
    ^r2:
      aie.use_lock(%p2, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb2 : memref<17xi32> offset = 0 len = 16)
      aie.use_lock(%c2, Release, %one)
      aie.next_bd ^r2
    ^s3:
      %3 = aie.dma_start(S2MM, 3, ^r3, ^s4)
    ^r3:
      aie.use_lock(%p3, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb3 : memref<17xi32> offset = 0 len = 16)
      aie.use_lock(%c3, Release, %one)
      aie.next_bd ^r3
    ^s4:
      %4 = aie.dma_start(S2MM, 4, ^r4, ^s5)
    ^r4:
      aie.use_lock(%p4, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb4 : memref<17xi32> offset = 0 len = 16)
      aie.use_lock(%c4, Release, %one)
      aie.next_bd ^r4
    ^s5:
      %5 = aie.dma_start(S2MM, 5, ^r5, ^join)
    ^r5:
      aie.use_lock(%p5, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb5 : memref<17xi32> offset = 0 len = 16)
      aie.use_lock(%c5, Release, %one)
      aie.next_bd ^r5
    ^join:
      %6 = aie.dma_start(MM2S, 0, ^j0, ^end)
    ^j0:
      aie.use_lock(%c0, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb0 : memref<17xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 6, pkt_type = 0>}
      aie.use_lock(%p0, Release, %one)
      aie.next_bd ^j1
    ^j1:
      aie.use_lock(%c1, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb1 : memref<17xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 6, pkt_type = 0>}
      aie.use_lock(%p1, Release, %one)
      aie.next_bd ^j2
    ^j2:
      aie.use_lock(%c2, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb2 : memref<17xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 6, pkt_type = 0>}
      aie.use_lock(%p2, Release, %one)
      aie.next_bd ^j3
    ^j3:
      aie.use_lock(%c3, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb3 : memref<17xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 6, pkt_type = 0>}
      aie.use_lock(%p3, Release, %one)
      aie.next_bd ^j4
    ^j4:
      aie.use_lock(%c4, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb4 : memref<17xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 6, pkt_type = 0>}
      aie.use_lock(%p4, Release, %one)
      aie.next_bd ^j5
    ^j5:
      aie.use_lock(%c5, AcquireGreaterEqual, %one)
      aie.dma_bd(%mb5 : memref<17xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 6, pkt_type = 0>}
      aie.use_lock(%p5, Release, %one)
      aie.next_bd ^j0
    ^end:
      aie.end
    }
  }
}
