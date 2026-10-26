//===- arbiter_deadlock_exhausted.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 | FileCheck %s

// Companion to arbiter_deadlock_avoidance.mlir: seven flows pass through the
// switchbox of memtile (0,1), and nothing tells the router they cannot
// deadlock one another. Nothing programs the S2MM channels of memtile (1,1),
// so each is assumed to wait on anything on its tile, which makes the six
// flows into it mutually dependent; flow 6 feeds the DMA the other six leave,
// whose MM2S channels are unprogrammed too. Seven flows that must not share
// an arbiter, and six arbiters, whatever the routing: the router fails and
// names the assumption behind the hazard.

// CHECK: error: Unable to find a legal routing: at tile (0, 1), no two of
// CHECK-SAME: packet flow (0, 0) DMA:0 -> (0, 1) DMA:0 (id 6) can share an arbiter, and each takes one there whatever the routing, but the switchbox has 6 free. For example,
// CHECK-SAME: Nothing in the design programs (1, 1) S2MM {{[0-5]}}, so it is assumed to wait on anything on its tile.

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %m  = aie.tile(0, 1)
    %m1 = aie.tile(1, 1)
    aie.packet_flow(0) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%m1, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%m, DMA : 1>  aie.packet_dest<%m1, DMA : 1> }
    aie.packet_flow(2) { aie.packet_source<%m, DMA : 2>  aie.packet_dest<%m1, DMA : 2> }
    aie.packet_flow(3) { aie.packet_source<%m, DMA : 3>  aie.packet_dest<%m1, DMA : 3> }
    aie.packet_flow(4) { aie.packet_source<%m, DMA : 4>  aie.packet_dest<%m1, DMA : 4> }
    aie.packet_flow(5) { aie.packet_source<%m, DMA : 5>  aie.packet_dest<%m1, DMA : 5> }
    aie.packet_flow(6) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
  }
}

// -----

// The same flows once memtile (1,1) programs its S2MM channels: each drains
// into its own buffer, independently of the others, so the six flows into it
// no longer depend on one another and two of them share an arbiter. Flow 6
// still depends on all of them and keeps arbiter 0 to itself.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[FLOW6:.*]] = aie.amsel<0> (0)
// CHECK-NOT:     aie.amsel<0>
// CHECK:         aie.amsel<{{[1-5]}}> (1)
// CHECK:         aie.masterset(DMA : 0, %[[FLOW6]])
// CHECK-NOT:     %[[FLOW6]])
// CHECK:         aie.rule(31, 6, %[[FLOW6]])
// CHECK-NOT:     %[[FLOW6]])
// CHECK:       }

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %m  = aie.tile(0, 1)
    %m1 = aie.tile(1, 1)
    %buf0 = aie.buffer(%m1) {sym_name = "buf0"} : memref<256xi32>
    %prod0 = aie.lock(%m1, 0) {init = 1 : i32, sym_name = "prod0"}
    %cons0 = aie.lock(%m1, 1) {init = 0 : i32, sym_name = "cons0"}
    %buf1 = aie.buffer(%m1) {sym_name = "buf1"} : memref<256xi32>
    %prod1 = aie.lock(%m1, 2) {init = 1 : i32, sym_name = "prod1"}
    %cons1 = aie.lock(%m1, 3) {init = 0 : i32, sym_name = "cons1"}
    %buf2 = aie.buffer(%m1) {sym_name = "buf2"} : memref<256xi32>
    %prod2 = aie.lock(%m1, 4) {init = 1 : i32, sym_name = "prod2"}
    %cons2 = aie.lock(%m1, 5) {init = 0 : i32, sym_name = "cons2"}
    %buf3 = aie.buffer(%m1) {sym_name = "buf3"} : memref<256xi32>
    %prod3 = aie.lock(%m1, 6) {init = 1 : i32, sym_name = "prod3"}
    %cons3 = aie.lock(%m1, 7) {init = 0 : i32, sym_name = "cons3"}
    %buf4 = aie.buffer(%m1) {sym_name = "buf4"} : memref<256xi32>
    %prod4 = aie.lock(%m1, 8) {init = 1 : i32, sym_name = "prod4"}
    %cons4 = aie.lock(%m1, 9) {init = 0 : i32, sym_name = "cons4"}
    %buf5 = aie.buffer(%m1) {sym_name = "buf5"} : memref<256xi32>
    %prod5 = aie.lock(%m1, 10) {init = 1 : i32, sym_name = "prod5"}
    %cons5 = aie.lock(%m1, 11) {init = 0 : i32, sym_name = "cons5"}
    %memtile_dma_1_1 = aie.memtile_dma(%m1) {
      %c1 = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:
      aie.use_lock(%prod0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf0 : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(%cons0, Release, %c1)
      aie.next_bd ^bb1
    ^bb3:
      %d1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:
      aie.use_lock(%prod1, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf1 : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(%cons1, Release, %c1)
      aie.next_bd ^bb4
    ^bb6:
      %d2 = aie.dma_start(S2MM, 2, ^bb7, ^bb9)
    ^bb7:
      aie.use_lock(%prod2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf2 : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(%cons2, Release, %c1)
      aie.next_bd ^bb7
    ^bb9:
      %d3 = aie.dma_start(S2MM, 3, ^bb10, ^bb12)
    ^bb10:
      aie.use_lock(%prod3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf3 : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(%cons3, Release, %c1)
      aie.next_bd ^bb10
    ^bb12:
      %d4 = aie.dma_start(S2MM, 4, ^bb13, ^bb15)
    ^bb13:
      aie.use_lock(%prod4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf4 : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(%cons4, Release, %c1)
      aie.next_bd ^bb13
    ^bb15:
      %d5 = aie.dma_start(S2MM, 5, ^bb16, ^bb18)
    ^bb16:
      aie.use_lock(%prod5, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf5 : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(%cons5, Release, %c1)
      aie.next_bd ^bb16
    ^bb18:
      aie.end
    }
    aie.packet_flow(0) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%m1, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%m, DMA : 1>  aie.packet_dest<%m1, DMA : 1> }
    aie.packet_flow(2) { aie.packet_source<%m, DMA : 2>  aie.packet_dest<%m1, DMA : 2> }
    aie.packet_flow(3) { aie.packet_source<%m, DMA : 3>  aie.packet_dest<%m1, DMA : 3> }
    aie.packet_flow(4) { aie.packet_source<%m, DMA : 4>  aie.packet_dest<%m1, DMA : 4> }
    aie.packet_flow(5) { aie.packet_source<%m, DMA : 5>  aie.packet_dest<%m1, DMA : 5> }
    aie.packet_flow(6) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
  }
}
