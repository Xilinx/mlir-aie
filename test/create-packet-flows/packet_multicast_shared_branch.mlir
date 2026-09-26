//===- packet_multicast_shared_branch.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s

// The link from (0,2) down to (0,1) has four channels, and four streams need
// it: the two circuits, the packets from (0,2) Core:0, and id 23, bound for
// (0,0) and (0,1). If id 23 splits at (0,2) and takes two channels down, the
// link is one short and routing never converges. It must split at (0,1).

// NOWARN-NOT: {{warning|error}}

// CHECK-LABEL: aie.switchbox(%tile_0_2)
// CHECK:         aie.masterset(South : 3, %[[A:[0-9]+]])
// CHECK:           aie.rule(31, 23, %[[A]])

module {
  aie.device(npu2_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    %l_0_1_0 = aie.lock(%t_0_1, 0) {init = 1 : i32, sym_name = "l_0_1_0"}
    %l_0_1_1 = aie.lock(%t_0_1, 1) {init = 0 : i32, sym_name = "l_0_1_1"}
    %l_0_1_2 = aie.lock(%t_0_1, 2) {init = 1 : i32, sym_name = "l_0_1_2"}
    %l_0_1_3 = aie.lock(%t_0_1, 3) {init = 0 : i32, sym_name = "l_0_1_3"}
    %l_0_1_4 = aie.lock(%t_0_1, 4) {init = 2 : i32, sym_name = "l_0_1_4"}
    %l_0_1_5 = aie.lock(%t_0_1, 5) {init = 0 : i32, sym_name = "l_0_1_5"}
    %l_0_1_6 = aie.lock(%t_0_1, 6) {init = 2 : i32, sym_name = "l_0_1_6"}
    %l_0_1_7 = aie.lock(%t_0_1, 7) {init = 0 : i32, sym_name = "l_0_1_7"}
    %l_0_1_8 = aie.lock(%t_0_1, 8) {init = 1 : i32, sym_name = "l_0_1_8"}
    %l_0_1_9 = aie.lock(%t_0_1, 9) {init = 0 : i32, sym_name = "l_0_1_9"}
    %l_0_2_0 = aie.lock(%t_0_2, 0) {init = 1 : i32, sym_name = "l_0_2_0"}
    %l_0_2_1 = aie.lock(%t_0_2, 1) {init = 0 : i32, sym_name = "l_0_2_1"}
    %l_0_3_0 = aie.lock(%t_0_3, 0) {init = 0 : i32, sym_name = "l_0_3_0"}
    %l_0_3_1 = aie.lock(%t_0_3, 1) {init = 1 : i32, sym_name = "l_0_3_1"}
    %l_0_4_0 = aie.lock(%t_0_4, 0) {init = 2 : i32, sym_name = "l_0_4_0"}
    %l_0_4_1 = aie.lock(%t_0_4, 1) {init = 0 : i32, sym_name = "l_0_4_1"}
    %l_0_5_0 = aie.lock(%t_0_5, 0) {init = 2 : i32, sym_name = "l_0_5_0"}
    %l_0_5_1 = aie.lock(%t_0_5, 1) {init = 0 : i32, sym_name = "l_0_5_1"}
    %b_0_1_0 = aie.buffer(%t_0_1) {sym_name = "b_0_1_0"} : memref<64xi32>
    %b_0_1_1 = aie.buffer(%t_0_1) {sym_name = "b_0_1_1"} : memref<8xi32>
    %b_0_1_2 = aie.buffer(%t_0_1) {sym_name = "b_0_1_2"} : memref<8xi32>
    %b_0_1_3 = aie.buffer(%t_0_1) {sym_name = "b_0_1_3"} : memref<16xi32>
    %b_0_1_4 = aie.buffer(%t_0_1) {sym_name = "b_0_1_4"} : memref<8xi32>
    %b_0_1_5 = aie.buffer(%t_0_1) {sym_name = "b_0_1_5"} : memref<8xi32>
    %dma_0_1 = aie.memtile_dma(%t_0_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 4, ^p0b0, ^p1)
    ^p0b0:
      aie.use_lock(%l_0_1_1, AcquireGreaterEqual, %one)
      aie.dma_bd(%b_0_1_0 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 24>}
      aie.use_lock(%l_0_1_0, Release, %one)
      aie.next_bd ^end
    ^p1:
      %d1 = aie.dma_start(S2MM, 0, ^p1b0, ^p2)
    ^p1b0:
      aie.use_lock(%l_0_1_0, AcquireGreaterEqual, %one)
      aie.dma_bd(%b_0_1_1 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%l_0_1_1, Release, %one)
      aie.next_bd ^p1b0
    ^p2:
      %d2 = aie.dma_start(S2MM, 2, ^p2b0, ^p3)
    ^p2b0:
      aie.use_lock(%l_0_1_2, AcquireGreaterEqual, %one)
      aie.dma_bd(%b_0_1_2 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%l_0_1_3, Release, %one)
      aie.next_bd ^p2b0
    ^p3:
      %d3 = aie.dma_start(S2MM, 3, ^p3b0, ^p4)
    ^p3b0:
      aie.use_lock(%l_0_1_4, AcquireGreaterEqual, %one)
      aie.dma_bd(%b_0_1_3 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%l_0_1_5, Release, %one)
      aie.next_bd ^p3b0
    ^p4:
      %d4 = aie.dma_start(S2MM, 4, ^p4b0, ^p5)
    ^p4b0:
      aie.use_lock(%l_0_1_6, AcquireGreaterEqual, %one)
      aie.dma_bd(%b_0_1_4 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%l_0_1_7, Release, %one)
      aie.next_bd ^p4b0
    ^p5:
      %d5 = aie.dma_start(S2MM, 5, ^p5b0, ^end)
    ^p5b0:
      aie.use_lock(%l_0_1_8, AcquireGreaterEqual, %one)
      aie.dma_bd(%b_0_1_5 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%l_0_1_9, Release, %one)
      aie.next_bd ^p5b0
    ^end:
      aie.end
    }
    %b_0_2_0 = aie.buffer(%t_0_2) {sym_name = "b_0_2_0"} : memref<64xi32>
    %b_0_2_1 = aie.buffer(%t_0_2) {sym_name = "b_0_2_1"} : memref<32xi32>
    %dma_0_2 = aie.mem(%t_0_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 1, ^p0b0, ^p1)
    ^p0b0:
      aie.dma_bd(%b_0_2_0 : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^end
    ^p1:
      %d1 = aie.dma_start(S2MM, 1, ^p1b0, ^end)
    ^p1b0:
      aie.use_lock(%l_0_2_0, AcquireGreaterEqual, %one)
      aie.dma_bd(%b_0_2_1 : memref<32xi32> offset = 0 len = 32)
      aie.use_lock(%l_0_2_1, Release, %one)
      aie.next_bd ^p1b0
    ^end:
      aie.end
    }
    %b_0_3_0 = aie.buffer(%t_0_3) {sym_name = "b_0_3_0"} : memref<8xi32>
    %b_0_3_1 = aie.buffer(%t_0_3) {sym_name = "b_0_3_1"} : memref<64xi32>
    %b_0_3_2 = aie.buffer(%t_0_3) {sym_name = "b_0_3_2"} : memref<16xi32>
    %b_0_3_3 = aie.buffer(%t_0_3) {sym_name = "b_0_3_3"} : memref<8xi32>
    %dma_0_3 = aie.mem(%t_0_3) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^p0b0, ^p1)
    ^p0b0:
      aie.use_lock(%l_0_3_0, AcquireGreaterEqual, %one)
      aie.dma_bd(%b_0_3_0 : memref<8xi32> offset = 0 len = 8) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 16>}
      aie.use_lock(%l_0_3_1, Release, %one)
      aie.next_bd ^end
    ^p1:
      %d1 = aie.dma_start(MM2S, 1, ^p1b0, ^p2)
    ^p1b0:
      aie.dma_bd(%b_0_3_1 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 19>}
      aie.next_bd ^p1b0
    ^p2:
      %d2 = aie.dma_start(S2MM, 0, ^p2b0, ^p3)
    ^p2b0:
      aie.dma_bd(%b_0_3_2 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^p2b0
    ^p3:
      %d3 = aie.dma_start(S2MM, 1, ^p3b0, ^end)
    ^p3b0:
      aie.dma_bd(%b_0_3_3 : memref<8xi32> offset = 0 len = 8)
      aie.next_bd ^p3b0
    ^end:
      aie.end
    }
    %b_0_4_0 = aie.buffer(%t_0_4) {sym_name = "b_0_4_0"} : memref<32xi32>
    %dma_0_4 = aie.mem(%t_0_4) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 1, ^p0b0, ^end)
    ^p0b0:
      aie.use_lock(%l_0_4_0, AcquireGreaterEqual, %one)
      aie.dma_bd(%b_0_4_0 : memref<32xi32> offset = 0 len = 32)
      aie.use_lock(%l_0_4_1, Release, %one)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %b_0_5_0 = aie.buffer(%t_0_5) {sym_name = "b_0_5_0"} : memref<32xi32>
    %b_0_5_1 = aie.buffer(%t_0_5) {sym_name = "b_0_5_1"} : memref<32xi32>
    %dma_0_5 = aie.mem(%t_0_5) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^p1)
    ^p0b0:
      aie.use_lock(%l_0_5_0, AcquireGreaterEqual, %one)
      aie.dma_bd(%b_0_5_0 : memref<32xi32> offset = 0 len = 32)
      aie.use_lock(%l_0_5_1, Release, %one)
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(S2MM, 1, ^p1b0, ^end)
    ^p1b0:
      aie.dma_bd(%b_0_5_1 : memref<32xi32> offset = 0 len = 32)
      aie.next_bd ^p1b0
    ^end:
      aie.end
    }
    %core_0_2 = aie.core(%t_0_2) {
      %one = arith.constant 1 : i32
      aie.use_lock(%l_0_2_0, Release, %one)
      aie.use_lock(%l_0_2_1, AcquireGreaterEqual, %one)
      aie.end
    }
    %core_0_3 = aie.core(%t_0_3) {
      %one = arith.constant 1 : i32
      aie.use_lock(%l_0_3_0, Release, %one)
      aie.use_lock(%l_0_3_1, AcquireGreaterEqual, %one)
      aie.end
    }
    %core_0_4 = aie.core(%t_0_4) {
      %one = arith.constant 1 : i32
      aie.use_lock(%l_0_4_0, Release, %one)
      aie.use_lock(%l_0_4_1, AcquireGreaterEqual, %one)
      aie.end
    }
    %core_0_5 = aie.core(%t_0_5) {
      %one = arith.constant 1 : i32
      aie.use_lock(%l_0_5_1, AcquireGreaterEqual, %one)
      aie.use_lock(%l_0_5_0, Release, %one)
      aie.end
    }
    aie.flow(%t_0_5, Core : 0, %t_0_1, DMA : 2)
    aie.flow(%t_0_2, DMA : 1, %t_0_0, DMA : 1)
    aie.flow(%t_0_2, DMA : 1, %t_0_5, Core : 0)
    aie.packet_flow(30) { aie.packet_source<%t_0_1, DMA : 4> aie.packet_dest<%t_0_1, DMA : 1> }
    aie.packet_flow(0) { aie.packet_source<%t_0_1, DMA : 4> aie.packet_dest<%t_0_0, DMA : 0> aie.packet_dest<%t_0_3, DMA : 1> aie.packet_dest<%t_0_4, DMA : 0> }
    aie.packet_flow(23) { aie.packet_source<%t_0_3, DMA : 0> aie.packet_dest<%t_0_0, DMA : 0> aie.packet_dest<%t_0_1, DMA : 3> aie.packet_dest<%t_0_5, DMA : 1> }
    aie.packet_flow(7) { aie.packet_source<%t_0_3, DMA : 0> aie.packet_dest<%t_0_3, Core : 0> aie.packet_dest<%t_0_5, DMA : 0> } {priority_route = true}
    aie.packet_flow(19) { aie.packet_source<%t_0_3, DMA : 1> aie.packet_dest<%t_0_2, Core : 0> aie.packet_dest<%t_0_2, DMA : 1> aie.packet_dest<%t_0_4, Core : 0> }
    aie.packet_flow(22) { aie.packet_source<%t_0_2, Core : 0> aie.packet_dest<%t_0_1, DMA : 4> }
    aie.packet_flow(3) { aie.packet_source<%t_0_2, Core : 0> aie.packet_dest<%t_0_0, DMA : 0> aie.packet_dest<%t_0_1, DMA : 0> aie.packet_dest<%t_0_1, DMA : 5> }
  }
}
