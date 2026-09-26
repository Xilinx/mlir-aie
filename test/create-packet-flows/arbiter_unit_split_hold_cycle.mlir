//===- arbiter_unit_split_hold_cycle.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s

// Id 15 goes from memtile (2,1) back to its own DMA:0 and DMA:5. Coming back
// in on one port, it would tie those two master ports to one arbiter, and then
// ids 17 (to DMA:0) and 23 (to DMA:5) share it and can hold it against each
// other. Neither flow is to blame: the multicast that ties the ports together
// must come back in on two ports instead.

// NOWARN-NOT: {{warning|error}}

// CHECK-LABEL: aie.switchbox(%mem_tile_2_1)
// CHECK-DAG:     %[[A:.*]] = aie.amsel<2> (0)
// CHECK-DAG:     %[[B:.*]] = aie.amsel<3> (0)
// CHECK-DAG:     aie.masterset(DMA : 0, %[[B]]) {keep_pkt_header = true}
// CHECK-DAG:     aie.masterset(DMA : 5, %[[A]]) {keep_pkt_header = true}
// CHECK-DAG:     aie.rule(31, 15, %[[A]])
// CHECK-DAG:     aie.rule(31, 15, %[[B]])
// CHECK-DAG:     aie.rule(31, 17, %[[B]])
// CHECK-DAG:     aie.rule(31, 23, %[[A]])

module {
  aie.device(npu1) {
    %t_1_0 = aie.tile(1, 0)
    %t_1_1 = aie.tile(1, 1)
    %t_1_3 = aie.tile(1, 3)
    %t_1_4 = aie.tile(1, 4)
    %t_1_5 = aie.tile(1, 5)
    %t_2_1 = aie.tile(2, 1)
    %t_2_2 = aie.tile(2, 2)
    %t_2_3 = aie.tile(2, 3)
    %t_2_4 = aie.tile(2, 4)
    %l_1_1_0 = aie.lock(%t_1_1, 0) {init = 2 : i32, sym_name = "l_1_1_0"}
    %l_1_1_1 = aie.lock(%t_1_1, 1) {init = 0 : i32, sym_name = "l_1_1_1"}
    %l_1_1_2 = aie.lock(%t_1_1, 2) {init = 2 : i32, sym_name = "l_1_1_2"}
    %l_1_1_3 = aie.lock(%t_1_1, 3) {init = 0 : i32, sym_name = "l_1_1_3"}
    %l_1_4_0 = aie.lock(%t_1_4, 0) {init = 1 : i32, sym_name = "l_1_4_0"}
    %l_1_4_1 = aie.lock(%t_1_4, 1) {init = 0 : i32, sym_name = "l_1_4_1"}
    %l_2_1_0 = aie.lock(%t_2_1, 0) {init = 2 : i32, sym_name = "l_2_1_0"}
    %l_2_1_1 = aie.lock(%t_2_1, 1) {init = 0 : i32, sym_name = "l_2_1_1"}
    %l_2_4_0 = aie.lock(%t_2_4, 0) {init = 2 : i32, sym_name = "l_2_4_0"}
    %l_2_4_1 = aie.lock(%t_2_4, 1) {init = 0 : i32, sym_name = "l_2_4_1"}
    %b_1_1_0 = aie.buffer(%t_1_1) {sym_name = "b_1_1_0"} : memref<32xi32>
    %b_1_1_1 = aie.buffer(%t_1_1) {sym_name = "b_1_1_1"} : memref<32xi32>
    %dma_1_1 = aie.memtile_dma(%t_1_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^p1)
    ^p0b0:
      aie.use_lock(%l_1_1_0, AcquireGreaterEqual, %one)
      aie.dma_bd(%b_1_1_0 : memref<32xi32> offset = 0 len = 32)
      aie.use_lock(%l_1_1_1, Release, %one)
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(S2MM, 5, ^p1b0, ^end)
    ^p1b0:
      aie.use_lock(%l_1_1_2, AcquireGreaterEqual, %one)
      aie.dma_bd(%b_1_1_1 : memref<32xi32> offset = 0 len = 32)
      aie.use_lock(%l_1_1_3, Release, %one)
      aie.next_bd ^p1b0
    ^end:
      aie.end
    }
    %b_1_4_0 = aie.buffer(%t_1_4) {sym_name = "b_1_4_0"} : memref<16xi32>
    %dma_1_4 = aie.mem(%t_1_4) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^end)
    ^p0b0:
      aie.use_lock(%l_1_4_0, AcquireGreaterEqual, %one)
      aie.dma_bd(%b_1_4_0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%l_1_4_1, Release, %one)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %b_1_5_0 = aie.buffer(%t_1_5) {sym_name = "b_1_5_0"} : memref<64xi32>
    %b_1_5_1 = aie.buffer(%t_1_5) {sym_name = "b_1_5_1"} : memref<16xi32>
    %dma_1_5 = aie.mem(%t_1_5) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_1_5_0 : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^p0b1
    ^p0b1:
      aie.dma_bd(%b_1_5_1 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %b_2_1_0 = aie.buffer(%t_2_1) {sym_name = "b_2_1_0"} : memref<64xi32>
    %b_2_1_1 = aie.buffer(%t_2_1) {sym_name = "b_2_1_1"} : memref<64xi32>
    %b_2_1_2 = aie.buffer(%t_2_1) {sym_name = "b_2_1_2"} : memref<16xi32>
    %b_2_1_3 = aie.buffer(%t_2_1) {sym_name = "b_2_1_3"} : memref<8xi32>
    %b_2_1_4 = aie.buffer(%t_2_1) {sym_name = "b_2_1_4"} : memref<16xi32>
    %dma_2_1 = aie.memtile_dma(%t_2_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 2, ^p0b0, ^p1)
    ^p0b0:
      aie.dma_bd(%b_2_1_0 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 12>}
      aie.next_bd ^p0b1
    ^p0b1:
      aie.dma_bd(%b_2_1_1 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 12>}
      aie.next_bd ^end
    ^p1:
      %d1 = aie.dma_start(MM2S, 3, ^p1b0, ^p2)
    ^p1b0:
      aie.dma_bd(%b_2_1_2 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^p2:
      %d2 = aie.dma_start(S2MM, 0, ^p2b0, ^p3)
    ^p2b0:
      aie.use_lock(%l_2_1_0, AcquireGreaterEqual, %one)
      aie.dma_bd(%b_2_1_3 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%l_2_1_1, Release, %one)
      aie.next_bd ^p2b0
    ^p3:
      %d3 = aie.dma_start(S2MM, 5, ^p3b0, ^end)
    ^p3b0:
      aie.dma_bd(%b_2_1_4 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^p3b0
    ^end:
      aie.end
    }
    %b_2_4_0 = aie.buffer(%t_2_4) {sym_name = "b_2_4_0"} : memref<32xi32>
    %b_2_4_1 = aie.buffer(%t_2_4) {sym_name = "b_2_4_1"} : memref<8xi32>
    %dma_2_4 = aie.mem(%t_2_4) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^p1)
    ^p0b0:
      aie.dma_bd(%b_2_4_0 : memref<32xi32> offset = 0 len = 32)
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(S2MM, 1, ^p1b0, ^end)
    ^p1b0:
      aie.use_lock(%l_2_4_0, AcquireGreaterEqual, %one)
      aie.dma_bd(%b_2_4_1 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%l_2_4_1, Release, %one)
      aie.next_bd ^p1b0
    ^end:
      aie.end
    }
    %core_1_4 = aie.core(%t_1_4) {
      %one = arith.constant 1 : i32
      aie.use_lock(%l_1_4_1, AcquireGreaterEqual, %one)
      aie.use_lock(%l_1_4_0, Release, %one)
      aie.end
    }
    %core_2_4 = aie.core(%t_2_4) {
      %one = arith.constant 1 : i32
      aie.use_lock(%l_2_4_0, Release, %one)
      aie.use_lock(%l_2_4_1, AcquireGreaterEqual, %one)
      aie.end
    }
    aie.packet_flow(11) { aie.packet_source<%t_2_1, DMA : 2> aie.packet_dest<%t_1_0, DMA : 1> }
    aie.packet_flow(16) { aie.packet_source<%t_2_1, DMA : 2> aie.packet_dest<%t_1_0, DMA : 1> aie.packet_dest<%t_1_3, Core : 0> aie.packet_dest<%t_1_5, Core : 0> }
    aie.packet_flow(12) { aie.packet_source<%t_2_1, DMA : 2> aie.packet_dest<%t_1_4, DMA : 0> }
    aie.packet_flow(31) { aie.packet_source<%t_2_4, Core : 0> aie.packet_dest<%t_1_4, Core : 0> }
    aie.packet_flow(23) { aie.packet_source<%t_2_4, Core : 0> aie.packet_dest<%t_2_1, DMA : 5> aie.packet_dest<%t_2_4, DMA : 1> }
    aie.packet_flow(14) { aie.packet_source<%t_2_2, Core : 0> aie.packet_dest<%t_1_4, DMA : 0> aie.packet_dest<%t_2_1, DMA : 5> aie.packet_dest<%t_2_2, DMA : 1> }
    aie.packet_flow(26) { aie.packet_source<%t_2_3, Core : 0> aie.packet_dest<%t_1_1, DMA : 1> } {keep_pkt_header = true}
    aie.packet_flow(17) { aie.packet_source<%t_2_4, DMA : 1> aie.packet_dest<%t_1_1, DMA : 0> aie.packet_dest<%t_1_4, DMA : 1> aie.packet_dest<%t_2_1, DMA : 0> }
    aie.packet_flow(15) { aie.packet_source<%t_2_1, DMA : 3> aie.packet_dest<%t_1_0, DMA : 0> aie.packet_dest<%t_2_1, DMA : 0> aie.packet_dest<%t_2_1, DMA : 5> } {keep_pkt_header = true}
  }
}
