//===- packet_msel_uncontended_tile.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s

// With these DMA programs no two streams into memtile (5,1) can stall each
// other, so any of them may share an arbiter there. Filling arbiter 0 one
// flow at a time used up its msels on a design with arbiters to spare, and
// then crashed. The master sets now spread over two arbiters.

// Nothing programs the DMAs, so the design itself may deadlock.
// NOWARN: warning: Flows can deadlock however they are routed
// NOWARN-NOT: {{warning|error}}

// CHECK-LABEL: aie.switchbox(%mem_tile_5_1)
// CHECK-DAG:     aie.masterset(DMA : 2, %[[A:[0-9]+]]){{$}}
// CHECK-DAG:     aie.masterset(DMA : 5, %[[B:[0-9]+]]) {keep_pkt_header = true}
// CHECK-DAG:     %[[A]] = aie.amsel<{{[0-9]}}> (0)
// CHECK-DAG:     %[[B]] = aie.amsel<{{[0-9]}}> (0)

module {
  aie.device(npu2) {
    %t_4_1 = aie.tile(4, 1)
    %t_4_2 = aie.tile(4, 2)
    %t_4_4 = aie.tile(4, 4)
    %t_4_5 = aie.tile(4, 5)
    %t_5_1 = aie.tile(5, 1)
    %t_5_2 = aie.tile(5, 2)
    %t_5_4 = aie.tile(5, 4)
    %t_5_5 = aie.tile(5, 5)
    %t_6_0 = aie.tile(6, 0)
    %t_6_1 = aie.tile(6, 1)
    %t_6_2 = aie.tile(6, 2)
    %t_6_3 = aie.tile(6, 3)
    %t_6_4 = aie.tile(6, 4)
    %t_6_5 = aie.tile(6, 5)
    %l_4_1_0 = aie.lock(%t_4_1, 0) {init = 2 : i32, sym_name = "l_4_1_0"}
    %l_4_1_1 = aie.lock(%t_4_1, 1) {init = 0 : i32, sym_name = "l_4_1_1"}
    %l_5_1_0 = aie.lock(%t_5_1, 0) {init = 1 : i32, sym_name = "l_5_1_0"}
    %l_5_1_1 = aie.lock(%t_5_1, 1) {init = 0 : i32, sym_name = "l_5_1_1"}
    %l_6_1_0 = aie.lock(%t_6_1, 0) {init = 2 : i32, sym_name = "l_6_1_0"}
    %l_6_1_1 = aie.lock(%t_6_1, 1) {init = 0 : i32, sym_name = "l_6_1_1"}
    %b_4_1_0 = aie.buffer(%t_4_1) {sym_name = "b_4_1_0"} : memref<8xi32>
    %dma_4_1 = aie.memtile_dma(%t_4_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 4, ^p0b0, ^end)
    ^p0b0:
      aie.use_lock(%l_4_1_0, AcquireGreaterEqual, %one)
      aie.dma_bd(%b_4_1_0 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%l_4_1_1, Release, %one)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %b_4_5_0 = aie.buffer(%t_4_5) {sym_name = "b_4_5_0"} : memref<16xi32>
    %dma_4_5 = aie.mem(%t_4_5) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_4_5_0 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %b_5_1_0 = aie.buffer(%t_5_1) {sym_name = "b_5_1_0"} : memref<8xi32>
    %b_5_1_1 = aie.buffer(%t_5_1) {sym_name = "b_5_1_1"} : memref<8xi32>
    %b_5_1_2 = aie.buffer(%t_5_1) {sym_name = "b_5_1_2"} : memref<8xi32>
    %dma_5_1 = aie.memtile_dma(%t_5_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^p1)
    ^p0b0:
      aie.dma_bd(%b_5_1_0 : memref<8xi32> offset = 0 len = 8)
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(S2MM, 2, ^p1b0, ^p2)
    ^p1b0:
      aie.use_lock(%l_5_1_0, AcquireGreaterEqual, %one)
      aie.dma_bd(%b_5_1_1 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%l_5_1_1, Release, %one)
      aie.next_bd ^p1b0
    ^p2:
      %d2 = aie.dma_start(S2MM, 5, ^p2b0, ^end)
    ^p2b0:
      aie.dma_bd(%b_5_1_2 : memref<8xi32> offset = 0 len = 8)
      aie.next_bd ^p2b0
    ^end:
      aie.end
    }
    %b_5_2_0 = aie.buffer(%t_5_2) {sym_name = "b_5_2_0"} : memref<16xi32>
    %b_5_2_1 = aie.buffer(%t_5_2) {sym_name = "b_5_2_1"} : memref<8xi32>
    %dma_5_2 = aie.mem(%t_5_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_5_2_0 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 10>}
      aie.next_bd ^p0b1
    ^p0b1:
      aie.dma_bd(%b_5_2_1 : memref<8xi32> offset = 0 len = 8) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 10>}
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %b_5_4_0 = aie.buffer(%t_5_4) {sym_name = "b_5_4_0"} : memref<16xi32>
    %b_5_4_1 = aie.buffer(%t_5_4) {sym_name = "b_5_4_1"} : memref<64xi32>
    %dma_5_4 = aie.mem(%t_5_4) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 1, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_5_4_0 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 19>}
      aie.next_bd ^p0b1
    ^p0b1:
      aie.dma_bd(%b_5_4_1 : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %b_6_1_0 = aie.buffer(%t_6_1) {sym_name = "b_6_1_0"} : memref<64xi32>
    %b_6_1_1 = aie.buffer(%t_6_1) {sym_name = "b_6_1_1"} : memref<16xi32>
    %b_6_1_2 = aie.buffer(%t_6_1) {sym_name = "b_6_1_2"} : memref<16xi32>
    %dma_6_1 = aie.memtile_dma(%t_6_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 1, ^p0b0, ^p1)
    ^p0b0:
      aie.dma_bd(%b_6_1_0 : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^p0b1
    ^p0b1:
      aie.dma_bd(%b_6_1_1 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^p1:
      %d1 = aie.dma_start(S2MM, 1, ^p1b0, ^end)
    ^p1b0:
      aie.use_lock(%l_6_1_0, AcquireGreaterEqual, %one)
      aie.dma_bd(%b_6_1_2 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%l_6_1_1, Release, %one)
      aie.next_bd ^p1b0
    ^end:
      aie.end
    }
    %b_6_2_0 = aie.buffer(%t_6_2) {sym_name = "b_6_2_0"} : memref<16xi32>
    %b_6_2_1 = aie.buffer(%t_6_2) {sym_name = "b_6_2_1"} : memref<32xi32>
    %dma_6_2 = aie.mem(%t_6_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 1, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_6_2_0 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^p0b1
    ^p0b1:
      aie.dma_bd(%b_6_2_1 : memref<32xi32> offset = 0 len = 32)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %b_6_4_0 = aie.buffer(%t_6_4) {sym_name = "b_6_4_0"} : memref<16xi32>
    %b_6_4_1 = aie.buffer(%t_6_4) {sym_name = "b_6_4_1"} : memref<64xi32>
    %dma_6_4 = aie.mem(%t_6_4) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_6_4_0 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^p0b1
    ^p0b1:
      aie.dma_bd(%b_6_4_1 : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.packet_flow(24) { aie.packet_source<%t_6_5, Core : 0> aie.packet_dest<%t_5_1, DMA : 2> aie.packet_dest<%t_5_1, DMA : 5> }
    aie.packet_flow(11) { aie.packet_source<%t_4_2, Core : 0> aie.packet_dest<%t_4_1, DMA : 0> aie.packet_dest<%t_5_1, DMA : 5> }
    aie.packet_flow(25) { aie.packet_source<%t_4_2, Core : 0> aie.packet_dest<%t_5_1, DMA : 5> } {keep_pkt_header = true}
    aie.packet_flow(19) { aie.packet_source<%t_5_4, DMA : 1> aie.packet_dest<%t_5_1, DMA : 2> }
    aie.packet_flow(28) { aie.packet_source<%t_5_4, DMA : 1> aie.packet_dest<%t_4_1, DMA : 4> aie.packet_dest<%t_5_1, DMA : 2> aie.packet_dest<%t_6_1, DMA : 2> }
  }
}

