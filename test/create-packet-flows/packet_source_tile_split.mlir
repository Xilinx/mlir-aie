//===- packet_source_tile_split.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s

// Ids 21 and 8 leave core (0,5) on DMA:0 and DMA:1, and can deadlock if they
// share an arbiter. Both head south, and packet flows share a channel for
// free, so a penalty on the channel they share moves them to the next one
// together. The router must put them on different channels.

// Nothing programs the DMAs, so the design itself may deadlock.
// NOWARN: warning: Flows can deadlock however they are routed
// NOWARN-NOT: {{warning|error}}

// CHECK-LABEL: aie.switchbox(%tile_0_5)
// CHECK-DAG:     %[[A:.*]] = aie.amsel<0> (0)
// CHECK-DAG:     %[[B:.*]] = aie.amsel<1> (0)
// CHECK-DAG:     aie.masterset(South : {{[0-9]+}}, %[[A]])
// CHECK-DAG:     aie.masterset(South : {{[0-9]+}}, %[[B]])
// CHECK:         aie.packet_rules(DMA : 1) {
// CHECK-NEXT:      aie.rule(31, 8, %[[B]])
// CHECK:         aie.packet_rules(DMA : 0) {
// CHECK-NEXT:      aie.rule(31, 21, %[[A]])

module {
  aie.device(npu2_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    %l_0_1_0 = aie.lock(%t_0_1, 0) {init = 2 : i32, sym_name = "l_0_1_0"}
    %l_0_1_1 = aie.lock(%t_0_1, 1) {init = 0 : i32, sym_name = "l_0_1_1"}
    %l_0_4_0 = aie.lock(%t_0_4, 0) {init = 2 : i32, sym_name = "l_0_4_0"}
    %l_0_4_1 = aie.lock(%t_0_4, 1) {init = 0 : i32, sym_name = "l_0_4_1"}
    %l_0_5_0 = aie.lock(%t_0_5, 0) {init = 0 : i32, sym_name = "l_0_5_0"}
    %l_0_5_1 = aie.lock(%t_0_5, 1) {init = 1 : i32, sym_name = "l_0_5_1"}
    %b_0_1_0 = aie.buffer(%t_0_1) {sym_name = "b_0_1_0"} : memref<16xi32>
    %b_0_1_1 = aie.buffer(%t_0_1) {sym_name = "b_0_1_1"} : memref<64xi32>
    %b_0_1_2 = aie.buffer(%t_0_1) {sym_name = "b_0_1_2"} : memref<8xi32>
    %b_0_1_3 = aie.buffer(%t_0_1) {sym_name = "b_0_1_3"} : memref<8xi32>
    %dma_0_1 = aie.memtile_dma(%t_0_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^p0b0, ^p1)
    ^p0b0:
      aie.dma_bd(%b_0_1_0 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(MM2S, 1, ^p1b0, ^p2)
    ^p1b0:
      aie.dma_bd(%b_0_1_1 : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^p1b1
    ^p1b1:
      aie.dma_bd(%b_0_1_2 : memref<8xi32> offset = 0 len = 8)
      aie.next_bd ^end
    ^p2:
      %d2 = aie.dma_start(S2MM, 3, ^p2b0, ^end)
    ^p2b0:
      aie.use_lock(%l_0_1_0, AcquireGreaterEqual, %one)
      aie.dma_bd(%b_0_1_3 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%l_0_1_1, Release, %one)
      aie.next_bd ^p2b0
    ^end:
      aie.end
    }
    %b_0_2_0 = aie.buffer(%t_0_2) {sym_name = "b_0_2_0"} : memref<8xi32>
    %dma_0_2 = aie.mem(%t_0_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 1, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_0_2_0 : memref<8xi32> offset = 0 len = 8)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %b_0_3_0 = aie.buffer(%t_0_3) {sym_name = "b_0_3_0"} : memref<32xi32>
    %dma_0_3 = aie.mem(%t_0_3) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 1, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_0_3_0 : memref<32xi32> offset = 0 len = 32)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %b_0_4_0 = aie.buffer(%t_0_4) {sym_name = "b_0_4_0"} : memref<16xi32>
    %b_0_4_1 = aie.buffer(%t_0_4) {sym_name = "b_0_4_1"} : memref<8xi32>
    %dma_0_4 = aie.mem(%t_0_4) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^p1)
    ^p0b0:
      aie.use_lock(%l_0_4_0, AcquireGreaterEqual, %one)
      aie.dma_bd(%b_0_4_0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%l_0_4_1, Release, %one)
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(S2MM, 1, ^p1b0, ^end)
    ^p1b0:
      aie.dma_bd(%b_0_4_1 : memref<8xi32> offset = 0 len = 8)
      aie.next_bd ^p1b0
    ^end:
      aie.end
    }
    %b_0_5_0 = aie.buffer(%t_0_5) {sym_name = "b_0_5_0"} : memref<16xi32>
    %b_0_5_1 = aie.buffer(%t_0_5) {sym_name = "b_0_5_1"} : memref<8xi32>
    %dma_0_5 = aie.mem(%t_0_5) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^p0b0, ^p1)
    ^p0b0:
      aie.use_lock(%l_0_5_0, AcquireGreaterEqual, %one)
      aie.dma_bd(%b_0_5_0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%l_0_5_1, Release, %one)
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(MM2S, 1, ^p1b0, ^end)
    ^p1b0:
      aie.dma_bd(%b_0_5_1 : memref<8xi32> offset = 0 len = 8) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 8>}
      aie.next_bd ^p1b0
    ^end:
      aie.end
    }
    %core_0_4 = aie.core(%t_0_4) {
      %one = arith.constant 1 : i32
      aie.use_lock(%l_0_4_1, AcquireGreaterEqual, %one)
      aie.use_lock(%l_0_4_0, Release, %one)
      aie.end
    }
    %core_0_5 = aie.core(%t_0_5) {
      %one = arith.constant 1 : i32
      aie.use_lock(%l_0_5_0, Release, %one)
      aie.use_lock(%l_0_5_1, AcquireGreaterEqual, %one)
      aie.end
    }
    aie.packet_flow(14) { aie.packet_source<%t_0_2, DMA : 1> aie.packet_dest<%t_0_1, DMA : 3> aie.packet_dest<%t_0_5, DMA : 1> } {priority_route = true}
    aie.packet_flow(21) { aie.packet_source<%t_0_5, DMA : 0> aie.packet_dest<%t_0_2, DMA : 1> }
    aie.packet_flow(8) { aie.packet_source<%t_0_5, DMA : 1> aie.packet_dest<%t_0_1, DMA : 3> } {keep_pkt_header = true}
  }
}

