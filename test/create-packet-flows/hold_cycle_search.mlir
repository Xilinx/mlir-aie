//===- hold_cycle_search.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Found by the model-based router harness. On a candidate routing, the
// shortest walk closing a hold cycle needs two trees holding one arbiter at
// once, which no state has, so the hold-cycle search rules that holder out and
// walks again until no walk is left. Few designs reach that part of the search.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[S01_0:.*]] = aie.amsel<5> (1)
// CHECK:         %[[S01_1:.*]] = aie.amsel<5> (2)
// CHECK:         %[[S01_2:.*]] = aie.amsel<3> (3)
// CHECK:         %[[S01_3:.*]] = aie.amsel<4> (3)
// CHECK:         %[[S01_4:.*]] = aie.amsel<5> (3)
// CHECK:         %[[S01_5:.*]] = aie.masterset(DMA : 1, %[[S01_1]])
// CHECK:         %[[S01_6:.*]] = aie.masterset(North : 0, %[[S01_2]]) {is_ctrl_pkt_overlay}
// CHECK:         %[[S01_7:.*]] = aie.masterset(North : 1, %[[S01_1]], %[[S01_4]]) {is_ctrl_pkt_overlay}
// CHECK:         %[[S01_8:.*]] = aie.masterset(North : 2, %[[S01_0]]) {is_ctrl_pkt_overlay}
// CHECK:         %[[S01_9:.*]] = aie.masterset(North : 4, %[[S01_3]]) {is_ctrl_pkt_overlay}
// CHECK:         aie.packet_rules(DMA : 1) {
// CHECK:         aie.rule(31, 26, %[[S01_3]]) {priority_route}
// CHECK:         } {is_ctrl_pkt_overlay}
// CHECK:         aie.packet_rules(South : 1) {
// CHECK:         aie.rule(31, 20, %[[S01_1]])
// CHECK:         aie.rule(31, 0, %[[S01_4]])
// CHECK:         aie.rule(31, 29, %[[S01_4]])
// CHECK:         }
// CHECK:         aie.packet_rules(South : 0) {
// CHECK:         aie.rule(31, 0, %[[S01_2]])
// CHECK:         } {is_ctrl_pkt_overlay}
// CHECK:         aie.packet_rules(South : 2) {
// CHECK:         aie.rule(31, 0, %[[S01_0]])
// CHECK:         } {is_ctrl_pkt_overlay}
// CHECK-LABEL: aie.switchbox(%tile_0_2)
// CHECK:         %[[S02_0:.*]] = aie.amsel<0> (0)
// CHECK:         %[[S02_1:.*]] = aie.amsel<5> (2)
// CHECK:         %[[S02_2:.*]] = aie.amsel<3> (3)
// CHECK:         %[[S02_3:.*]] = aie.amsel<4> (3)
// CHECK:         %[[S02_4:.*]] = aie.amsel<5> (3)
// CHECK:         %[[S02_5:.*]] = aie.masterset(Core : 0, %[[S02_0]])
// CHECK:         %[[S02_6:.*]] = aie.masterset(DMA : 0, %[[S02_1]], %[[S02_4]]) {is_ctrl_pkt_overlay}
// CHECK:         %[[S02_7:.*]] = aie.masterset(North : 0, %[[S02_4]]) {is_ctrl_pkt_overlay}
// CHECK:         %[[S02_8:.*]] = aie.masterset(North : 2, %[[S02_2]]) {is_ctrl_pkt_overlay}
// CHECK:         %[[S02_9:.*]] = aie.masterset(North : 3, %[[S02_3]]) {is_ctrl_pkt_overlay}
// CHECK:         aie.packet_rules(South : 4) {
// CHECK:         aie.rule(31, 26, %[[S02_2]])
// CHECK:         } {is_ctrl_pkt_overlay}
// CHECK:         aie.packet_rules(South : 1) {
// CHECK:         aie.rule(11, 0, %[[S02_3]])
// CHECK:         aie.rule(31, 29, %[[S02_0]])
// CHECK:         }
// CHECK:         aie.packet_rules(South : 0) {
// CHECK:         aie.rule(31, 0, %[[S02_4]])
// CHECK:         } {is_ctrl_pkt_overlay}
// CHECK:         aie.packet_rules(South : 2) {
// CHECK:         aie.rule(31, 0, %[[S02_1]])
// CHECK:         } {is_ctrl_pkt_overlay}

module {
  aie.device(npu1_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    %l_0_4_0 = aie.lock(%t_0_4, 0) {init = 1 : i32, sym_name = "l_0_4_0"}
    %l_0_4_1 = aie.lock(%t_0_4, 1) {init = 0 : i32, sym_name = "l_0_4_1"}
    %l_0_5_0 = aie.lock(%t_0_5, 0) {init = 1 : i32, sym_name = "l_0_5_0"}
    %l_0_5_1 = aie.lock(%t_0_5, 1) {init = 0 : i32, sym_name = "l_0_5_1"}
    %b_0_1_0 = aie.buffer(%t_0_1) {sym_name = "b_0_1_0"} : memref<16xi32>
    %b_0_1_1 = aie.buffer(%t_0_1) {sym_name = "b_0_1_1"} : memref<8xi32>
    %b_0_1_2 = aie.buffer(%t_0_1) {sym_name = "b_0_1_2"} : memref<32xi32>
    %b_0_1_3 = aie.buffer(%t_0_1) {sym_name = "b_0_1_3"} : memref<16xi32>
    %b_0_1_4 = aie.buffer(%t_0_1) {sym_name = "b_0_1_4"} : memref<32xi32>
    %b_0_1_5 = aie.buffer(%t_0_1) {sym_name = "b_0_1_5"} : memref<32xi32>
    %dma_0_1 = aie.memtile_dma(%t_0_1) {
      %d0 = aie.dma_start(S2MM, 4, ^p0b0, ^p1)
    ^p0b0:
      aie.dma_bd(%b_0_1_0 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(MM2S, 2, ^p1b0, ^p2)
    ^p1b0:
      aie.dma_bd(%b_0_1_1 : memref<8xi32> offset = 0 len = 8)
      aie.next_bd ^end
    ^p2:
      %d2 = aie.dma_start(MM2S, 3, ^p2b0, ^p3)
    ^p2b0:
      aie.dma_bd(%b_0_1_2 : memref<32xi32> offset = 0 len = 32)
      aie.next_bd ^p2b1
    ^p2b1:
      aie.dma_bd(%b_0_1_3 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^p3:
      %d3 = aie.dma_start(MM2S, 5, ^p3b0, ^end)
    ^p3b0:
      aie.dma_bd(%b_0_1_4 : memref<32xi32> offset = 0 len = 32)
      aie.next_bd ^p3b1
    ^p3b1:
      aie.dma_bd(%b_0_1_5 : memref<32xi32> offset = 0 len = 32)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %b_0_4_6 = aie.buffer(%t_0_4) {sym_name = "b_0_4_6"} : memref<8xi32>
    %dma_0_4 = aie.mem(%t_0_4) {
      %c1 = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 1, ^p0b0, ^end)
    ^p0b0:
      aie.use_lock(%l_0_4_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_0_4_6 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%l_0_4_1, Release, %c1)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %b_0_5_7 = aie.buffer(%t_0_5) {sym_name = "b_0_5_7"} : memref<32xi32>
    %dma_0_5 = aie.mem(%t_0_5) {
      %c1 = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 1, ^p0b0, ^end)
    ^p0b0:
      aie.use_lock(%l_0_5_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_0_5_7 : memref<32xi32> offset = 0 len = 32)
      aie.use_lock(%l_0_5_1, Release, %c1)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %core_0_4 = aie.core(%t_0_4) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%l_0_4_1, AcquireGreaterEqual, %c1)
      aie.use_lock(%l_0_4_0, Release, %c1)
      aie.end
    }
    %core_0_5 = aie.core(%t_0_5) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%l_0_5_1, AcquireGreaterEqual, %c1)
      aie.use_lock(%l_0_5_0, Release, %c1)
      aie.end
    }
    aie.packet_flow(0) { aie.packet_source<%t_0_0, DMA : 1> aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_2, DMA : 0> aie.packet_dest<%t_0_5, DMA : 1> } {priority_route = true}
    aie.packet_flow(29) { aie.packet_source<%t_0_0, DMA : 1> aie.packet_dest<%t_0_2, Core : 0> }
    aie.packet_flow(20) { aie.packet_source<%t_0_0, DMA : 1> aie.packet_dest<%t_0_1, DMA : 1> aie.packet_dest<%t_0_3, DMA : 1> }
    aie.packet_flow(26) { aie.packet_source<%t_0_1, DMA : 1> aie.packet_dest<%t_0_3, DMA : 1> } {priority_route = true}
    aie.shim_dma_allocation @in0_0(%t_0_0, MM2S, 0, <pkt_id = 0, pkt_type = 0>)
    aie.shim_dma_allocation @in0_1(%t_0_0, MM2S, 1, <pkt_id = 29, pkt_type = 0>)
    aie.shim_dma_allocation @out0_0(%t_0_0, S2MM, 0)
    aie.runtime_sequence @seq0(%a0: memref<16xi32>, %a1: memref<64xi32>, %a2: memref<16xi32>, %a3: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%a0[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) { metadata = @out0_0, id = 0 : i64, issue_token = true } : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%a1[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1], packet = <pkt_id = 0, pkt_type = 0>) { metadata = @in0_0, id = 1 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_wait {symbol = @out0_0}
      aiex.npu.dma_memcpy_nd(%a2[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1], packet = <pkt_id = 20, pkt_type = 0>) { metadata = @in0_1, id = 2 : i64, issue_token = true } : memref<16xi32>
      aiex.npu.dma_wait {symbol = @in0_0}
      aiex.npu.dma_memcpy_nd(%a3[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1], packet = <pkt_id = 29, pkt_type = 0>) { metadata = @in0_1, id = 3 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_wait {symbol = @in0_1}
    }
  }
}
