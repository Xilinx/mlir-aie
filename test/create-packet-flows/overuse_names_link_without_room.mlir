//===- overuse_names_link_without_room.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s

// Flows 14 and 30 cannot share an arbiter, but fixed mastersets leave (0, 1)
// one North channel and one arbiter. The link from (0, 0) to (0, 1) is
// overused too, yet has free channels, so the error names the link from
// (0, 1) to (0, 2). Reduced from router_properties.py seed 1012.

// CHECK: error: Unable to find a legal routing: the flows from (0, 0) DMA:0 and (0, 0) DMA:1 need the links from tile (0, 1) to (0, 2), and the router found no routing that fits them.

module {
  aie.device(npu1_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    %sb_0_1 = aie.switchbox(%t_0_1) {
      %a1_0_0 = aie.amsel<1> (0)
      %a1_1_1 = aie.amsel<1> (1)
      %a1_2_2 = aie.amsel<1> (2)
      %a1_3_3 = aie.amsel<1> (3)
      %m_North_1 = aie.masterset(North : 1, %a1_0_0, %a1_1_1, %a1_2_2, %a1_3_3)
      %a2_0_4 = aie.amsel<2> (0)
      %a2_1_5 = aie.amsel<2> (1)
      %a2_2_6 = aie.amsel<2> (2)
      %a2_3_7 = aie.amsel<2> (3)
      %m_North_2 = aie.masterset(North : 2, %a2_0_4, %a2_1_5, %a2_2_6, %a2_3_7)
      %a3_0_8 = aie.amsel<3> (0)
      %a3_1_9 = aie.amsel<3> (1)
      %a3_2_10 = aie.amsel<3> (2)
      %a3_3_11 = aie.amsel<3> (3)
      %m_North_0 = aie.masterset(North : 0, %a3_0_8, %a3_1_9, %a3_2_10, %a3_3_11)
      %a4_0_12 = aie.amsel<4> (0)
      %a4_1_13 = aie.amsel<4> (1)
      %a4_2_14 = aie.amsel<4> (2)
      %a4_3_15 = aie.amsel<4> (3)
      %m_North_5 = aie.masterset(North : 5, %a4_0_12, %a4_1_13, %a4_2_14, %a4_3_15)
      %a5_0_16 = aie.amsel<5> (0)
      %a5_1_17 = aie.amsel<5> (1)
      %a5_2_18 = aie.amsel<5> (2)
      %a5_3_19 = aie.amsel<5> (3)
      %m_North_4 = aie.masterset(North : 4, %a5_0_16, %a5_1_17, %a5_2_18, %a5_3_19)
    }
    aie.packet_flow(15) { aie.packet_source<%t_0_3, DMA : 0> aie.packet_dest<%t_0_4, DMA : 0> }
    aie.packet_flow(14) { aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_4, DMA : 1> }
    aie.packet_flow(30) { aie.packet_source<%t_0_0, DMA : 1> aie.packet_dest<%t_0_3, Core : 0> }
    aie.shim_dma_allocation @in0_0(%t_0_0, MM2S, 0, <pkt_id = 14, pkt_type = 0>)
    aie.runtime_sequence @seq0(%a0: memref<32xi32>, %a1: memref<16xi32>) {
      %task0 = aiex.dma_configure_task(%t_0_0, MM2S, 1) {
        aie.dma_bd(%a0 : memref<32xi32> offset = 0 len = 32) {bd_id = 0 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
        aie.end
      } {issue_token = true}
      aiex.npu.dma_memcpy_nd(%a1[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1], packet = <pkt_id = 14, pkt_type = 0>) { metadata = @in0_0, id = 1 : i64, issue_token = true } : memref<16xi32>
      %lb0 = arith.constant 0 : index
      %ub0 = arith.constant 2 : index
      %st0 = arith.constant 1 : index
      scf.for %i0 = %lb0 to %ub0 step %st0 {
        aiex.dma_start_task(%task0)
      }
      aiex.dma_await_task(%task0)
      aiex.npu.dma_wait {symbol = @in0_0}
    }
  }
}
