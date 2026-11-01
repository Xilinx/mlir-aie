//===- cut_counts_packets_kept_apart.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s
// RUN: not aie-opt --aie-create-pathfinder-flows="allow-deadlock-prone=true" %s 2>&1 >/dev/null | FileCheck %s

// Five circuit trees and two packet flows that cannot share an arbiter cross
// from row 1 to row 2. A channel's master port has one arbiter, so the packet
// flows need a channel each: seven, where there are six. Shaped after
// router_mutation.py npu2 seed 207370059 (port-cap), whose 47 circuit trees
// and conflicting packet flows took 49 of 48 channels and ran the relax
// ladder for about 110 s before failing.

// CHECK: error: Unable to find a legal routing: the circuit flows from (0, 1) DMA:0, (0, 1) DMA:1, (0, 1) DMA:2, (0, 1) DMA:3 and 1 more, and packet flow (0, 0) DMA:0 -> (0, 4) DMA:1 (id 14) and packet flow (0, 0) DMA:1 -> (0, 3) Core:0 (id 30), no two of which can share an arbiter, cross from row 1 to row 2, which takes 7 channels across, and it has 6 channels. For example,

module {
  aie.device(npu1_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    aie.flow(%t_0_1, DMA : 0, %t_0_3, DMA : 0)
    aie.flow(%t_0_1, DMA : 1, %t_0_3, DMA : 1)
    aie.flow(%t_0_1, DMA : 2, %t_0_4, Core : 0)
    aie.flow(%t_0_1, DMA : 3, %t_0_5, DMA : 0)
    aie.flow(%t_0_1, DMA : 4, %t_0_5, DMA : 1)
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
