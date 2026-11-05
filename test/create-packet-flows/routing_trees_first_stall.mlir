//===- routing_trees_first_stall.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: asserts
// RUN: not aie-opt --aie-create-pathfinder-flows -debug-only=aie-pathfinder %s 2>&1 >/dev/null | FileCheck %s

// Once the overuse stalls, the circuit trees on its channels are routed first;
// an attempt gives up when doing so has moved maxReorderedStall trees without
// fewer overused channels, rather than run to the iteration limit. The design
// is then rejected for its packet flows' arbiter hold cycle, as the unreduced
// one is. Reduced from the router nightly's npu2 seed 207610162, port-cap
// mutant.
// CHECK-NOT: giving up after {{[0-9]+}} iterations (the iteration limit)
// CHECK: giving up after {{[0-9]+}} iterations (circuit trees routed first stopped helping)
// CHECK-NOT: giving up after {{[0-9]+}} iterations (the iteration limit)
// CHECK: error: Unable to find a legal routing: packet flows can deadlock holding arbiters

module {
  aie.device(npu2_4col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    %t_1_0 = aie.tile(1, 0)
    %t_1_1 = aie.tile(1, 1)
    %t_1_2 = aie.tile(1, 2)
    %t_1_3 = aie.tile(1, 3)
    %t_1_4 = aie.tile(1, 4)
    %t_1_5 = aie.tile(1, 5)
    %t_2_0 = aie.tile(2, 0)
    %t_2_1 = aie.tile(2, 1)
    %t_2_2 = aie.tile(2, 2)
    %t_2_3 = aie.tile(2, 3)
    %t_2_4 = aie.tile(2, 4)
    %t_2_5 = aie.tile(2, 5)
    %t_3_0 = aie.tile(3, 0)
    %t_3_1 = aie.tile(3, 1)
    %t_3_2 = aie.tile(3, 2)
    %t_3_3 = aie.tile(3, 3)
    %t_3_4 = aie.tile(3, 4)
    %t_3_5 = aie.tile(3, 5)
    %b_1_1_4 = aie.buffer(%t_1_1) {sym_name = "b_1_1_4"} : memref<16xi32>
    %dma_1_1 = aie.memtile_dma(%t_1_1) {
      %d0 = aie.dma_start(S2MM, 1, ^p0b0, ^p1)
    ^p0b0:
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(S2MM, 2, ^p1b0, ^p2)
    ^p1b0:
      aie.next_bd ^p1b0
    ^p2:
      %d2 = aie.dma_start(S2MM, 3, ^p2b0, ^p3)
    ^p2b0:
      aie.next_bd ^p2b0
    ^p3:
      %d3 = aie.dma_start(S2MM, 4, ^p3b0, ^p4)
    ^p3b0:
      aie.next_bd ^p3b0
    ^p4:
      %d4 = aie.dma_start(MM2S, 4, ^p4b0, ^end)
    ^p4b0:
      aie.dma_bd(%b_1_1_4 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^p4b0
    ^end:
      aie.end
    }
    %dma_1_2 = aie.mem(%t_1_2) {
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^end)
    ^p0b0:
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %dma_1_5 = aie.mem(%t_1_5) {
      %d0 = aie.dma_start(S2MM, 1, ^p0b0, ^end)
    ^p0b0:
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    aie.flow(%t_0_2, Core : 0, %t_3_4, Core : 0)
    aie.flow(%t_2_1, DMA : 0, %t_0_4, DMA : 0)
    aie.flow(%t_3_1, DMA : 0, %t_3_3, DMA : 0)
    aie.flow(%t_2_2, Core : 0, %t_2_3, Core : 0)
    aie.flow(%t_0_0, DMA : 1, %t_2_4, DMA : 1)
    aie.flow(%t_3_1, DMA : 2, %t_2_3, DMA : 0)
    aie.flow(%t_1_1, DMA : 5, %t_2_4, Core : 0)
    aie.flow(%t_0_1, DMA : 0, %t_2_3, DMA : 1)
    aie.flow(%t_1_2, DMA : 1, %t_0_5, Core : 0)
    aie.flow(%t_3_0, DMA : 0, %t_1_4, Core : 0)
    aie.flow(%t_0_1, DMA : 5, %t_0_5, DMA : 1)
    aie.flow(%t_3_2, Core : 0, %t_3_4, DMA : 0)
    aie.flow(%t_3_2, DMA : 0, %t_3_5, Core : 0)
    aie.flow(%t_3_0, DMA : 1, %t_3_3, Core : 0)
    aie.flow(%t_2_1, DMA : 2, %t_3_5, DMA : 1)
    aie.flow(%t_1_2, DMA : 0, %t_1_4, DMA : 1)
    aie.flow(%t_2_2, DMA : 1, %t_2_5, Core : 0)
    aie.flow(%t_3_1, DMA : 1, %t_1_3, Core : 0)
    aie.flow(%t_2_0, DMA : 1, %t_0_4, DMA : 1)
    aie.flow(%t_2_1, DMA : 3, %t_3_3, DMA : 1)
    aie.flow(%t_1_1, DMA : 3, %t_0_3, DMA : 0)
    aie.flow(%t_1_1, DMA : 2, %t_1_5, DMA : 0)
    aie.flow(%t_0_1, DMA : 4, %t_2_5, DMA : 1)
    aie.packet_flow(15) { aie.packet_source<%t_1_1, DMA : 4> aie.packet_dest<%t_1_0, DMA : 0> aie.packet_dest<%t_1_5, Core : 0> aie.packet_dest<%t_1_5, DMA : 1> } {keep_pkt_header = false}
    aie.packet_flow(9) { aie.packet_source<%t_1_0, DMA : 1> aie.packet_dest<%t_1_1, DMA : 4> aie.packet_dest<%t_1_2, DMA : 0> aie.packet_dest<%t_1_5, DMA : 1> }
    aie.runtime_sequence @seq0(%a0: memref<64xi32>, %a1: memref<32xi32>) {
      %task0 = aiex.dma_configure_task(%t_1_0, MM2S, 1) {
        aie.dma_bd(%a0 : memref<64xi32> offset = 0 len = 64) {bd_id = 0 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 9>}
        aie.end
      } {issue_token = true}
      %lb0 = arith.constant 0 : index
      %ub0 = arith.constant 2 : index
      %st0 = arith.constant 1 : index
      scf.for %i0 = %lb0 to %ub0 step %st0 {
      }
      aiex.dma_start_task(%task0)
    }
  }
}
