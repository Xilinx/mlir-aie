//===- packet_meeting_reaches_loopback.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOERR --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOERR --allow-empty

// NOERR-NOT: {{warning|error}}

// Ids 28 and 18 reach the same three receivers, one of them (0, 1) DMA:2,
// which id 18 is sent from and comes straight back into through the (0, 1)
// switchbox. The tree of id 28 meets the tree of id 18 at (0, 1), which
// takes it into all three, the port id 18 sends itself on included.
// Reduced from router_properties.py npu1 seed 207461204.

module {
  aie.device(npu1_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    %b_0_1_0 = aie.buffer(%t_0_1) {sym_name = "b_0_1_0"} : memref<32xi32>
    %b_0_1_1 = aie.buffer(%t_0_1) {sym_name = "b_0_1_1"} : memref<32xi32>
    %b_0_1_2 = aie.buffer(%t_0_1) {sym_name = "b_0_1_2"} : memref<8xi32>
    %dma_0_1 = aie.memtile_dma(%t_0_1) {
      %d0 = aie.dma_start(S2MM, 2, ^p0b0, ^p1)
    ^p0b0:
      aie.dma_bd(%b_0_1_0 : memref<32xi32> offset = 0 len = 32)
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(MM2S, 2, ^p1b0, ^end, repeat_count = 1)
    ^p1b0:
      aie.dma_bd(%b_0_1_1 : memref<32xi32> offset = 0 len = 32) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 18>}
      aie.next_bd ^p1b1
    ^p1b1:
      aie.dma_bd(%b_0_1_2 : memref<8xi32> offset = 0 len = 8)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %b_0_3_3 = aie.buffer(%t_0_3) {sym_name = "b_0_3_3"} : memref<8xi32>
    %dma_0_3 = aie.mem(%t_0_3) {
      %d0 = aie.dma_start(S2MM, 1, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_0_3_3 : memref<8xi32> offset = 0 len = 8)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %b_0_4_4 = aie.buffer(%t_0_4) {sym_name = "b_0_4_4"} : memref<8xi32>
    %b_0_4_5 = aie.buffer(%t_0_4) {sym_name = "b_0_4_5"} : memref<8xi32>
    %dma_0_4 = aie.mem(%t_0_4) {
      %d0 = aie.dma_start(MM2S, 1, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_0_4_4 : memref<8xi32> offset = 0 len = 8) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 28>}
      aie.next_bd ^p0b1
    ^p0b1:
      aie.dma_bd(%b_0_4_5 : memref<8xi32> offset = 0 len = 8)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    aie.flow(%t_0_5, DMA : 1, %t_0_0, DMA : 0)
    aie.packet_flow(28) { aie.packet_source<%t_0_4, DMA : 1> aie.packet_dest<%t_0_1, DMA : 2> aie.packet_dest<%t_0_3, DMA : 1> aie.packet_dest<%t_0_5, DMA : 0> } {keep_pkt_header = false, priority_route = false}
    aie.packet_flow(18) { aie.packet_source<%t_0_1, DMA : 2> aie.packet_dest<%t_0_1, DMA : 2> aie.packet_dest<%t_0_3, DMA : 1> aie.packet_dest<%t_0_5, DMA : 0> }
    aie.packet_flow(15) { aie.packet_source<%t_0_0, TileControl : 0> aie.packet_dest<%t_0_0, South : 0> } {keep_pkt_header = true, priority_route = true}
    aie.shim_dma_allocation @out0_0(%t_0_0, S2MM, 0)
    aie.runtime_sequence @seq0(%a0: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%a0[0, 0, 0, 0][1, 1, 1, 64][64, 0, 0, 1]) { metadata = @out0_0, id = 0 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_wait {symbol = @out0_0}
    }
  }
}
