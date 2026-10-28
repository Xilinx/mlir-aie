//===- packet_stall_retracts_id_splits.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows="allow-deadlock-prone=true" %s 2>&1 >/dev/null | FileCheck %s

// Id splits a hold cycle in an early routing asked of (0,3) DMA:0's tree take
// a channel down from (0,3) it does not have. A stall drops them, and the
// design routes.
// Reduced from router_mutation.py npu1 seed 207370212 (pin-config).

// CHECK-NOT: error
// CHECK: warning: Packet flows into receivers they share can deadlock
// CHECK-NOT: error

module {
  aie.device(npu1_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    %b_0_3_0 = aie.buffer(%t_0_3) {sym_name = "b_0_3_0"} : memref<32xi32>
    %b_0_3_1 = aie.buffer(%t_0_3) {sym_name = "b_0_3_1"} : memref<16xi32>
    %dma_0_3 = aie.mem(%t_0_3) {
      %d0 = aie.dma_start(MM2S, 0, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_0_3_0 : memref<32xi32> offset = 0 len = 32) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
      aie.next_bd ^p0b1
    ^p0b1:
      aie.dma_bd(%b_0_3_1 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %sb_0_2 = aie.switchbox(%t_0_2) {
      aie.connect<DMA : 0, North : 4>
    }
    %sb_0_3 = aie.switchbox(%t_0_3) {
      aie.connect<South : 4, North : 0>
    }
    %sb_0_4 = aie.switchbox(%t_0_4) {
      aie.connect<South : 0, North : 3>
      aie.connect<South : 2, North : 4>
    }
    %sb_0_5 = aie.switchbox(%t_0_5) {
      aie.connect<South : 3, DMA : 1>
    }
    aie.flow(%t_0_3, Core : 0, %t_0_1, DMA : 5)
    aie.flow(%t_0_3, Core : 0, %t_0_1, DMA : 1)
    aie.packet_flow(9) { aie.packet_source<%t_0_3, DMA : 0> aie.packet_source<%t_0_4, Core : 0> aie.packet_dest<%t_0_5, Core : 0> }
    aie.packet_flow(24) { aie.packet_source<%t_0_3, DMA : 0> aie.packet_dest<%t_0_0, DMA : 0> aie.packet_dest<%t_0_2, DMA : 0> }
    aie.packet_flow(5) { aie.packet_source<%t_0_3, DMA : 0> aie.packet_source<%t_0_2, Core : 0> aie.packet_dest<%t_0_1, DMA : 0> aie.packet_dest<%t_0_2, DMA : 0> aie.packet_dest<%t_0_5, Core : 0> }
    aie.packet_flow(14) { aie.packet_source<%t_0_1, DMA : 2> aie.packet_dest<%t_0_4, DMA : 0> }
    aie.packet_flow(28) { aie.packet_source<%t_0_1, DMA : 2> aie.packet_source<%t_0_1, DMA : 5> aie.packet_dest<%t_0_0, DMA : 0> }
    aie.packet_flow(11) { aie.packet_source<%t_0_4, Core : 0> aie.packet_dest<%t_0_2, DMA : 0> }
    aie.packet_flow(20) { aie.packet_source<%t_0_4, Core : 0> aie.packet_dest<%t_0_2, Core : 0> aie.packet_dest<%t_0_3, DMA : 0> aie.packet_dest<%t_0_3, DMA : 1> }
    aie.packet_flow(8) { aie.packet_source<%t_0_3, DMA : 1> aie.packet_source<%t_0_1, DMA : 0> aie.packet_dest<%t_0_1, DMA : 0> aie.packet_dest<%t_0_3, Core : 0> }
  }
}
