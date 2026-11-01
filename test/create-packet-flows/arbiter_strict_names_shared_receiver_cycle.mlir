//===- arbiter_strict_names_shared_receiver_cycle.mlir ---------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 | FileCheck %s --check-prefix=STRICT
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false allow-deadlock-prone=true" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN

// The first routing puts id 31 and id 7 on one arbiter at (2,2), where they
// can deadlock, and no arbiter assignment avoids it. Sharing channels then
// finds a routing whose only fault is the hold cycle the two flows of id 7
// can make through (2,3) DMA:1, which they share. That is all that keeps the
// design from routing, so without allow-deadlock-prone the error names it and
// how to route anyway. Reduced from router_properties.py npu1 seed 14822.

// STRICT:      error: Unable to find a legal routing: Packet flows into receivers they share can deadlock holding arbiters across switchboxes, and no routing found avoids it:
// STRICT-SAME: Set allow-deadlock-prone

// WARN: warning: Packet flows into receivers they share can deadlock holding arbiters across switchboxes, and no routing found avoids it:

module {
  aie.device(npu1_3col) {
    %t_2_0 = aie.tile(2, 0)
    %t_2_1 = aie.tile(2, 1)
    %t_2_2 = aie.tile(2, 2)
    %t_2_3 = aie.tile(2, 3)
    %t_2_4 = aie.tile(2, 4)
    %t_2_5 = aie.tile(2, 5)
    %l_2_1_0 = aie.lock(%t_2_1, 0) {init = 1 : i32, sym_name = "l_2_1_0"}
    %l_2_1_1 = aie.lock(%t_2_1, 1) {init = 0 : i32, sym_name = "l_2_1_1"}
    %b_2_1_0 = aie.buffer(%t_2_1) {sym_name = "b_2_1_0"} : memref<32xi32>
    %dma_2_1 = aie.memtile_dma(%t_2_1) {
      %c1 = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 5, ^p0b0, ^end)
    ^p0b0:
      aie.use_lock(%l_2_1_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_2_1_0 : memref<32xi32> offset = 0 len = 32)
      aie.use_lock(%l_2_1_1, Release, %c1)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %b_2_2_1 = aie.buffer(%t_2_2) {sym_name = "b_2_2_1"} : memref<32xi32>
    %dma_2_2 = aie.mem(%t_2_2) {
      %d0 = aie.dma_start(S2MM, 1, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_2_2_1 : memref<32xi32> offset = 0 len = 32)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %b_2_3_2 = aie.buffer(%t_2_3) {sym_name = "b_2_3_2"} : memref<16xi32>
    %b_2_3_3 = aie.buffer(%t_2_3) {sym_name = "b_2_3_3"} : memref<8xi32>
    %dma_2_3 = aie.mem(%t_2_3) {
      %d0 = aie.dma_start(MM2S, 1, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_2_3_2 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 10>}
      aie.next_bd ^p0b1
    ^p0b1:
      aie.dma_bd(%b_2_3_3 : memref<8xi32> offset = 0 len = 8) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 10>}
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    aie.packet_flow(29) { aie.packet_source<%t_2_4, DMA : 1> aie.packet_dest<%t_2_2, DMA : 1> aie.packet_dest<%t_2_4, DMA : 1> }
    aie.packet_flow(10) { aie.packet_source<%t_2_2, Core : 0> aie.packet_dest<%t_2_1, DMA : 5> }
    aie.packet_flow(27) { aie.packet_source<%t_2_3, DMA : 1> aie.packet_source<%t_2_5, Core : 0> aie.packet_dest<%t_2_1, DMA : 2> }
    aie.packet_flow(15) { aie.packet_source<%t_2_1, DMA : 4> aie.packet_dest<%t_2_3, DMA : 0> }
    aie.packet_flow(7) { aie.packet_source<%t_2_4, DMA : 0> aie.packet_source<%t_2_2, DMA : 1> aie.packet_dest<%t_2_3, DMA : 1> aie.packet_dest<%t_2_5, DMA : 0> }
    aie.packet_flow(10) { aie.packet_source<%t_2_1, DMA : 0> aie.packet_dest<%t_2_3, Core : 0> }
    aie.packet_flow(31) { aie.packet_source<%t_2_1, DMA : 2> aie.packet_dest<%t_2_3, DMA : 1> }
  }
}
