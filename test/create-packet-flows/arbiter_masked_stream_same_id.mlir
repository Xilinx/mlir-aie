//===- arbiter_masked_stream_same_id.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 | FileCheck %s

// Two flows from core (0,5) DMA:0 to core (0,3) DMA:0 state id 2, one
// unmasked and one with a mask that also claims id 3. They carry different
// ids, so each is a stream of its own. Core (0,5) sends only id 3, so the
// unmasked flow is silent but the masked one is not. Flows 0 and 2 cross
// between cores (0,3) and (0,4), and each switchbox there has one arbiter
// left, so each can hold the arbiter the other needs, whichever flow comes
// first.

// CHECK: error: Unable to find a legal routing: packet flows can deadlock holding arbiters across switchboxes, and no arbiter assignment found avoids it.

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    %sb03 = aie.switchbox(%t03) {
      %a1_0 = aie.amsel<1> (0)  %a1_1 = aie.amsel<1> (1)  %a1_2 = aie.amsel<1> (2)  %a1_3 = aie.amsel<1> (3)
      %a2_0 = aie.amsel<2> (0)  %a2_1 = aie.amsel<2> (1)  %a2_2 = aie.amsel<2> (2)  %a2_3 = aie.amsel<2> (3)
      %a3_0 = aie.amsel<3> (0)  %a3_1 = aie.amsel<3> (1)  %a3_2 = aie.amsel<3> (2)  %a3_3 = aie.amsel<3> (3)
      %a4_0 = aie.amsel<4> (0)  %a4_1 = aie.amsel<4> (1)  %a4_2 = aie.amsel<4> (2)  %a4_3 = aie.amsel<4> (3)
      %a5_0 = aie.amsel<5> (0)  %a5_1 = aie.amsel<5> (1)  %a5_2 = aie.amsel<5> (2)  %a5_3 = aie.amsel<5> (3)
      %m1 = aie.masterset(North : 0, %a1_0, %a1_1, %a1_2, %a1_3)
      %m2 = aie.masterset(North : 1, %a2_0, %a2_1, %a2_2, %a2_3)
      %m3 = aie.masterset(North : 2, %a3_0, %a3_1, %a3_2, %a3_3)
      %m4 = aie.masterset(North : 3, %a4_0, %a4_1, %a4_2, %a4_3)
      %m5 = aie.masterset(North : 4, %a5_0, %a5_1, %a5_2, %a5_3)
    }
    %sb04 = aie.switchbox(%t04) {
      %a1_0 = aie.amsel<1> (0)  %a1_1 = aie.amsel<1> (1)  %a1_2 = aie.amsel<1> (2)  %a1_3 = aie.amsel<1> (3)
      %a2_0 = aie.amsel<2> (0)  %a2_1 = aie.amsel<2> (1)  %a2_2 = aie.amsel<2> (2)  %a2_3 = aie.amsel<2> (3)
      %a3_0 = aie.amsel<3> (0)  %a3_1 = aie.amsel<3> (1)  %a3_2 = aie.amsel<3> (2)  %a3_3 = aie.amsel<3> (3)
      %a4_0 = aie.amsel<4> (0)  %a4_1 = aie.amsel<4> (1)  %a4_2 = aie.amsel<4> (2)  %a4_3 = aie.amsel<4> (3)
      %a5_0 = aie.amsel<5> (0)  %a5_1 = aie.amsel<5> (1)  %a5_2 = aie.amsel<5> (2)  %a5_3 = aie.amsel<5> (3)
      %m1 = aie.masterset(North : 0, %a1_0, %a1_1, %a1_2, %a1_3)
      %m2 = aie.masterset(North : 1, %a2_0, %a2_1, %a2_2, %a2_3)
      %m3 = aie.masterset(North : 2, %a3_0, %a3_1, %a3_2, %a3_3)
      %m4 = aie.masterset(North : 3, %a4_0, %a4_1, %a4_2, %a4_3)
      %m5 = aie.masterset(North : 4, %a5_0, %a5_1, %a5_2, %a5_3)
    }
    aie.packet_flow(0) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t04, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%t05, DMA : 0> aie.packet_dest<%t03, DMA : 0> }
    aie.packet_flow(2, mask = 30) { aie.packet_source<%t05, DMA : 0> aie.packet_dest<%t03, DMA : 0> }
    %buf = aie.buffer(%t05) : memref<16xi32>
    aie.mem(%t05) {
      %0 = aie.dma_start(MM2S, 0, ^send, ^end)
    ^send:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
      aie.next_bd ^send
    ^end:
      aie.end
    }
  }
}

// -----

// CHECK: error: Unable to find a legal routing: packet flows can deadlock holding arbiters across switchboxes, and no arbiter assignment found avoids it.

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    %sb03 = aie.switchbox(%t03) {
      %a1_0 = aie.amsel<1> (0)  %a1_1 = aie.amsel<1> (1)  %a1_2 = aie.amsel<1> (2)  %a1_3 = aie.amsel<1> (3)
      %a2_0 = aie.amsel<2> (0)  %a2_1 = aie.amsel<2> (1)  %a2_2 = aie.amsel<2> (2)  %a2_3 = aie.amsel<2> (3)
      %a3_0 = aie.amsel<3> (0)  %a3_1 = aie.amsel<3> (1)  %a3_2 = aie.amsel<3> (2)  %a3_3 = aie.amsel<3> (3)
      %a4_0 = aie.amsel<4> (0)  %a4_1 = aie.amsel<4> (1)  %a4_2 = aie.amsel<4> (2)  %a4_3 = aie.amsel<4> (3)
      %a5_0 = aie.amsel<5> (0)  %a5_1 = aie.amsel<5> (1)  %a5_2 = aie.amsel<5> (2)  %a5_3 = aie.amsel<5> (3)
      %m1 = aie.masterset(North : 0, %a1_0, %a1_1, %a1_2, %a1_3)
      %m2 = aie.masterset(North : 1, %a2_0, %a2_1, %a2_2, %a2_3)
      %m3 = aie.masterset(North : 2, %a3_0, %a3_1, %a3_2, %a3_3)
      %m4 = aie.masterset(North : 3, %a4_0, %a4_1, %a4_2, %a4_3)
      %m5 = aie.masterset(North : 4, %a5_0, %a5_1, %a5_2, %a5_3)
    }
    %sb04 = aie.switchbox(%t04) {
      %a1_0 = aie.amsel<1> (0)  %a1_1 = aie.amsel<1> (1)  %a1_2 = aie.amsel<1> (2)  %a1_3 = aie.amsel<1> (3)
      %a2_0 = aie.amsel<2> (0)  %a2_1 = aie.amsel<2> (1)  %a2_2 = aie.amsel<2> (2)  %a2_3 = aie.amsel<2> (3)
      %a3_0 = aie.amsel<3> (0)  %a3_1 = aie.amsel<3> (1)  %a3_2 = aie.amsel<3> (2)  %a3_3 = aie.amsel<3> (3)
      %a4_0 = aie.amsel<4> (0)  %a4_1 = aie.amsel<4> (1)  %a4_2 = aie.amsel<4> (2)  %a4_3 = aie.amsel<4> (3)
      %a5_0 = aie.amsel<5> (0)  %a5_1 = aie.amsel<5> (1)  %a5_2 = aie.amsel<5> (2)  %a5_3 = aie.amsel<5> (3)
      %m1 = aie.masterset(North : 0, %a1_0, %a1_1, %a1_2, %a1_3)
      %m2 = aie.masterset(North : 1, %a2_0, %a2_1, %a2_2, %a2_3)
      %m3 = aie.masterset(North : 2, %a3_0, %a3_1, %a3_2, %a3_3)
      %m4 = aie.masterset(North : 3, %a4_0, %a4_1, %a4_2, %a4_3)
      %m5 = aie.masterset(North : 4, %a5_0, %a5_1, %a5_2, %a5_3)
    }
    aie.packet_flow(0) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t04, DMA : 0> }
    aie.packet_flow(2, mask = 30) { aie.packet_source<%t05, DMA : 0> aie.packet_dest<%t03, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%t05, DMA : 0> aie.packet_dest<%t03, DMA : 0> }
    %buf = aie.buffer(%t05) : memref<16xi32>
    aie.mem(%t05) {
      %0 = aie.dma_start(MM2S, 0, ^send, ^end)
    ^send:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
      aie.next_bd ^send
    ^end:
      aie.end
    }
  }
}
