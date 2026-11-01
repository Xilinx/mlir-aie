//===- coverage_existing_stream_ids.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s
// RUN: not aie-opt --split-input-file --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=ERR

// Routing already present carries ids 0-3 from (0,5) to (0,4) S2MM 1, and
// (0,5) MM2S 0 sends only id 1. (0,4) has only that route's arbiter left, so
// new flow 5 must share it, and the route's receiver drains only as flow 5's
// sender runs. One 64-byte packet with id 1 fits the receiver.

// CHECK-LABEL: aie.switchbox(%tile_0_4)
// CHECK:         %[[OLD:.*]] = aie.amsel<0> (0)
// CHECK:         aie.rule(28, 0, %[[OLD]])
// CHECK:         %[[NEW:.*]] = aie.amsel<0> (1)
// CHECK:         aie.rule(31, 5, %[[NEW]])

module {
  aie.device(npu1_1col) {
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    %sb05 = aie.switchbox(%t05) {
      %a0 = aie.amsel<0> (0)
      aie.masterset(South : 0, %a0)
      aie.packet_rules(DMA : 0) {
        aie.rule(28, 0, %a0)
      }
    }
    %sb04 = aie.switchbox(%t04) {
      %a0_0 = aie.amsel<0> (0)
      aie.masterset(DMA : 1, %a0_0)
      aie.packet_rules(North : 0) {
        aie.rule(28, 0, %a0_0)
      }
      %a1_0 = aie.amsel<1> (0)  %a1_1 = aie.amsel<1> (1)  %a1_2 = aie.amsel<1> (2)  %a1_3 = aie.amsel<1> (3)
      %a2_0 = aie.amsel<2> (0)  %a2_1 = aie.amsel<2> (1)  %a2_2 = aie.amsel<2> (2)  %a2_3 = aie.amsel<2> (3)
      %a3_0 = aie.amsel<3> (0)  %a3_1 = aie.amsel<3> (1)  %a3_2 = aie.amsel<3> (2)  %a3_3 = aie.amsel<3> (3)
      %a4_0 = aie.amsel<4> (0)  %a4_1 = aie.amsel<4> (1)  %a4_2 = aie.amsel<4> (2)  %a4_3 = aie.amsel<4> (3)
      %a5_0 = aie.amsel<5> (0)  %a5_1 = aie.amsel<5> (1)  %a5_2 = aie.amsel<5> (2)  %a5_3 = aie.amsel<5> (3)
      aie.masterset(North : 0, %a1_0, %a1_1, %a1_2, %a1_3)
      aie.masterset(North : 1, %a2_0, %a2_1, %a2_2, %a2_3)
      aie.masterset(North : 2, %a3_0, %a3_1, %a3_2, %a3_3)
      aie.masterset(North : 3, %a4_0, %a4_1, %a4_2, %a4_3)
      aie.masterset(Core : 0, %a5_0, %a5_1, %a5_2, %a5_3)
    }
    aie.packet_flow(5) {
      aie.packet_source<%t04, DMA : 0>
      aie.packet_dest<%t03, DMA : 1>
    }
    %src = aie.buffer(%t05) : memref<32xi32>
    aie.mem(%t05) {
      %0 = aie.dma_start(MM2S, 0, ^send, ^end)
    ^send:
      aie.dma_bd(%src : memref<32xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %buf = aie.buffer(%t04) : memref<16xi32>
    %prod = aie.lock(%t04, 0) {init = 1 : i32}
    %cons = aie.lock(%t04, 1) {init = 0 : i32}
    aie.mem(%t04) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 1, ^recv, ^next)
    ^recv:
      aie.use_lock(%prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %one)
      aie.next_bd ^recv
    ^next:
      %1 = aie.dma_start(MM2S, 0, ^send, ^end)
    ^send:
      aie.use_lock(%cons, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
      aie.use_lock(%prod, Release, %one)
      aie.next_bd ^send
    ^end:
      aie.end
    }
  }
}

// -----

// A 128-byte one does not.

// ERR: error: Unable to find a legal routing: {{.*}} packet flow (0, 5) DMA:0 -> (0, 4) DMA:1 (id 1) can hold arbiter 0 at tile (0, 4) that packet flow (0, 4) DMA:0 -> (0, 3) DMA:1 (id 5) needs.
// ERR-NOT: error

module {
  aie.device(npu1_1col) {
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    %sb05 = aie.switchbox(%t05) {
      %a0 = aie.amsel<0> (0)
      aie.masterset(South : 0, %a0)
      aie.packet_rules(DMA : 0) {
        aie.rule(28, 0, %a0)
      }
    }
    %sb04 = aie.switchbox(%t04) {
      %a0_0 = aie.amsel<0> (0)
      aie.masterset(DMA : 1, %a0_0)
      aie.packet_rules(North : 0) {
        aie.rule(28, 0, %a0_0)
      }
      %a1_0 = aie.amsel<1> (0)  %a1_1 = aie.amsel<1> (1)  %a1_2 = aie.amsel<1> (2)  %a1_3 = aie.amsel<1> (3)
      %a2_0 = aie.amsel<2> (0)  %a2_1 = aie.amsel<2> (1)  %a2_2 = aie.amsel<2> (2)  %a2_3 = aie.amsel<2> (3)
      %a3_0 = aie.amsel<3> (0)  %a3_1 = aie.amsel<3> (1)  %a3_2 = aie.amsel<3> (2)  %a3_3 = aie.amsel<3> (3)
      %a4_0 = aie.amsel<4> (0)  %a4_1 = aie.amsel<4> (1)  %a4_2 = aie.amsel<4> (2)  %a4_3 = aie.amsel<4> (3)
      %a5_0 = aie.amsel<5> (0)  %a5_1 = aie.amsel<5> (1)  %a5_2 = aie.amsel<5> (2)  %a5_3 = aie.amsel<5> (3)
      aie.masterset(North : 0, %a1_0, %a1_1, %a1_2, %a1_3)
      aie.masterset(North : 1, %a2_0, %a2_1, %a2_2, %a2_3)
      aie.masterset(North : 2, %a3_0, %a3_1, %a3_2, %a3_3)
      aie.masterset(North : 3, %a4_0, %a4_1, %a4_2, %a4_3)
      aie.masterset(Core : 0, %a5_0, %a5_1, %a5_2, %a5_3)
    }
    aie.packet_flow(5) {
      aie.packet_source<%t04, DMA : 0>
      aie.packet_dest<%t03, DMA : 1>
    }
    %src = aie.buffer(%t05) : memref<32xi32>
    aie.mem(%t05) {
      %0 = aie.dma_start(MM2S, 0, ^send, ^end)
    ^send:
      aie.dma_bd(%src : memref<32xi32> offset = 0 len = 32) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %buf = aie.buffer(%t04) : memref<16xi32>
    %prod = aie.lock(%t04, 0) {init = 1 : i32}
    %cons = aie.lock(%t04, 1) {init = 0 : i32}
    aie.mem(%t04) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 1, ^recv, ^next)
    ^recv:
      aie.use_lock(%prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons, Release, %one)
      aie.next_bd ^recv
    ^next:
      %1 = aie.dma_start(MM2S, 0, ^send, ^end)
    ^send:
      aie.use_lock(%cons, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
      aie.use_lock(%prod, Release, %one)
      aie.next_bd ^send
    ^end:
      aie.end
    }
  }
}
