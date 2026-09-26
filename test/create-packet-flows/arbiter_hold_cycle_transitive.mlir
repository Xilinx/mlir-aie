//===- arbiter_hold_cycle_transitive.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s

// A cycle of waits can close through arbiters at two switchboxes, with no two
// flows at either one able to deadlock each other:
//
//   flow 0: core (0,2) --> core (0,4) S2MM 0
//   flow 1: core (0,5) --> core (0,4) S2MM 1
//   flow 2: core (0,5) --> memtile (0,1)
//   flow 3: core (0,4) --> core (0,3)
//
// Core (0,4) takes a buffer from flow 1 before each one from flow 0. Existing
// master sets leave (0,5) one arbiter, so flows 1 and 2 share it, and (0,3) two
// arbiters for flows 0, 2 and 3. Should flows 0 and 2 share at (0,3): flow 0
// fills (0,4)'s buffer and stalls holding the arbiter at (0,3); flow 2 takes
// the arbiter at (0,5) and waits for that one; and flow 1, which (0,4) needs
// next, waits behind flow 2 at (0,5).
//
// Planning (0,3) alone puts flow 2 with flow 0. Following waits through the
// whole design finds the cycle and moves flow 3 there instead.

// NOWARN-NOT: {{warning|error}}

// CHECK-LABEL: aie.switchbox(%tile_0_3)
// CHECK:         %[[FLOW0:.*]] = aie.amsel<0> (0)
// CHECK:         %[[FLOW2:.*]] = aie.amsel<1> (0)
// CHECK:         %[[FLOW3:.*]] = aie.amsel<0> (1)
// CHECK:         aie.masterset(DMA : 1, %[[FLOW3]])
// CHECK:         aie.masterset(South : {{[0-3]}}, %[[FLOW2]])
// CHECK:         aie.masterset(North : 5, %[[FLOW0]])
// CHECK:         aie.rule(31, 3, %[[FLOW3]])
// CHECK:         aie.rule(31, 2, %[[FLOW2]])
// CHECK:         aie.rule(31, 0, %[[FLOW0]])

// CHECK-LABEL: aie.switchbox(%tile_0_5)
// CHECK:         %[[FLOW1:.*]] = aie.amsel<0> (0)
// CHECK:         %[[FLOW2_05:.*]] = aie.amsel<0> (1)
// CHECK:         aie.rule(31, 2, %[[FLOW2_05]])
// CHECK:         aie.rule(31, 1, %[[FLOW1]])

module {
  aie.device(npu1_1col) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)

    // Arbiters 2-5 taken at (0,3).
    %sb03 = aie.switchbox(%t03) {
      %a2_0 = aie.amsel<2> (0)  %a2_1 = aie.amsel<2> (1)  %a2_2 = aie.amsel<2> (2)  %a2_3 = aie.amsel<2> (3)
      %a3_0 = aie.amsel<3> (0)  %a3_1 = aie.amsel<3> (1)  %a3_2 = aie.amsel<3> (2)  %a3_3 = aie.amsel<3> (3)
      %a4_0 = aie.amsel<4> (0)  %a4_1 = aie.amsel<4> (1)  %a4_2 = aie.amsel<4> (2)  %a4_3 = aie.amsel<4> (3)
      %a5_0 = aie.amsel<5> (0)  %a5_1 = aie.amsel<5> (1)  %a5_2 = aie.amsel<5> (2)  %a5_3 = aie.amsel<5> (3)
      aie.masterset(North : 0, %a2_0, %a2_1, %a2_2, %a2_3)
      aie.masterset(North : 1, %a3_0, %a3_1, %a3_2, %a3_3)
      aie.masterset(North : 2, %a4_0, %a4_1, %a4_2, %a4_3)
      aie.masterset(North : 3, %a5_0, %a5_1, %a5_2, %a5_3)
    }
    // Arbiters 1-5 taken at (0,5).
    %sb05 = aie.switchbox(%t05) {
      %a1_0 = aie.amsel<1> (0)  %a1_1 = aie.amsel<1> (1)  %a1_2 = aie.amsel<1> (2)  %a1_3 = aie.amsel<1> (3)
      %a2_0 = aie.amsel<2> (0)  %a2_1 = aie.amsel<2> (1)  %a2_2 = aie.amsel<2> (2)  %a2_3 = aie.amsel<2> (3)
      %a3_0 = aie.amsel<3> (0)  %a3_1 = aie.amsel<3> (1)  %a3_2 = aie.amsel<3> (2)  %a3_3 = aie.amsel<3> (3)
      %a4_0 = aie.amsel<4> (0)  %a4_1 = aie.amsel<4> (1)  %a4_2 = aie.amsel<4> (2)  %a4_3 = aie.amsel<4> (3)
      %a5_0 = aie.amsel<5> (0)  %a5_1 = aie.amsel<5> (1)  %a5_2 = aie.amsel<5> (2)  %a5_3 = aie.amsel<5> (3)
      aie.masterset(DMA : 0, %a1_0, %a1_1, %a1_2, %a1_3)
      aie.masterset(DMA : 1, %a2_0, %a2_1, %a2_2, %a2_3)
      aie.masterset(Core : 0, %a3_0, %a3_1, %a3_2, %a3_3)
      aie.masterset(South : 2, %a4_0, %a4_1, %a4_2, %a4_3)
      aie.masterset(South : 3, %a5_0, %a5_1, %a5_2, %a5_3)
    }

    aie.packet_flow(0) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t04, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%t05, DMA : 0> aie.packet_dest<%t04, DMA : 1> }
    aie.packet_flow(2) { aie.packet_source<%t05, DMA : 1> aie.packet_dest<%t01, DMA : 0> }
    aie.packet_flow(3) { aie.packet_source<%t04, DMA : 0> aie.packet_dest<%t03, DMA : 1> }

    %src0 = aie.buffer(%t02) : memref<16xi32>
    aie.mem(%t02) {
      %0 = aie.dma_start(MM2S, 0, ^send0, ^end)
    ^send0:
      aie.dma_bd(%src0 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.next_bd ^send0
    ^end:
      aie.end
    }

    %src1 = aie.buffer(%t05) : memref<16xi32>
    %src2 = aie.buffer(%t05) : memref<16xi32>
    aie.mem(%t05) {
      %0 = aie.dma_start(MM2S, 0, ^send1, ^next)
    ^send1:
      aie.dma_bd(%src1 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.next_bd ^send1
    ^next:
      %1 = aie.dma_start(MM2S, 1, ^send2, ^end)
    ^send2:
      aie.dma_bd(%src2 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.next_bd ^send2
    ^end:
      aie.end
    }

    %buf0 = aie.buffer(%t04) : memref<16xi32>
    %buf1 = aie.buffer(%t04) : memref<16xi32>
    %src3 = aie.buffer(%t04) : memref<16xi32>
    %prod0 = aie.lock(%t04, 0) {init = 1 : i32}
    %cons0 = aie.lock(%t04, 1) {init = 0 : i32}
    %prod1 = aie.lock(%t04, 2) {init = 1 : i32}
    %cons1 = aie.lock(%t04, 3) {init = 0 : i32}
    aie.mem(%t04) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^recv0, ^next1)
    ^recv0:
      aie.use_lock(%prod0, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons0, Release, %one)
      aie.next_bd ^recv0
    ^next1:
      %1 = aie.dma_start(S2MM, 1, ^recv1, ^next2)
    ^recv1:
      aie.use_lock(%prod1, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf1 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%cons1, Release, %one)
      aie.next_bd ^recv1
    ^next2:
      %2 = aie.dma_start(MM2S, 0, ^send3, ^end)
    ^send3:
      aie.dma_bd(%src3 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
      aie.next_bd ^send3
    ^end:
      aie.end
    }
    aie.core(%t04) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        aie.use_lock(%cons1, AcquireGreaterEqual, %one)
        aie.use_lock(%cons0, AcquireGreaterEqual, %one)
        aie.use_lock(%prod0, Release, %one)
        aie.use_lock(%prod1, Release, %one)
      }
      aie.end
    }

    %dst3 = aie.buffer(%t03) : memref<16xi32>
    aie.mem(%t03) {
      %0 = aie.dma_start(S2MM, 1, ^recv, ^end)
    ^recv:
      aie.dma_bd(%dst3 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^recv
    ^end:
      aie.end
    }

    %dst2 = aie.buffer(%t01) : memref<16xi32>
    aie.memtile_dma(%t01) {
      %0 = aie.dma_start(S2MM, 0, ^recv, ^end)
    ^recv:
      aie.dma_bd(%dst2 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^recv
    ^end:
      aie.end
    }
  }
}
