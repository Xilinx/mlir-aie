//===- arbiter_hold_cycle_wormhole.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 | FileCheck %s --check-prefix=NOHOPS
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s

// Two packet flows cross between cores (0,3) and (0,4), one going up and one
// going down, and existing master sets leave each switchbox one arbiter:
//
//   flow 0: core (0,2) --> core (0,4)
//   flow 1: core (0,5) --> core (0,3)
//
// A packet holds every arbiter it has taken until its tail passes. Sharing
// arbiter 0 at both tiles lets flow 0's packet take it at (0,3) and wait for
// it at (0,4) while flow 1's packet holds it at (0,4) and waits for it at
// (0,3). Neither receiver has to stall for this: two packets in flight at
// once are enough, and on hardware it hangs.
//
// In one column the routing is fixed, so with every hop packet-switched no
// arbiter assignment avoids it.

// NOHOPS: error: Unable to find a legal routing: packet flows can deadlock holding arbiters across switchboxes, and no arbiter assignment found avoids it.
// NOHOPS-SAME: packet flow (0, 5) DMA:0 -> (0, 3) DMA:0 (id 1) can hold arbiter 0 at tile (0, 4) that packet flow (0, 2) DMA:0 -> (0, 4) DMA:0 (id 0) needs.
// NOHOPS-SAME: packet flow (0, 2) DMA:0 -> (0, 4) DMA:0 (id 0) can hold arbiter 0 at tile (0, 3) that packet flow (0, 5) DMA:0 -> (0, 3) DMA:0 (id 1) needs.

// With more master ports than free arbiters, each flow passes the tile where
// it is not delivered on a circuit, which takes no arbiter.

// NOWARN-NOT: {{warning|error}}

// CHECK-LABEL:   aie.switchbox(%tile_0_3)
// CHECK:           aie.connect<South : {{[0-5]}}, North : 5>
// CHECK:           %[[A3:.*]] = aie.amsel<0> (0)
// CHECK:           aie.masterset(DMA : 0, %[[A3]])
// CHECK:           aie.packet_rules(North : {{[0-3]}}) {
// CHECK-NEXT:        aie.rule(31, 1, %[[A3]])
// CHECK-LABEL:   aie.switchbox(%tile_0_4)
// CHECK:           aie.connect<North : {{[0-3]}}, South : {{[0-3]}}>
// CHECK:           %[[A4:.*]] = aie.amsel<0> (0)
// CHECK:           aie.masterset(DMA : 0, %[[A4]])
// CHECK:           aie.packet_rules(South : {{[0-5]}}) {
// CHECK-NEXT:        aie.rule(31, 0, %[[A4]])

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)

    // Arbiters 1-5 taken at (0,3) and (0,4).
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
    aie.packet_flow(1) { aie.packet_source<%t05, DMA : 0> aie.packet_dest<%t03, DMA : 0> }
  }
}
