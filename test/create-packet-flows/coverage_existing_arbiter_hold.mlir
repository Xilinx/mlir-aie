//===- coverage_existing_arbiter_hold.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s
// RUN: not aie-opt --split-input-file --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=ERR

// Packet flow 1 is already routed into core (0,4) on arbiter 0. Nothing
// programs (0,4)'s DMAs, so its receiver may wait on flow 3's sender, and
// flow 3 must take another arbiter.

// CHECK-LABEL: aie.switchbox(%tile_0_4)
// CHECK:         %[[FLOW1:.*]] = aie.amsel<0> (0)
// CHECK:         aie.rule(31, 1, %[[FLOW1]])
// CHECK:         %[[FLOW3:.*]] = aie.amsel<1> (0)
// CHECK:         aie.rule(31, 3, %[[FLOW3]])

module {
  aie.device(npu1_1col) {
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    %sb05 = aie.switchbox(%t05) {
      %a0_0 = aie.amsel<0> (0)
      aie.masterset(South : 0, %a0_0)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 1, %a0_0)
      }
    }
    %sb04 = aie.switchbox(%t04) {
      %a0_0 = aie.amsel<0> (0)
      aie.masterset(DMA : 1, %a0_0)
      aie.packet_rules(North : 0) {
        aie.rule(31, 1, %a0_0)
      }
    }
    aie.packet_flow(3) {
      aie.packet_source<%t04, DMA : 0>
      aie.packet_dest<%t03, DMA : 1>
    }
  }
}

// -----

// With arbiters 1-5 taken as well, flow 3 has nowhere else to go.

// ERR: error: Unable to find a legal routing: packet flows can deadlock holding arbiters across switchboxes, and no arbiter assignment found avoids it. packet flow (0, 5) DMA:0 -> (0, 4) DMA:1 (id 1) can hold arbiter 0 at tile (0, 4) that packet flow (0, 4) DMA:0 -> (0, 3) DMA:1 (id 3) needs.
// ERR-NOT: error

module {
  aie.device(npu1_1col) {
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    %sb05 = aie.switchbox(%t05) {
      %a0_0 = aie.amsel<0> (0)
      aie.masterset(South : 0, %a0_0)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 1, %a0_0)
      }
    }
    %sb04 = aie.switchbox(%t04) {
      %a0_0 = aie.amsel<0> (0)
      aie.masterset(DMA : 1, %a0_0)
      aie.packet_rules(North : 0) {
        aie.rule(31, 1, %a0_0)
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
    aie.packet_flow(3) {
      aie.packet_source<%t04, DMA : 0>
      aie.packet_dest<%t03, DMA : 1>
    }
  }
}
