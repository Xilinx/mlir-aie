//===- arbiter_cut_tile_clique.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 | FileCheck %s --check-prefix=NOHOPS
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s

// Three flows from core (0,2) to core (0,4), into channels nothing programs,
// so any one can stall waiting on another: no two may share an arbiter. In one
// column every route passes (0,3), where existing master sets leave two
// arbiters. With every hop packet-switched, no routing works, and the router
// says so before searching for one.

// NOHOPS: error: Unable to find a legal routing: at tile (0, 3), no two of
// NOHOPS-SAME: packet flow (0, 2) DMA:0 -> (0, 4) DMA:0 (id 0), packet flow (0, 2) DMA:1 -> (0, 4) DMA:1 (id 1), packet flow (0, 2) Core:0 -> (0, 4) Core:0 (id 2)
// NOHOPS-SAME: can share an arbiter, and each takes one there whatever the routing, but the switchbox has 2 free.

// Three master ports for two free arbiters: one flow crosses (0,3) on a
// circuit, and the other two take an arbiter each.

// NOWARN-NOT: {{warning|error}}

// CHECK-LABEL: aie.switchbox(%tile_0_3)
// CHECK:         aie.connect<South : {{[0-5]}}, North : {{[0-5]}}>
// CHECK:         %[[A:.*]] = aie.amsel<0> (0)
// CHECK:         %[[B:.*]] = aie.amsel<1> (0)
// CHECK-DAG:     aie.masterset(North : {{[0-5]}}, %[[A]])
// CHECK-DAG:     aie.masterset(North : {{[0-5]}}, %[[B]])
// CHECK-NOT:     aie.masterset

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)

    // Arbiters 2-5 taken at (0,3).
    %sb03 = aie.switchbox(%t03) {
      %a2_0 = aie.amsel<2> (0)  %a2_1 = aie.amsel<2> (1)  %a2_2 = aie.amsel<2> (2)  %a2_3 = aie.amsel<2> (3)
      %a3_0 = aie.amsel<3> (0)  %a3_1 = aie.amsel<3> (1)  %a3_2 = aie.amsel<3> (2)  %a3_3 = aie.amsel<3> (3)
      %a4_0 = aie.amsel<4> (0)  %a4_1 = aie.amsel<4> (1)  %a4_2 = aie.amsel<4> (2)  %a4_3 = aie.amsel<4> (3)
      %a5_0 = aie.amsel<5> (0)  %a5_1 = aie.amsel<5> (1)  %a5_2 = aie.amsel<5> (2)  %a5_3 = aie.amsel<5> (3)
      aie.masterset(DMA : 0, %a2_0, %a2_1, %a2_2, %a2_3)
      aie.masterset(DMA : 1, %a3_0, %a3_1, %a3_2, %a3_3)
      aie.masterset(Core : 0, %a4_0, %a4_1, %a4_2, %a4_3)
      aie.masterset(South : 0, %a5_0, %a5_1, %a5_2, %a5_3)
    }

    aie.packet_flow(0) { aie.packet_source<%t02, DMA : 0>  aie.packet_dest<%t04, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%t02, DMA : 1>  aie.packet_dest<%t04, DMA : 1> }
    aie.packet_flow(2) { aie.packet_source<%t02, Core : 0> aie.packet_dest<%t04, Core : 0> }
  }
}
