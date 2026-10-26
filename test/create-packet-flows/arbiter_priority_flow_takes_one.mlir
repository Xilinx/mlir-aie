//===- arbiter_priority_flow_takes_one.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s

// Six flows into memtile (0,1) channels nothing programs, and a prioritized
// one out of it: no two may share an arbiter. Six arbiters would do if the
// prioritized flow left (0,1) on a circuit, but a prioritized hop is never
// circuit switched, so the router says no routing works before searching.

// CHECK: error: Unable to find a legal routing: at tile (0, 1), no two of
// CHECK-SAME: packet flow (0, 1) DMA:1 -> (2, 1) DMA:4 (id 4) can share an
// CHECK-SAME: arbiter, and each takes one there whatever the routing, but the switchbox has 6 free.

module {
  aie.device(npu2_3col) {
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_1_1 = aie.tile(1, 1)
    %t_1_4 = aie.tile(1, 4)
    %t_2_0 = aie.tile(2, 0)
    %t_2_1 = aie.tile(2, 1)
    aie.packet_flow(16) { aie.packet_source<%t_0_2, DMA : 1> aie.packet_dest<%t_0_1, DMA : 4> }
    aie.packet_flow(8) { aie.packet_source<%t_1_4, Core : 0> aie.packet_dest<%t_0_1, DMA : 2> }
    aie.packet_flow(25) { aie.packet_source<%t_1_1, DMA : 0> aie.packet_dest<%t_0_1, DMA : 5> }
    aie.packet_flow(30) { aie.packet_source<%t_1_4, DMA : 1> aie.packet_dest<%t_0_1, DMA : 1> }
    aie.packet_flow(30) { aie.packet_source<%t_1_1, DMA : 4> aie.packet_dest<%t_0_1, DMA : 3> }
    aie.packet_flow(31) { aie.packet_source<%t_2_0, DMA : 1> aie.packet_dest<%t_0_1, DMA : 0> }
    aie.packet_flow(4) { aie.packet_source<%t_0_1, DMA : 1> aie.packet_dest<%t_2_1, DMA : 4> } {priority_route = true}
  }
}
