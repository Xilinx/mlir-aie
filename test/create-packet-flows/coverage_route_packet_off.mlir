//===- coverage_route_packet_off.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false route-packet=false" %s | FileCheck %s
// RUN: not aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 | FileCheck %s --check-prefix=ERR

// Every arbiter at tile (0,4) is taken, so none is left for packet flow 3.
// With packet routing off the flow is left alone rather than rejected.

// CHECK:     aie.packet_flow(3)
// ERR:       error: Unable to find a legal routing: at tile (0, 4), packet flow (0, 4) DMA:0 -> (0, 3) DMA:1 (id 3) takes an arbiter whatever the routing, but the switchbox has none free

module {
  aie.device(npu1_1col) {
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %sb04 = aie.switchbox(%t04) {
      %a0_0 = aie.amsel<0> (0)  %a0_1 = aie.amsel<0> (1)  %a0_2 = aie.amsel<0> (2)  %a0_3 = aie.amsel<0> (3)
      %a1_0 = aie.amsel<1> (0)  %a1_1 = aie.amsel<1> (1)  %a1_2 = aie.amsel<1> (2)  %a1_3 = aie.amsel<1> (3)
      %a2_0 = aie.amsel<2> (0)  %a2_1 = aie.amsel<2> (1)  %a2_2 = aie.amsel<2> (2)  %a2_3 = aie.amsel<2> (3)
      %a3_0 = aie.amsel<3> (0)  %a3_1 = aie.amsel<3> (1)  %a3_2 = aie.amsel<3> (2)  %a3_3 = aie.amsel<3> (3)
      %a4_0 = aie.amsel<4> (0)  %a4_1 = aie.amsel<4> (1)  %a4_2 = aie.amsel<4> (2)  %a4_3 = aie.amsel<4> (3)
      %a5_0 = aie.amsel<5> (0)  %a5_1 = aie.amsel<5> (1)  %a5_2 = aie.amsel<5> (2)  %a5_3 = aie.amsel<5> (3)
      aie.masterset(North : 0, %a0_0, %a0_1, %a0_2, %a0_3)
      aie.masterset(North : 1, %a1_0, %a1_1, %a1_2, %a1_3)
      aie.masterset(North : 2, %a2_0, %a2_1, %a2_2, %a2_3)
      aie.masterset(North : 3, %a3_0, %a3_1, %a3_2, %a3_3)
      aie.masterset(Core : 0, %a4_0, %a4_1, %a4_2, %a4_3)
      aie.masterset(DMA : 1, %a5_0, %a5_1, %a5_2, %a5_3)
    }
    aie.packet_flow(3) {
      aie.packet_source<%t04, DMA : 0>
      aie.packet_dest<%t03, DMA : 1>
    }
  }
}
