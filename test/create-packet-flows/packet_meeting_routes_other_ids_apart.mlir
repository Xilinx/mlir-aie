//===- packet_meeting_routes_other_ids_apart.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --verify-diagnostics %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" --verify-diagnostics %s | FileCheck %s

// Id 3 from (0, 3) DMA:1 and (0, 0) DMA:0 reaches two receivers both send to,
// so the trees meet. (0, 3) DMA:1 also sends ids 4 and 20, which conflict with
// (0, 0)'s id 3, so they leave (0, 3) as parts of their own and the trees
// meet at (0, 3) on id 3 alone.

// CHECK-LABEL: aie.switchbox(%tile_0_3)
// CHECK:         %[[MEET:.*]] = aie.amsel<0> (0)
// CHECK:         %[[ID20:.*]] = aie.amsel<1> (0)
// CHECK:         %[[ID4:.*]] = aie.amsel<2> (0)
// CHECK:         aie.packet_rules(South : 1) {
// CHECK-NEXT:      aie.rule(31, 3, %[[MEET]])
// CHECK-NEXT:    }
// CHECK:         aie.packet_rules(DMA : 1) {
// CHECK-NEXT:      aie.rule(31, 3, %[[MEET]])
// CHECK-NEXT:      aie.rule(31, 20, %[[ID20]])
// CHECK-NEXT:      aie.rule(31, 4, %[[ID4]])

module {
  aie.device(npu1_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_5 = aie.tile(0, 5)
    aie.flow(%t_0_3, DMA : 0, %t_0_2, DMA : 1)
    aie.packet_flow(4) { aie.packet_source<%t_0_5, Core : 0> aie.packet_source<%t_0_3, DMA : 1> aie.packet_dest<%t_0_1, DMA : 2> }
    aie.packet_flow(20) { aie.packet_source<%t_0_3, DMA : 1> aie.packet_source<%t_0_5, DMA : 1> aie.packet_dest<%t_0_1, DMA : 5> }
    aie.packet_flow(3) { aie.packet_source<%t_0_3, DMA : 1> aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_1, DMA : 4> aie.packet_dest<%t_0_5, DMA : 1> }
  }
}
