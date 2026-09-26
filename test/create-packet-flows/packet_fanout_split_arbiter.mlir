//===- packet_fanout_split_arbiter.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s | FileCheck %s

// Memtile (4,1) DMA:1 sends id 6 up North:5, id 9 down South:2, and id 14
// both ways. A rule selects one arbiter, so every master port id 14 leaves by
// has to be on the same arbiter as the others: all three ids take arbiter 0,
// each on its own msel. Taking arbiter 1 for id 9's South:2 would leave no
// arbiter for id 14 that North:5 is also on, and id 14 would never reach (4,2)
// DMA:1.

// CHECK-LABEL: aie.switchbox(%mem_tile_4_1)
// CHECK:         %[[ID6:.*]] = aie.amsel<0> (0)
// CHECK:         %[[ID9:.*]] = aie.amsel<0> (1)
// CHECK:         %[[ID14:.*]] = aie.amsel<0> (2)
// CHECK:         aie.masterset(South : 2, %[[ID9]], %[[ID14]])
// CHECK:         aie.masterset(North : 5, %[[ID6]], %[[ID14]])
// CHECK:         aie.packet_rules(DMA : 1) {
// CHECK-NEXT:      aie.rule(31, 6, %[[ID6]])
// CHECK-NEXT:      aie.rule(31, 9, %[[ID9]])
// CHECK-NEXT:      aie.rule(31, 14, %[[ID14]])
// CHECK-LABEL: aie.switchbox(%tile_4_2)
// CHECK:         %[[DMA1:.*]] = aie.amsel<1> (0)
// CHECK:         aie.masterset(DMA : 1, %[[DMA1]])
// CHECK:         aie.packet_rules(South : 5) {
// CHECK:           aie.rule(31, 14, %[[DMA1]])

module {
  aie.device(npu2) {
    %t_4_0 = aie.tile(4, 0)
    %t_4_1 = aie.tile(4, 1)
    %t_4_2 = aie.tile(4, 2)
    %t_4_5 = aie.tile(4, 5)
    aie.packet_flow(14) { aie.packet_source<%t_4_1, DMA : 1> aie.packet_dest<%t_4_0, DMA : 0> aie.packet_dest<%t_4_1, DMA : 4> aie.packet_dest<%t_4_2, DMA : 1> }
    aie.packet_flow(9) { aie.packet_source<%t_4_1, DMA : 1> aie.packet_dest<%t_4_1, DMA : 3> }
    aie.packet_flow(6) { aie.packet_source<%t_4_1, DMA : 1> aie.packet_dest<%t_4_5, Core : 0> }
  }
}
