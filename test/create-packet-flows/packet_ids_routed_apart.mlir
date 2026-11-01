//===- packet_ids_routed_apart.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s | FileCheck %s

// Id 29 leaves (2, 1) DMA:3 by DMA:3, whose arbiter id 17 needs too, and id 23
// must not share an arbiter with id 17. Both ids go to (2, 3) DMA:0, so one
// tree from (2, 1) DMA:3 would carry them by the same master ports; id 23
// leaves by a North channel of its own instead.

// CHECK-LABEL: aie.switchbox(%mem_tile_2_1) {
// CHECK:         %[[ID29:.*]] = aie.amsel<0> (0)
// CHECK:         %[[ID23:.*]] = aie.amsel<1> (0)
// CHECK:         %[[ID17:.*]] = aie.amsel<0> (1)
// CHECK:         aie.masterset(DMA : 3, %[[ID29]], %[[ID17]])
// CHECK:         aie.masterset(North : {{[0-9]}}, %[[ID29]])
// CHECK:         aie.masterset(North : {{[0-9]}}, %[[ID23]])
// CHECK:         aie.packet_rules(DMA : 3) {
// CHECK-NEXT:      aie.rule(31, 23, %[[ID23]])
// CHECK-NEXT:      aie.rule(31, 29, %[[ID29]])
// CHECK-LABEL: aie.switchbox(%tile_2_3) {
// CHECK:         aie.masterset(DMA : 0, %[[TO_DMA0:[0-9]+]])
// CHECK:         aie.rule(31, 23, %[[TO_DMA0]])
// CHECK:         aie.rule(31, 29, %[[TO_DMA0]])

module {
  aie.device(npu2_3col) {
    %t_1_0 = aie.tile(1, 0)
    %t_2_1 = aie.tile(2, 1)
    %t_2_3 = aie.tile(2, 3)
    aie.packet_flow(17) { aie.packet_source<%t_2_3, DMA : 0> aie.packet_source<%t_1_0, DMA : 1> aie.packet_dest<%t_2_1, DMA : 3> }
    aie.packet_flow(29) { aie.packet_source<%t_2_1, DMA : 3> aie.packet_dest<%t_2_1, DMA : 3> aie.packet_dest<%t_2_3, DMA : 0> }
    aie.packet_flow(23) { aie.packet_source<%t_2_1, DMA : 3> aie.packet_dest<%t_2_3, DMA : 0> }
  }
}
