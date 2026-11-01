//===- packet_tree_branches_onto_its_own_arbiter.mlir ----------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s

// Id 28 from (3,5) reaches (0,1) DMA:2 on the arbiter DMA:2 and DMA:4 already
// share, so branching to DMA:4 at (0,1) puts no new flow on it. Branching
// higher up enters (0,1) twice and needs a fifth msel there. Reduced from
// router_mutation.py npu2 seed 1902.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         aie.masterset(DMA : 4, %[[SEL:[0-9]+]])
// CHECK:           aie.rule(31, 28, %[[SEL]])
// CHECK:           aie.rule(31, 28, %[[SEL]])
// CHECK-LABEL: aie.switchbox(%tile_0_2)

module {
  aie.device(npu2_4col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    %t_3_5 = aie.tile(3, 5)
    aie.packet_flow(28) { aie.packet_source<%t_0_4, DMA : 0> aie.packet_source<%t_3_5, Core : 0> aie.packet_dest<%t_0_1, DMA : 2> aie.packet_dest<%t_0_1, DMA : 4> }
    aie.packet_flow(1) { aie.packet_source<%t_0_4, DMA : 0> aie.packet_source<%t_0_2, DMA : 0> aie.packet_dest<%t_0_0, DMA : 0> aie.packet_dest<%t_0_1, DMA : 2> }
    aie.packet_flow(23) { aie.packet_source<%t_0_4, DMA : 0> aie.packet_dest<%t_0_5, DMA : 0> }
  }
}
