//===- packet_shared_ids_split_last.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN --allow-empty

// (0,4) DMA:0 sends ids 0, 24 and 25, and must route apart from (0,0) DMA:0,
// (0,3) DMA:0 and (0,4) DMA:1. Routing each id it sends that those do not as a
// part of its own needs more channels down from (0,4) than it has, so the
// router does that only once routing the source as one tree fails. As one
// tree, its ids leave (0,4) on one arbiter through three msels.
// Reduced from router_properties.py npu2 seed 2256.

// CHECK-LABEL: aie.switchbox(%tile_0_4)
// CHECK-DAG:     %[[M0:.*]] = aie.amsel<0> (0)
// CHECK-DAG:     %[[M1:.*]] = aie.amsel<0> (1)
// CHECK-DAG:     %[[M2:.*]] = aie.amsel<0> (2)
// CHECK:         aie.packet_rules(DMA : 0) {
// CHECK-NEXT:      aie.rule(31, 25, %[[M2]])
// CHECK-NEXT:      aie.rule(31, 0, %[[M0]])
// CHECK-NEXT:      aie.rule(31, 24, %[[M1]])
// CHECK-NEXT:    }

// WARN-NOT: {{warning|error}}

module {
  aie.device(npu2_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    aie.flow(%t_0_1, DMA : 2, %t_0_5, DMA : 0)
    aie.flow(%t_0_1, DMA : 2, %t_0_4, DMA : 0)
    aie.flow(%t_0_1, DMA : 4, %t_0_2, DMA : 1)
    aie.flow(%t_0_5, DMA : 1, %t_0_1, DMA : 5)
    aie.flow(%t_0_1, DMA : 1, %t_0_2, Core : 0)
    aie.packet_flow(9) { aie.packet_source<%t_0_4, DMA : 1> aie.packet_dest<%t_0_2, DMA : 0> }
    aie.packet_flow(24) { aie.packet_source<%t_0_4, DMA : 0> aie.packet_source<%t_0_3, DMA : 0> aie.packet_dest<%t_0_2, DMA : 0> aie.packet_dest<%t_0_3, DMA : 0> aie.packet_dest<%t_0_5, Core : 0> }
    aie.packet_flow(0) { aie.packet_source<%t_0_4, DMA : 0> aie.packet_source<%t_0_0, DMA : 1> aie.packet_dest<%t_0_1, DMA : 4> }
    aie.packet_flow(25) { aie.packet_source<%t_0_4, DMA : 0> aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_3, Core : 0> aie.packet_dest<%t_0_5, Core : 0> }
  }
}
