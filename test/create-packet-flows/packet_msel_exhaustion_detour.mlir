//===- packet_msel_exhaustion_detour.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s

// Five ids leave (0,2) DMA:0 for five different sets of master ports, all
// holding Core:0. Routed straight, they need one arbiter with five msels, and
// an arbiter has four; this used to crash. Ids 3 and 4 go up to (0,3) and back
// down to DMA:1, which takes a second arbiter at (0,2). Id 5 goes up too, so
// ids 3 and 5 share a msel, and one rule after the rule for id 1 takes both.

// Nothing programs the DMAs, so the design itself may deadlock.
// NOWARN: warning: Flows can deadlock however they are routed
// NOWARN-NOT: {{warning|error}}

// CHECK-LABEL: aie.switchbox(%tile_0_2)
// CHECK-DAG:     aie.masterset(DMA : 1, %[[DOWN:[0-9]+]])
// CHECK-DAG:     aie.masterset(North : 1, %[[UP35:[0-9]+]], %[[UP4:[0-9]+]])
// CHECK:         aie.packet_rules(North : 1) {
// CHECK-NEXT:      aie.rule(24, 0, %[[DOWN]])
// CHECK:         aie.packet_rules(DMA : 0) {
// CHECK-NEXT:      aie.rule(31, 4, %[[UP4]])
// CHECK-NEXT:      aie.rule(31, 2, %{{.*}})
// CHECK-NEXT:      aie.rule(31, 1, %{{.*}})
// CHECK-NEXT:      aie.rule(25, 1, %[[UP35]])
// CHECK-NEXT:    }
// CHECK-LABEL: aie.switchbox(%tile_0_3)
// CHECK-DAG:     aie.masterset(DMA : 0, %[[DMA:[0-9]+]])
// CHECK-DAG:     aie.masterset(South : 1, %[[BACK:[0-9]+]])
// CHECK:         aie.packet_rules(South : 1) {
// CHECK-NEXT:      aie.rule(31, 5, %[[DMA]])
// CHECK-NEXT:      aie.rule(31, 4, %[[BACK]])
// CHECK-NEXT:      aie.rule(31, 3, %[[BACK]])
// CHECK-NEXT:    }

module {
  aie.device(npu1_1col) {
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    aie.packet_flow(1) { aie.packet_source<%t_0_2, DMA : 0> aie.packet_dest<%t_0_2, Core : 0> }
    aie.packet_flow(2) { aie.packet_source<%t_0_2, DMA : 0> aie.packet_dest<%t_0_2, Core : 0> aie.packet_dest<%t_0_2, DMA : 0> }
    aie.packet_flow(3) { aie.packet_source<%t_0_2, DMA : 0> aie.packet_dest<%t_0_2, Core : 0> aie.packet_dest<%t_0_2, DMA : 1> }
    aie.packet_flow(4) { aie.packet_source<%t_0_2, DMA : 0> aie.packet_dest<%t_0_2, Core : 0> aie.packet_dest<%t_0_2, DMA : 0> aie.packet_dest<%t_0_2, DMA : 1> }
    aie.packet_flow(5) { aie.packet_source<%t_0_2, DMA : 0> aie.packet_dest<%t_0_2, Core : 0> aie.packet_dest<%t_0_3, DMA : 0> }
  }
}
