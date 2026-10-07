//===- packet_rules_first_match.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s

// A slave port holds four packet rules, and a packet takes the first rule it
// matches. The shortest routes carry all five flows up column 3 into core
// (3,2) on one port, where ids 24 and 6 go north, 28 and 1 west, and 3 to the
// DMA. Rules that each match only their own ids need five. In order, four do:
// the last matches every even id, and only 24 and 6 are left to reach it.

// NOWARN-NOT: {{warning|error}}

// CHECK-LABEL: aie.switchbox(%tile_3_2)
// CHECK-DAG:     aie.masterset(DMA : 0, %[[DMA:[0-9]+]])
// CHECK-DAG:     aie.masterset(West : 0, %[[WEST:[0-9]+]])
// CHECK-DAG:     aie.masterset(North : {{[0-9]+}}, %[[NORTH:[0-9]+]])
// CHECK:         aie.packet_rules(South : {{[0-9]+}}) {
// CHECK-NEXT:      aie.rule(31, 1, %[[WEST]])
// CHECK-NEXT:      aie.rule(31, 3, %[[DMA]])
// CHECK-NEXT:      aie.rule(31, 28, %[[WEST]])
// CHECK-NEXT:      aie.rule(1, 0, %[[NORTH]])
// CHECK-NEXT:    }

module {
  aie.device(npu2) {
    %t_2_1 = aie.tile(2, 1)
    %t_2_4 = aie.tile(2, 4)
    %t_2_5 = aie.tile(2, 5)
    %t_3_0 = aie.tile(3, 0)
    %t_3_1 = aie.tile(3, 1)
    %t_3_2 = aie.tile(3, 2)
    aie.packet_flow(24) { aie.packet_source<%t_3_0, DMA : 0> aie.packet_dest<%t_2_4, DMA : 0> }
    aie.packet_flow(6) { aie.packet_source<%t_3_0, DMA : 0> aie.packet_dest<%t_2_5, DMA : 0> }
    aie.packet_flow(28) { aie.packet_source<%t_3_1, DMA : 5> aie.packet_dest<%t_2_5, DMA : 0> }
    aie.packet_flow(3) { aie.packet_source<%t_3_1, DMA : 5> aie.packet_dest<%t_3_2, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%t_3_1, DMA : 5> aie.packet_dest<%t_2_1, DMA : 4> }
  }
}
