//===- packet_rule_slots_reroute.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s

// A slave port holds four packet rules. The shortest routes carry all five
// flows up column 3 into core (3,2) on one port, where ids 24 and 6 go north,
// 28 and 1 west, and 3 to the DMA: two rules for each pair and one for id 3,
// five in all. The router takes ids 24 and 6 up column 2 instead.

// NOWARN-NOT: {{warning|error}}

// CHECK-LABEL: aie.switchbox(%tile_2_4)
// CHECK:         %[[TO_2_5:.*]] = aie.amsel<0> (0)
// CHECK:         %[[TO_DMA:.*]] = aie.amsel<1> (0)
// CHECK:         aie.packet_rules(South : 2) {
// CHECK-NEXT:      aie.rule(31, 6, %[[TO_2_5]])
// CHECK-NEXT:      aie.rule(31, 24, %[[TO_DMA]])
// CHECK-LABEL: aie.switchbox(%shim_noc_tile_3_0)
// CHECK:         aie.masterset(West : {{[0-9]+}}
// CHECK-LABEL: aie.switchbox(%tile_3_2)
// CHECK:         aie.packet_rules(South : {{[0-9]+}}) {
// CHECK-NEXT:      aie.rule(2, 0, %{{.*}})
// CHECK-NEXT:      aie.rule(31, 3, %{{.*}})
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
