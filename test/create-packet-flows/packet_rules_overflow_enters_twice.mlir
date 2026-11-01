//===- packet_rules_overflow_enters_twice.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s

// Five destinations on one slave port need five rules, over the 4-slot budget.
// Existing connections leave (0,2) two links down to the memtile, so (0,2)
// sends id 0 down the second one and the memtile takes it on a slave port of
// its own.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[D1:.*]] = aie.amsel<0> (0)
// CHECK:         %[[D2:.*]] = aie.amsel<1> (0)
// CHECK:         %[[D3:.*]] = aie.amsel<2> (0)
// CHECK:         %[[D4:.*]] = aie.amsel<3> (0)
// CHECK:         %[[D0:.*]] = aie.amsel<4> (0)
// CHECK:         aie.masterset(DMA : 0, %[[D0]])
// CHECK:         aie.packet_rules(North : 0) {
// CHECK-NEXT:      aie.rule(31, 4, %[[D4]])
// CHECK-NEXT:      aie.rule(31, 3, %[[D3]])
// CHECK-NEXT:      aie.rule(31, 2, %[[D2]])
// CHECK-NEXT:      aie.rule(31, 1, %[[D1]])
// CHECK-NEXT:    }
// CHECK:         aie.packet_rules(North : 3) {
// CHECK-NEXT:      aie.rule(31, 0, %[[D0]])
// CHECK-NEXT:    }
// CHECK-LABEL: aie.switchbox(%tile_0_2)
// CHECK:         %[[APART:.*]] = aie.amsel<0> (0)
// CHECK:         %[[REST:.*]] = aie.amsel<1> (0)
// CHECK:         aie.masterset(South : 0, %[[REST]])
// CHECK:         aie.masterset(South : 3, %[[APART]])
// CHECK:         aie.packet_rules(DMA : 0) {
// CHECK-NEXT:      aie.rule(29, 1, %[[REST]])
// CHECK-NEXT:      aie.rule(31, 2, %[[REST]])
// CHECK-NEXT:      aie.rule(31, 4, %[[REST]])
// CHECK-NEXT:      aie.rule(31, 0, %[[APART]])
// CHECK-NEXT:    }

module @overbudget {
  aie.device(npu1_1col) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %sb01 = aie.switchbox(%t01) {
      aie.connect<DMA : 0, South : 0>
      aie.connect<DMA : 1, South : 1>
      aie.connect<DMA : 2, South : 2>
      aie.connect<DMA : 3, South : 3>
    }
    %sb02 = aie.switchbox(%t02) {
      aie.connect<DMA : 1, South : 1>
      aie.connect<Core : 0, South : 2>
    }
    aie.packet_flow(0x0) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t01, DMA : 0> }
    aie.packet_flow(0x1) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t01, DMA : 1> }
    aie.packet_flow(0x2) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t01, DMA : 2> }
    aie.packet_flow(0x3) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t01, DMA : 3> }
    aie.packet_flow(0x4) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t01, DMA : 4> }
  }
}
