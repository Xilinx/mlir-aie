//===- existing_packet_rules_block_circuit.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Packet rules already on (0,2) North:0 make that port packet switched, so the
// circuit flow from (0,3) enters (0,2) by another port, while the new packet
// flow joins the existing rules there.

// CHECK-LABEL: aie.switchbox(%tile_0_2)
// CHECK:         %[[OLD:.*]] = aie.amsel<0> (0)
// CHECK:         aie.masterset(DMA : 0, %[[OLD]])
// CHECK:         aie.connect<North : [[CIRCUIT:[1-3]]], South : 1>
// CHECK:         %[[NEW:.*]] = aie.amsel<1> (0)
// CHECK:         aie.masterset(South : [[PKT:[0-3]]], %[[NEW]])
// CHECK:         aie.packet_rules(North : 0) {
// CHECK-NEXT:      aie.rule(31, 3, %[[OLD]])
// CHECK-NEXT:      aie.rule(31, 4, %[[NEW]])
// CHECK-NEXT:    }
// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK-DAG:     aie.connect<North : 1, DMA : 1>
// CHECK-DAG:     aie.packet_rules(North : [[PKT]])
// CHECK-LABEL: aie.switchbox(%tile_0_3)
// CHECK:         aie.connect<DMA : 0, South : [[CIRCUIT]]>
// CHECK:         aie.masterset(South : 0,

module {
  aie.device(npu1_1col) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %sb02 = aie.switchbox(%t02) {
      %a0 = aie.amsel<0> (0)
      %m = aie.masterset(DMA : 0, %a0)
      aie.packet_rules(North : 0) {
        aie.rule(31, 3, %a0)
      }
    }
    aie.flow(%t03, DMA : 0, %t01, DMA : 1)
    aie.packet_flow(4) {
      aie.packet_source<%t03, DMA : 1>
      aie.packet_dest<%t01, DMA : 0>
    }
  }
}
