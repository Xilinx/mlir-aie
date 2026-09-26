//===- coverage_existing_rule_cover.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Ids 0-2 share a destination, but the rule covering them must not also match
// id 3, which an existing rule on the same slave port sends north.

// CHECK-LABEL: aie.switchbox(%tile_0_2)
// CHECK:         %[[N:.*]] = aie.amsel<0> (0)
// CHECK:         aie.packet_rules(DMA : 0) {
// CHECK-NEXT:      aie.rule(31, 3, %[[N]])
// CHECK-NEXT:      aie.rule(30, 0, %[[S:.*]])
// CHECK-NEXT:      aie.rule(31, 2, %[[S]])
// CHECK-NEXT:    }

module {
  aie.device(npu1_1col) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %sb02 = aie.switchbox(%t02) {
      %a0 = aie.amsel<0> (0)
      %m = aie.masterset(North : 0, %a0)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 3, %a0)
      }
    }
    aie.packet_flow(0) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t01, DMA : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t01, DMA : 0>
    }
    aie.packet_flow(2) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t01, DMA : 0>
    }
  }
}
