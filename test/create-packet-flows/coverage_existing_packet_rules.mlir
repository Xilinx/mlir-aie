//===- coverage_existing_packet_rules.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: not aie-opt --split-input-file --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=ERR

// A new packet flow from a slave port that already has packet_rules adds its
// rule to that block instead of opening a second one on the same port.

// CHECK-LABEL: aie.switchbox(%tile_0_2)
// CHECK:         %[[A0:.*]] = aie.amsel<0> (0)
// CHECK:         aie.packet_rules(DMA : 0) {
// CHECK-NEXT:      aie.rule(31, 3, %[[A0]])
// CHECK-NEXT:      aie.rule(31, 5, %{{.*}})
// CHECK-NEXT:    }
// CHECK-NOT:     aie.packet_rules(DMA : 0)
// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)

module {
  aie.device(npu1_1col) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %sb02 = aie.switchbox(%t02) {
      %a0 = aie.amsel<0> (0)
      %m = aie.masterset(South : 0, %a0)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 3, %a0)
      }
    }
    aie.packet_flow(5) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t01, DMA : 0>
    }
    aie.flow(%t03, DMA : 0, %t01, DMA : 1)
  }
}

// -----

// The existing rule already matches the new flow's id and sends it elsewhere.

// ERR: error: 'aie.rule' op can lead to false packet id match for id 3

module {
  aie.device(npu1_1col) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %sb02 = aie.switchbox(%t02) {
      %a0 = aie.amsel<0> (0)
      %m = aie.masterset(South : 0, %a0)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 3, %a0)
      }
    }
    aie.packet_flow(3) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t01, DMA : 0>
    }
    aie.flow(%t03, DMA : 0, %t01, DMA : 1)
  }
}
