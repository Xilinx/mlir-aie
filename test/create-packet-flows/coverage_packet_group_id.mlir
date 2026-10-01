//===- coverage_packet_group_id.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Flows 1 and 3 share a destination, so they share a channel down to it even
// though flow 2, added between them, belongs to another group.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         aie.packet_rules(North : {{[0-9]+}}) {
// CHECK-NEXT:      aie.rule(29, 1, %{{.*}})
// CHECK-NEXT:    }
// CHECK-NOT:     aie.packet_rules
// CHECK-LABEL: aie.switchbox(%tile_0_2)
// CHECK:         aie.rule(29, 1, %{{.*}})

module {
  aie.device(npu1_1col) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    aie.packet_flow(1) {
      aie.packet_source<%t03, DMA : 0>
      aie.packet_dest<%t01, DMA : 0>
    }
    aie.packet_flow(2) {
      aie.packet_source<%t04, DMA : 0>
      aie.packet_dest<%t02, DMA : 0>
    }
    aie.packet_flow(3) {
      aie.packet_source<%t03, DMA : 1>
      aie.packet_dest<%t01, DMA : 0>
    }
  }
}
