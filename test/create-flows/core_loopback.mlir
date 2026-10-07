//===- core_loopback.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows %s | FileCheck %s

// A core tile's switchbox cannot connect Core to Core, so a flow from a core
// back into itself leaves the tile and comes back through a neighbour.

// CHECK-LABEL: module
// CHECK:         aie.switchbox(%tile_0_2) {
// CHECK-DAG:       aie.connect<{{[A-Za-z]+ : [0-9]+}}, Core : 0>
// CHECK-DAG:       aie.connect<Core : 0, {{[A-Za-z]+ : [0-9]+}}>
// CHECK:         }

module {
  aie.device(npu1_1col) {
    %t = aie.tile(0, 2)
    aie.flow(%t, Core : 0, %t, Core : 0)
  }
}

// -----

// The same as a packet flow.

// CHECK-LABEL: module
// CHECK:         aie.switchbox(%tile_0_2) {
// CHECK:           aie.masterset(Core : 0, %[[TO_CORE:.*]])
// CHECK:           aie.packet_rules(Core : 0) {
// CHECK-NOT:         %[[TO_CORE]])
// CHECK:           }
// CHECK:           aie.rule(31, 3, %[[TO_CORE]])
// CHECK:         }

module {
  aie.device(npu1_1col) {
    %t = aie.tile(0, 2)
    aie.packet_flow(3) {
      aie.packet_source<%t, Core : 0>
      aie.packet_dest<%t, Core : 0>
    }
  }
}
