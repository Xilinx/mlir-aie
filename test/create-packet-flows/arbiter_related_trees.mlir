//===- arbiter_related_trees.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s

// Both sources of flow 12 reach both its receivers, so the two trees share
// the arbiter of each receiver's master port whatever the routing, and wait
// on each other there already. Sharing another arbiter on the way adds no
// wait. Counting it as a conflict kept them apart where they must meet, and
// routing failed.

// Nothing programs the DMAs, so the design itself may deadlock.
// NOWARN: warning: Flows can deadlock however they are routed
// NOWARN-NOT: {{warning|error}}

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK-DAG:     %[[A:[0-9]+]] = aie.amsel<[[ARB:[0-9]]]> ({{[0-9]}})
// CHECK-DAG:     %[[B:[0-9]+]] = aie.amsel<[[ARB]]> ({{[0-9]}})
// CHECK-DAG:     aie.masterset(DMA : 2, %[[A]], %[[B]])
// CHECK-DAG:     aie.rule(31, 12, %[[A]])
// CHECK-DAG:     aie.rule(31, 12, %[[B]])

module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    aie.packet_flow(12) {
      aie.packet_source<%t04, DMA : 0>
      aie.packet_source<%t03, DMA : 1>
      aie.packet_dest<%t00, DMA : 0>
      aie.packet_dest<%t01, DMA : 2>
    }
    aie.packet_flow(17) {
      aie.packet_source<%t01, DMA : 3>
      aie.packet_dest<%t04, DMA : 1>
    }
  }
}
