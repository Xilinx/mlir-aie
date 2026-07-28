//===- subcube_cover_optimal.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Exact-minimum cover: {0,4} (avoiding sibling id 1) combine into one masked
// rule (27,0), not two. The (27,0) witness fails if the cover is not minimal.

// CHECK-LABEL: module @m {
// CHECK:         aie.switchbox({{.*}}) {
// CHECK:           %[[C0:.*]] = aie.amsel<0> (0)
// CHECK:           %[[C1:.*]] = aie.amsel<1> (0)
// CHECK:           aie.masterset(Core : 0, %[[C0]])
// CHECK:           aie.masterset(Core : 1, %[[C1]])
// CHECK:           aie.packet_rules(West : 0) {
// CHECK-DAG:         aie.rule(27, 0, %[[C0]])
// CHECK-DAG:         aie.rule(31, 3, %[[C0]])
// CHECK-DAG:         aie.rule(31, 1, %[[C1]])
// CHECK:           }
// CHECK:         }

module @m {
 aie.device(xcvc1902) {
  %t11 = aie.tile(1, 1)
  aie.packet_flow(0x0) { aie.packet_source<%t11, West : 0>  aie.packet_dest<%t11, Core : 0> }
  aie.packet_flow(0x3) { aie.packet_source<%t11, West : 0>  aie.packet_dest<%t11, Core : 0> }
  aie.packet_flow(0x4) { aie.packet_source<%t11, West : 0>  aie.packet_dest<%t11, Core : 0> }
  aie.packet_flow(0x1) { aie.packet_source<%t11, West : 0>  aie.packet_dest<%t11, Core : 1> }
 }
}
