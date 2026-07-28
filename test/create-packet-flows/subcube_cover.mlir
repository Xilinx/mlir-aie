//===- subcube_cover.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// {2,4} routed while avoiding sibling id 0 on the same slave port: the cover
// splits into rules that miss 0, where the single common-bits mask would not.

// CHECK-LABEL: module @test_subcube_cover {
// CHECK:         %[[T:.*]] = aie.tile(1, 1)
// CHECK:         aie.switchbox(%[[T]]) {
// CHECK:           %[[A0:.*]] = aie.amsel<0> (0)
// CHECK:           %[[A1:.*]] = aie.amsel<1> (0)
// CHECK:           aie.masterset(Core : 0, %[[A0]])
// CHECK:           aie.masterset(Core : 1, %[[A1]])
// CHECK:           aie.packet_rules(West : 0) {
// CHECK-DAG:         aie.rule(31, 2, %[[A1]])
// CHECK-DAG:         aie.rule(31, 4, %[[A1]])
// CHECK-DAG:         aie.rule(31, 0, %[[A0]])
// CHECK:           }
// CHECK:         }
module @test_subcube_cover {
 aie.device(xcvc1902) {
  %t11 = aie.tile(1, 1)

  aie.packet_flow(0x0) {
    aie.packet_source<%t11, West : 0>
    aie.packet_dest<%t11, Core : 0>
  }

  aie.packet_flow(0x2) {
    aie.packet_source<%t11, West : 0>
    aie.packet_dest<%t11, Core : 1>
  }

  aie.packet_flow(0x4) {
    aie.packet_source<%t11, West : 0>
    aie.packet_dest<%t11, Core : 1>
  }
 }
}
