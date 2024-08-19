//===- test_create_packet_flows2.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// partial multicast
module @test_create_packet_flows2 {
 aie.device(xcvc1902) {
// CHECK-LABEL: module @test_create_packet_flows2 {
// CHECK:         %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:         %[[VAL_1:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:           %[[VAL_6:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_7:.*]] = aie.amsel<0> (1)
// CHECK:           %[[VAL_4:.*]] = aie.masterset(Core : 0, %[[VAL_2:.*]])
// VAL_3 should also appear here, but it's difficult to filecheck.
// CHECK:           %[[VAL_5:.*]] = aie.masterset(Core : 1,
// CHECK-SACore :      %[[VAL_2]]
// CHECK:           aie.packet_rules(West : 0) {
// CHECK:             aie.rule(31, 1, %[[VAL_3:.*]])
// CHECK:             aie.rule(31, 0, %[[VAL_2]])
// CHECK:           }
// CHECK:         }
// CHECK:       }
  %t11 = aie.tile(1, 1)

  aie.packet_flow(0x0) {
    aie.packet_source<%t11, West : 0>
    aie.packet_dest<%t11, Core : 0>
    aie.packet_dest<%t11, Core : 1>
  }

  aie.packet_flow(0x1) {
    aie.packet_source<%t11, West : 0>
    aie.packet_dest<%t11, Core : 1>
  }
 }
}
