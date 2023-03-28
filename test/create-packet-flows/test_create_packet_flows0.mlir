//===- test_create_packet_flows0.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-packet-flows %s | FileCheck %s

// one-to-many, single arbiter
module @test_create_packet_flows0 {
 AIE.device(xcvc1902) {
// CHECK-LABEL:   module @test_create_packet_flows0 {
// CHECK:           %[[VAL_0:.*]] = AIE.tile(1, 1)
// CHECK:           %[[VAL_1:.*]] = AIE.switchbox(%[[VAL_0]]) {
// The actual indices used for the amsel arguments is unimportant.
// CHECK:           %[[VAL_6:.*]] = AIE.amsel<0> (0)
// CHECK:           %[[VAL_7:.*]] = AIE.amsel<0> (1)
// CHECK:           %[[VAL_4:.*]] = AIE.masterset(Core : 0, %[[VAL_2:.*]])
// CHECK:           %[[VAL_5:.*]] = AIE.masterset(Core : 1, %[[VAL_3:.*]])
// CHECK:           AIE.packetrules(West : 0) {
// CHECK:             AIE.rule(31, 1, %[[VAL_3]])
// CHECK:             AIE.rule(31, 0, %[[VAL_2]])
// CHECK:           }
// CHECK:         }
// CHECK:       }
  %t11 = AIE.tile(1, 1)

  AIE.packet_flow(0x0) {
    AIE.packet_source<%t11, West : 0>
    AIE.packet_dest<%t11, Core : 0>
  }

  AIE.packet_flow(0x1) {
    AIE.packet_source<%t11, West : 0>
    AIE.packet_dest<%t11, Core : 1>
  }
 }
}
