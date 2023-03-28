//===- test_create_packet_flows1.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-packet-flows %s | FileCheck %s

// CHECK-LABEL: module @test_create_packet_flows1 {
// CHECK:   %0 = AIE.tile(1, 1)
// CHECK:   %1 = AIE.switchbox(%0) {
// CHECK:     %2 = AIE.amsel<0> (0)
// CHECK:     %3 = AIE.masterset(Core : 0, %2)
// CHECK:     AIE.packetrules(West : 1) {
// CHECK:       AIE.rule(31, 1, %2)
// CHECK:     }
// CHECK:     AIE.packetrules(West : 0) {
// CHECK:       AIE.rule(31, 0, %2)
// CHECK:     }
// CHECK:   }
// CHECK: }

// many-to-one, single arbiter
module @test_create_packet_flows1 {
 AIE.device(xcvc1902) {
  %t11 = AIE.tile(1, 1)

  AIE.packet_flow(0x0) {
    AIE.packet_source<%t11, West : 0>
    AIE.packet_dest<%t11, Core : 0>
  }

  AIE.packet_flow(0x1) {
    AIE.packet_source<%t11, West : 1>
    AIE.packet_dest<%t11, Core : 0>
  }
 }
}
