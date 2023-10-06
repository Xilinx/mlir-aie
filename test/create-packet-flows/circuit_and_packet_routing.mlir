//===- circuit_and_packet_routing.mlir -------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-packet-flows %s | FileCheck %s

// CHECK-LABEL: module @aie_module {
// CHECK:   %[[VAL_0:.*]] = AIE.tile(7, 2)
// CHECK:   %[[VAL_1:.*]] = AIE.switchbox(%[[VAL_0:.*]]) {
// CHECK:     %[[VAL_2:.*]] = AIE.amsel<0> (0)
// CHECK:     %[[VAL_3:.*]] = AIE.masterset(DMA : 1, %[[VAL_2:.*]])
// CHECK:     AIE.packetrules(North : 3) {
// CHECK:       AIE.rule(31, 10, %[[VAL_2:.*]])
// CHECK:     }
// CHECK:   }
// CHECK:   %[[VAL_5:.*]] = AIE.tile(7, 3)
// CHECK:   %[[VAL_6:.*]] = AIE.switchbox(%[[VAL_5:.*]]) {
// CHECK:     %[[VAL_7:.*]] = AIE.amsel<0> (0)
// CHECK:     %[[VAL_8:.*]] = AIE.masterset(South : 3, %[[VAL_7:.*]])
// CHECK:     AIE.packetrules(DMA : 0) {
// CHECK:       AIE.rule(31, 10, %[[VAL_7:.*]])
// CHECK:     }
// CHECK:   }
// CHECK:   AIE.flow(%[[VAL_0]], DMA : 1, %[[VAL_5]], DMA : 0)

//
// one circuit routing and one packet routing
//

module @aie_module  {
 AIE.device(xcvc1902) {
  %t70 = AIE.tile(7, 2)
  %t71 = AIE.tile(7, 3)

  AIE.packet_flow(0xA) {
    AIE.packet_source<%t71, DMA : 0>
    AIE.packet_dest<%t70, DMA : 1>
  }
  AIE.flow(%t70, DMA : 1, %t71, DMA : 0)
 }
}
