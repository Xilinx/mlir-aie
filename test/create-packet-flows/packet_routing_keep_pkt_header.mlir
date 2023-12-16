//===- packet_routing_keep_pkt_header.mlir ---------------------*- MLIR -*-===//
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
// CHECK:   %[[VAL_0:.*]] = AIE.tile(6, 2)
// CHECK:   %[[VAL_1:.*]] = AIE.switchbox(%[[VAL_0]]) {
// CHECK:     %[[VAL_2:.*]] = AIE.amsel<0> (0)
// CHECK:     %[[VAL_3:.*]] = AIE.masterset(DMA : 1, %[[VAL_2]])
// CHECK:     AIE.packet_rules(North : 3) {
// CHECK:       AIE.rule(31, 1, %[[VAL_2]])
// CHECK:     }
// CHECK:   }
// CHECK:   %[[VAL_4:.*]] = AIE.tile(6, 3)
// CHECK:   %[[VAL_5:.*]] = AIE.switchbox(%[[VAL_4]]) {
// CHECK:     %[[VAL_6:.*]] = AIE.amsel<0> (0)
// CHECK:     %[[VAL_7:.*]] = AIE.masterset(South : 3, %[[VAL_6]])
// CHECK:     AIE.packet_rules(DMA : 0) {
// CHECK:       AIE.rule(31, 1, %[[VAL_6]])
// CHECK:     }
// CHECK:   }
// CHECK:   %[[VAL_8:.*]] = AIE.tile(7, 2)
// CHECK:   %[[VAL_9:.*]] = AIE.switchbox(%[[VAL_8]]) {
// CHECK:     %[[VAL_10:.*]] = AIE.amsel<0> (0)
// CHECK:     %[[VAL_11:.*]] = AIE.masterset(DMA : 1, %[[VAL_10]]) {keep_pkt_header = "true"}
// CHECK:     AIE.packet_rules(North : 3) {
// CHECK:       AIE.rule(31, 2, %[[VAL_10]])
// CHECK:     }
// CHECK:   }
// CHECK:   %[[VAL_12:.*]] = AIE.tile(7, 3)
// CHECK:   %[[VAL_13:.*]] = AIE.switchbox(%[[VAL_12]]) {
// CHECK:     %[[VAL_14:.*]] = AIE.amsel<0> (0)
// CHECK:     %[[VAL_15:.*]] = AIE.masterset(South : 3, %[[VAL_14]])
// CHECK:     AIE.packet_rules(DMA : 0) {
// CHECK:       AIE.rule(31, 2, %[[VAL_14]])
// CHECK:     }
// CHECK:   }

//
// keep_pkt_header attribute overrides the downstream decision to drop the packet header
//

module @aie_module  {
 AIE.device(xcvc1902) {
  %t62 = AIE.tile(6, 2)
  %t63 = AIE.tile(6, 3)
  %t72 = AIE.tile(7, 2)
  %t73 = AIE.tile(7, 3)

  AIE.packet_flow(0x1) {
    AIE.packet_source<%t63, DMA : 0>
    AIE.packet_dest<%t62, DMA : 1>
  }

  AIE.packet_flow(0x2) {
    AIE.packet_source<%t73, DMA : 0>
    AIE.packet_dest<%t72, DMA : 1>
  } {keep_pkt_header = true}
 }
}
