//===- packet_routing_keep_pkt_header.mlir ---------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL: module @aie_module {
// CHECK:   %[[VAL_0:.*]] = aie.tile(6, 2)
// CHECK:   %[[VAL_1:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:     %[[VAL_2:.*]] = aie.amsel<0> (0)
// CHECK:     %[[VAL_3:.*]] = aie.masterset(DMA : 1, %[[VAL_2]])
// CHECK:     aie.packet_rules(North : 0) {
// CHECK:       aie.rule(31, 1, %[[VAL_2]])
// CHECK:     }
// CHECK:   }
// CHECK:   %[[VAL_4:.*]] = aie.tile(6, 3)
// CHECK:   %[[VAL_5:.*]] = aie.switchbox(%[[VAL_4]]) {
// CHECK:     %[[VAL_6:.*]] = aie.amsel<0> (0)
// CHECK:     %[[VAL_7:.*]] = aie.masterset(South : 0, %[[VAL_6]])
// CHECK:     aie.packet_rules(DMA : 0) {
// CHECK:       aie.rule(31, 1, %[[VAL_6]])
// CHECK:     }
// CHECK:   }
// CHECK:   %[[VAL_8:.*]] = aie.tile(7, 2)
// CHECK:   %[[VAL_9:.*]] = aie.switchbox(%[[VAL_8]]) {
// CHECK:     %[[VAL_10:.*]] = aie.amsel<0> (0)
// CHECK:     %[[VAL_11:.*]] = aie.masterset(DMA : 1, %[[VAL_10]]) {keep_pkt_header = "true"}
// CHECK:     aie.packet_rules(North : 0) {
// CHECK:       aie.rule(31, 2, %[[VAL_10]])
// CHECK:     }
// CHECK:   }
// CHECK:   %[[VAL_12:.*]] = aie.tile(7, 3)
// CHECK:   %[[VAL_13:.*]] = aie.switchbox(%[[VAL_12]]) {
// CHECK:     %[[VAL_14:.*]] = aie.amsel<0> (0)
// CHECK:     %[[VAL_15:.*]] = aie.masterset(South : 0, %[[VAL_14]])
// CHECK:     aie.packet_rules(DMA : 0) {
// CHECK:       aie.rule(31, 2, %[[VAL_14]])
// CHECK:     }
// CHECK:   }

//
// keep_pkt_header attribute overrides the downstream decision to drop the packet header
//

module @aie_module  {
 aie.device(xcvc1902) {
  %t62 = aie.tile(6, 2)
  %t63 = aie.tile(6, 3)
  %t72 = aie.tile(7, 2)
  %t73 = aie.tile(7, 3)

  aie.packet_flow(0x1) {
    aie.packet_source<%t63, DMA : 0>
    aie.packet_dest<%t62, DMA : 1>
  }

  aie.packet_flow(0x2) {
    aie.packet_source<%t73, DMA : 0>
    aie.packet_dest<%t72, DMA : 1>
  } {keep_pkt_header = true}
 }
}
