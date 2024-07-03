//===- test_create_packet_flows_shim1.mlir ----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL: module @aie_module {
// CHECK:   %[[VAL_0:.*]] = aie.tile(7, 0)
// CHECK:   %[[VAL_1:.*]] = aie.shim_mux(%[[VAL_0:.*]])  {
// CHECK:     aie.connect<DMA : 0, North : 3>
// CHECK:   }
// CHECK:   %[[VAL_2:.*]] = aie.switchbox(%[[VAL_0:.*]]) {
// CHECK:     %[[VAL_3:.*]] = aie.amsel<0> (0)
// CHECK:     %[[VAL_4:.*]] = aie.masterset(North : 0, %[[VAL_3:.*]])
// CHECK:     aie.packet_rules(South : 3) {
// CHECK:       aie.rule(31, 3, %[[VAL_3:.*]])
// CHECK:     }
// CHECK:   }
// CHECK:   %[[VAL_5:.*]] = aie.tile(7, 1)
// CHECK:   %[[VAL_6:.*]] = aie.switchbox(%[[VAL_5:.*]]) {
// CHECK:     %[[VAL_7:.*]] = aie.amsel<0> (0)
// CHECK:     %[[VAL_8:.*]] = aie.masterset(DMA : 0, %[[VAL_7:.*]])
// CHECK:     aie.packet_rules(South : 0) {
// CHECK:       aie.rule(31, 3, %[[VAL_7:.*]])
// CHECK:     }
// CHECK:   }

//
// one-to-one shim DMA source 
//
module @aie_module  {
 aie.device(xcvc1902) {
  %t70 = aie.tile(7, 0)
  %t71 = aie.tile(7, 1)

  aie.packet_flow(0x3) {
    aie.packet_source<%t70, DMA : 0>
    aie.packet_dest<%t71, DMA : 0>
  }
 }
}
