//===- test_create_packet_flows_shim1.mlir ----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-packet-flows %s | FileCheck %s

// CHECK-LABEL: module @aie_module {
// CHECK:   %[[VAL_0:.*]] = AIE.tile(7, 0)
// CHECK:   %[[VAL_1:.*]] = AIE.shimmux(%[[VAL_0:.*]])  {
// CHECK:     AIE.connect<DMA : 0, North : 3>
// CHECK:   }
// CHECK:   %[[VAL_2:.*]] = AIE.switchbox(%[[VAL_0:.*]]) {
// CHECK:     %[[VAL_3:.*]] = AIE.amsel<0> (0)
// CHECK:     %[[VAL_4:.*]] = AIE.masterset(North : 0, %[[VAL_3:.*]])
// CHECK:     AIE.packetrules(South : 3) {
// CHECK:       AIE.rule(31, 3, %[[VAL_3:.*]])
// CHECK:     }
// CHECK:   }
// CHECK:   %[[VAL_5:.*]] = AIE.tile(7, 1)
// CHECK:   %[[VAL_6:.*]] = AIE.switchbox(%[[VAL_5:.*]]) {
// CHECK:     %[[VAL_7:.*]] = AIE.amsel<0> (0)
// CHECK:     %[[VAL_8:.*]] = AIE.masterset(DMA : 0, %[[VAL_7:.*]])
// CHECK:     AIE.packetrules(South : 0) {
// CHECK:       AIE.rule(31, 3, %[[VAL_7:.*]])
// CHECK:     }
// CHECK:   }

//
// one-to-one shim DMA source 
//
module @aie_module  {
  %t70 = AIE.tile(7, 0)
  %t71 = AIE.tile(7, 1)

  AIE.packet_flow(0x3) {
    AIE.packet_source<%t70, DMA : 0>
    AIE.packet_dest<%t71, DMA : 0>
  }
}
