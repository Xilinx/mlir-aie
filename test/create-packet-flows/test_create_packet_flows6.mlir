//===- test_create_packet_flows6.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: stephenn
// RUN: aie-opt --aie-create-packet-flows %s | FileCheck %s

// Fixme: may fail non-deterministically

module @test_create_packet_flows6 {
 AIE.device(xcvc1902) {
// CHECK-LABEL: module @test_create_packet_flows6 {
// CHECK:         %[[VAL_0:.*]] = AIE.tile(2, 2)
// CHECK:         %[[VAL_1:.*]] = AIE.switchbox(%[[VAL_0]]) {
// CHECK:           %{{.*}} = AIE.amsel<0> (0)
// CHECK:           %[[VAL_3:.*]] = AIE.masterset(East : 0, %[[VAL_2:.*]])
// CHECK:           AIE.packetrules(DMA : 0) {
// CHECK:             AIE.rule(28, 3, %[[VAL_2]])
// CHECK:           }
// CHECK:         }

// CHECK:         %[[VAL_4:.*]] = AIE.tile(3, 2)
// CHECK:         %[[VAL_5:.*]] = AIE.switchbox(%[[VAL_4]]) {
// CHECK:           %{{.*}} = AIE.amsel<0> (0)
// CHECK:           %{{.*}} = AIE.amsel<0> (1)
// CHECK:           %[[VAL_8:.*]] = AIE.masterset(DMA : 0, %[[VAL_7:.*]])
// CHECK:           %[[VAL_9:.*]] = AIE.masterset(East : 0, %[[VAL_6:.*]])
// CHECK:           AIE.packetrules(West : 0) {
// CHECK:             AIE.rule(28, 3, %[[VAL_6]])
// CHECK:             AIE.rule(31, 0, %[[VAL_7]])
// CHECK:           }
// CHECK:         }

// CHECK:         %[[VAL_10:.*]] = AIE.tile(4, 2)
// CHECK:         %[[VAL_11:.*]] = AIE.switchbox(%[[VAL_10]]) {
// CHECK:           %{{.*}} = AIE.amsel<0> (0)
// CHECK:           %{{.*}} = AIE.amsel<0> (1)
// CHECK:           %[[VAL_14:.*]] = AIE.masterset(DMA : 0, %[[VAL_13:.*]])
// CHECK:           %[[VAL_15:.*]] = AIE.masterset(East : 0, %[[VAL_12:.*]])
// CHECK:           AIE.packetrules(West : 0) {
// CHECK:             AIE.rule(30, 3, %[[VAL_12]])
// CHECK:             AIE.rule(31, 1, %[[VAL_13]])
// CHECK:           }
// CHECK:         }

// CHECK:         %[[VAL_16:.*]] = AIE.tile(5, 2)
// CHECK:         %[[VAL_17:.*]] = AIE.switchbox(%[[VAL_16]]) {
// CHECK:           %[[VAL_18:.*]] = AIE.amsel<0> (0)
// CHECK:           %[[VAL_19:.*]] = AIE.amsel<0> (1)
// CHECK:           %[[VAL_20:.*]] = AIE.masterset(DMA : 0, %[[VAL_19]])
// CHECK:           %[[VAL_21:.*]] = AIE.masterset(East : 0, %[[VAL_18]])
// CHECK:           AIE.packetrules(West : 0) {
// CHECK:             AIE.rule(31, 3, %[[VAL_18]])
// CHECK:             AIE.rule(31, 2, %[[VAL_19]])
// CHECK:           }
// CHECK:         }

// CHECK:         %[[VAL_22:.*]] = AIE.tile(6, 2)
// CHECK:         %[[VAL_23:.*]] = AIE.switchbox(%[[VAL_22]]) {
// CHECK:           %{{.*}} = AIE.amsel<0> (0)
// CHECK:           %[[VAL_25:.*]] = AIE.masterset(DMA : 0, %[[VAL_24:.*]])
// CHECK:           AIE.packetrules(West : 0) {
// CHECK:             AIE.rule(31, 3, %[[VAL_24]])
// CHECK:           }
// CHECK:         }
// CHECK:       }
  %tile22 = AIE.tile(2, 2)
  %tile32 = AIE.tile(3, 2)
  %tile42 = AIE.tile(4, 2)
  %tile52 = AIE.tile(5, 2)
  %tile62 = AIE.tile(6, 2)

  // [2, 2] --> [3, 2] --> [4, 2] --> [5, 2] --> [6, 2]

  AIE.packet_flow(0x0) {
    AIE.packet_source<%tile22, DMA : 0>
    AIE.packet_dest<%tile32, DMA : 0>
  }

  AIE.packet_flow(0x1) {
    AIE.packet_source<%tile22, DMA : 0>
    AIE.packet_dest<%tile42, DMA : 0>
  }

  AIE.packet_flow(0x2) {
    AIE.packet_source<%tile22, DMA : 0>
    AIE.packet_dest<%tile52, DMA : 0>
  }

  AIE.packet_flow(0x3) {
    AIE.packet_source<%tile22, DMA : 0>
    AIE.packet_dest<%tile62, DMA : 0>
  }
 }
}
