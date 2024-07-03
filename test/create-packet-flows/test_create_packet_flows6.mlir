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
// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Fixme: may fail non-deterministically

module @test_create_packet_flows6 {
 aie.device(xcvc1902) {
// CHECK-LABEL: module @test_create_packet_flows6 {
// CHECK:         %[[VAL_0:.*]] = aie.tile(2, 2)
// CHECK:         %[[VAL_1:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:           %{{.*}} = aie.amsel<0> (0)
// CHECK:           %[[VAL_3:.*]] = aie.masterset(East : 0, %[[VAL_2:.*]])
// CHECK:           aie.packet_rules(DMA : 0) {
// CHECK:             aie.rule(28, 0, %[[VAL_2]])
// CHECK:           }
// CHECK:         }

// CHECK:         %[[VAL_4:.*]] = aie.tile(3, 2)
// CHECK:         %[[VAL_5:.*]] = aie.switchbox(%[[VAL_4]]) {
// CHECK:           %{{.*}} = aie.amsel<0> (0)
// CHECK:           %{{.*}} = aie.amsel<0> (1)
// CHECK:           %[[VAL_8:.*]] = aie.masterset(DMA : 0, %[[VAL_7:.*]])
// CHECK:           %[[VAL_9:.*]] = aie.masterset(East : 0, %[[VAL_6:.*]])
// CHECK:           aie.packet_rules(West : 0) {
// CHECK:             aie.rule(28, 0, %[[VAL_6]])
// CHECK:             aie.rule(31, 0, %[[VAL_7]])
// CHECK:           }
// CHECK:         }

// CHECK:         %[[VAL_10:.*]] = aie.tile(4, 2)
// CHECK:         %[[VAL_11:.*]] = aie.switchbox(%[[VAL_10]]) {
// CHECK:           %{{.*}} = aie.amsel<0> (0)
// CHECK:           %{{.*}} = aie.amsel<0> (1)
// CHECK:           %[[VAL_14:.*]] = aie.masterset(DMA : 0, %[[VAL_13:.*]])
// CHECK:           %[[VAL_15:.*]] = aie.masterset(East : 0, %[[VAL_12:.*]])
// CHECK:           aie.packet_rules(West : 0) {
// CHECK:             aie.rule(30, 2, %[[VAL_12]])
// CHECK:             aie.rule(31, 1, %[[VAL_13]])
// CHECK:           }
// CHECK:         }

// CHECK:         %[[VAL_16:.*]] = aie.tile(5, 2)
// CHECK:         %[[VAL_17:.*]] = aie.switchbox(%[[VAL_16]]) {
// CHECK:           %[[VAL_18:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_19:.*]] = aie.amsel<0> (1)
// CHECK:           %[[VAL_20:.*]] = aie.masterset(DMA : 0, %[[VAL_19]])
// CHECK:           %[[VAL_21:.*]] = aie.masterset(East : 0, %[[VAL_18]])
// CHECK:           aie.packet_rules(West : 0) {
// CHECK:             aie.rule(31, 3, %[[VAL_18]])
// CHECK:             aie.rule(31, 2, %[[VAL_19]])
// CHECK:           }
// CHECK:         }

// CHECK:         %[[VAL_22:.*]] = aie.tile(6, 2)
// CHECK:         %[[VAL_23:.*]] = aie.switchbox(%[[VAL_22]]) {
// CHECK:           %{{.*}} = aie.amsel<0> (0)
// CHECK:           %[[VAL_25:.*]] = aie.masterset(DMA : 0, %[[VAL_24:.*]])
// CHECK:           aie.packet_rules(West : 0) {
// CHECK:             aie.rule(31, 3, %[[VAL_24]])
// CHECK:           }
// CHECK:         }
// CHECK:       }
  %tile22 = aie.tile(2, 2)
  %tile32 = aie.tile(3, 2)
  %tile42 = aie.tile(4, 2)
  %tile52 = aie.tile(5, 2)
  %tile62 = aie.tile(6, 2)

  // [2, 2] --> [3, 2] --> [4, 2] --> [5, 2] --> [6, 2]

  aie.packet_flow(0x0) {
    aie.packet_source<%tile22, DMA : 0>
    aie.packet_dest<%tile32, DMA : 0>
  }

  aie.packet_flow(0x1) {
    aie.packet_source<%tile22, DMA : 0>
    aie.packet_dest<%tile42, DMA : 0>
  }

  aie.packet_flow(0x2) {
    aie.packet_source<%tile22, DMA : 0>
    aie.packet_dest<%tile52, DMA : 0>
  }

  aie.packet_flow(0x3) {
    aie.packet_source<%tile22, DMA : 0>
    aie.packet_dest<%tile62, DMA : 0>
  }
 }
}
