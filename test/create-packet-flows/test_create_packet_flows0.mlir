//===- test_create_packet_flows0.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// one-to-many, single arbiter
module @test_create_packet_flows0 {
 aie.device(xcvc1902) {
// CHECK-LABEL:   module @test_create_packet_flows0 {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_1:.*]] = aie.switchbox(%[[VAL_0]]) {
// The actual indices used for the amsel arguments is unimportant.
// CHECK:           %[[VAL_6:.*]] = aie.amsel<0> (0)
// CHECK:           %[[VAL_7:.*]] = aie.amsel<1> (0)
// CHECK:           %[[VAL_4:.*]] = aie.masterset(Core : 0, %[[VAL_2:.*]])
// CHECK:           %[[VAL_5:.*]] = aie.masterset(Core : 1, %[[VAL_3:.*]])
// CHECK:           aie.packet_rules(West : 0) {
// CHECK:             aie.rule(31, 1, %[[VAL_3]])
// CHECK:             aie.rule(31, 0, %[[VAL_2]])
// CHECK:           }
// CHECK:         }
// CHECK:       }
  %t11 = aie.tile(1, 1)

  aie.packet_flow(0x0) {
    aie.packet_source<%t11, West : 0>
    aie.packet_dest<%t11, Core : 0>
  }

  aie.packet_flow(0x1) {
    aie.packet_source<%t11, West : 0>
    aie.packet_dest<%t11, Core : 1>
  }
 }
}
