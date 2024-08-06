//===- keep_packet_flow_op.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows='keep-flow-op=true' %s | FileCheck %s

// CHECK-LABEL: module @aie_module {
// CHECK:   %[[VAL_0:.*]] = aie.tile(7, 0)
// CHECK:   %[[VAL_1:.*]] = aie.shim_mux(%[[VAL_0:.*]])  {
// CHECK:     aie.connect<North : 3, DMA : 1>
// CHECK:   }
// CHECK:   %[[VAL_2:.*]] = aie.switchbox(%[[VAL_0:.*]]) {
// CHECK:     %[[VAL_3:.*]] = aie.amsel<0> (0)
// CHECK:     %[[VAL_4:.*]] = aie.masterset(South : 3, %[[VAL_3:.*]])
// CHECK:     aie.packet_rules(North : 0) {
// CHECK:       aie.rule(31, 10, %[[VAL_3:.*]])
// CHECK:     }
// CHECK:   }
// CHECK:   %[[VAL_5:.*]] = aie.tile(7, 1)
// CHECK:   %[[VAL_6:.*]] = aie.switchbox(%[[VAL_5:.*]]) {
// CHECK:     %[[VAL_7:.*]] = aie.amsel<0> (0)
// CHECK:     %[[VAL_8:.*]] = aie.masterset(South : 0, %[[VAL_6:.*]])
// CHECK:     aie.packet_rules(DMA : 0) {
// CHECK:       aie.rule(31, 10, %[[VAL_7:.*]])
// CHECK:     }
// CHECK:   }
// CHECK:   aie.packet_flow(10) {
// CHECK:     aie.packet_source<%[[VAL_5:.*]], DMA : 0>
// CHECK:     aie.packet_dest<%[[VAL_0:.*]], DMA : 1>
// CHECK:   }

module @aie_module  {
 aie.device(xcvc1902) {
  %t70 = aie.tile(7, 0)
  %t71 = aie.tile(7, 1)

  aie.packet_flow(0xA) {
    aie.packet_source<%t71, DMA : 0>
    aie.packet_dest<%t70, DMA : 1>
  }
 }
}
