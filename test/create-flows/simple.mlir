//===- simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = AIE.tile(0, 1)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(1, 2)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(0, 2)
// CHECK:           %[[VAL_3:.*]] = AIE.switchbox(%[[VAL_0]]) {
// CHECK:             AIE.connect<DMA : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_4:.*]] = AIE.switchbox(%[[VAL_2]]) {
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_5:.*]] = AIE.switchbox(%[[VAL_1]]) {
// CHECK:             AIE.connect<West : 0, Core : 1>
// CHECK:           }
// CHECK:           AIE.packet_flow(16) {
// CHECK:             AIE.packet_source<%[[VAL_0]], Core : 0>
// CHECK:             AIE.packet_dest<%[[VAL_1]], Core : 0>
// CHECK:             AIE.packet_dest<%[[VAL_2]], DMA : 1>
// CHECK:           }
// CHECK:           AIE.wire(%[[VAL_0]] : Core, %[[VAL_3]] : Core)
// CHECK:           AIE.wire(%[[VAL_0]] : DMA, %[[VAL_3]] : DMA)
// CHECK:           AIE.wire(%[[VAL_2]] : Core, %[[VAL_4]] : Core)
// CHECK:           AIE.wire(%[[VAL_2]] : DMA, %[[VAL_4]] : DMA)
// CHECK:           AIE.wire(%[[VAL_3]] : North, %[[VAL_4]] : South)
// CHECK:           AIE.wire(%[[VAL_4]] : East, %[[VAL_5]] : West)
// CHECK:           AIE.wire(%[[VAL_1]] : Core, %[[VAL_5]] : Core)
// CHECK:           AIE.wire(%[[VAL_1]] : DMA, %[[VAL_5]] : DMA)
// CHECK:         }


module {
  AIE.device(xcvc1902) {
    %01 = AIE.tile(0, 1)
    %12 = AIE.tile(1, 2)
    %02 = AIE.tile(0, 2)
    AIE.flow(%01, DMA : 0, %12, Core : 1)
    AIE.packet_flow(0x10) {
      AIE.packet_source < %01, Core : 0>
      AIE.packet_dest < %12, Core : 0>
      AIE.packet_dest < %02, DMA : 1>
    }
  }
}
