//===- flow_test_1.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --split-input-file %s | FileCheck %s

// CHECK: %tile_2_0 = AIE.tile(2, 0)
// CHECK: %tile_3_0 = AIE.tile(3, 0)
// CHECK: %tile_6_0 = AIE.tile(6, 0)
// CHECK: %tile_7_0 = AIE.tile(7, 0)
// CHECK: %switchbox_2_0 = AIE.switchbox(%tile_2_0) {
// CHECK:   AIE.connect<North : 0, South : 2>
// CHECK:   AIE.connect<North : 1, South : 3>
// CHECK: }
// CHECK: %switchbox_3_0 = AIE.switchbox(%tile_3_0) {
// CHECK:   AIE.connect<North : 0, South : 2>
// CHECK:   AIE.connect<North : 1, South : 3>
// CHECK: }
// CHECK: %switchbox_6_0 = AIE.switchbox(%tile_6_0) {
// CHECK:   AIE.connect<North : 0, South : 2>
// CHECK:   AIE.connect<North : 1, South : 3>
// CHECK: }
// CHECK: %switchbox_7_0 = AIE.switchbox(%tile_7_0) {
// CHECK:   AIE.connect<North : 0, South : 2>
// CHECK:   AIE.connect<North : 1, South : 3>
// CHECK: }
// CHECK: AIE.wire(%switchbox_2_0 : East, %switchbox_3_0 : West)
// CHECK: AIE.wire(%switchbox_6_0 : East, %switchbox_7_0 : West)

module {
  AIE.device(xcvc1902) {
    %tile_2_0 = AIE.tile(2, 0)
    %tile_3_0 = AIE.tile(3, 0)
    %tile_6_0 = AIE.tile(6, 0)
    %tile_7_0 = AIE.tile(7, 0)

    %switchbox_2_0 = AIE.switchbox(%tile_2_0) {
      AIE.connect<North : 0, South : 2>
      AIE.connect<North : 1, South : 3>
    }
    %switchbox_3_0 = AIE.switchbox(%tile_3_0) {
      AIE.connect<North : 0, South : 2>
      AIE.connect<North : 1, South : 3>
    }
    %switchbox_6_0 = AIE.switchbox(%tile_6_0) {
      AIE.connect<North : 0, South : 2>
      AIE.connect<North : 1, South : 3>
    }
    %switchbox_7_0 = AIE.switchbox(%tile_7_0) {
      AIE.connect<North : 0, South : 2>
      AIE.connect<North : 1, South : 3>
    }
  }
}

// -----

// CHECK: %tile_0_3 = AIE.tile(0, 3)
// CHECK: %tile_1_4 = AIE.tile(1, 4)
// CHECK: %tile_3_3 = AIE.tile(3, 3)
// CHECK: %tile_4_2 = AIE.tile(4, 2)
// CHECK: %tile_5_3 = AIE.tile(5, 3)
// CHECK: %tile_6_3 = AIE.tile(6, 3)
// CHECK: %tile_7_4 = AIE.tile(7, 4)
// CHECK: %tile_9_2 = AIE.tile(9, 2)
// CHECK: %tile_10_2 = AIE.tile(10, 2)
// CHECK: %tile_11_3 = AIE.tile(11, 3)
// CHECK: %switchbox_0_3 = AIE.switchbox(%tile_0_3) {
// CHECK:   AIE.connect<South : 0, DMA : 0>
// CHECK:   AIE.connect<East : 0, DMA : 1>
// CHECK: }
// CHECK: %switchbox_1_4 = AIE.switchbox(%tile_1_4) {
// CHECK:   AIE.connect<East : 0, DMA : 0>
// CHECK: }
// CHECK: %switchbox_3_3 = AIE.switchbox(%tile_3_3) {
// CHECK:   AIE.connect<South : 0, DMA : 0>
// CHECK: }
// CHECK: %switchbox_4_2 = AIE.switchbox(%tile_4_2) {
// CHECK:   AIE.connect<South : 0, DMA : 0>
// CHECK: }
// CHECK: %switchbox_5_3 = AIE.switchbox(%tile_5_3) {
// CHECK:   AIE.connect<South : 0, DMA : 0>
// CHECK: }
// CHECK: %switchbox_6_3 = AIE.switchbox(%tile_6_3) {
// CHECK:   AIE.connect<South : 0, DMA : 0>
// CHECK:   AIE.connect<South : 1, DMA : 1>
// CHECK: }
// CHECK: %switchbox_7_4 = AIE.switchbox(%tile_7_4) {
// CHECK:   AIE.connect<South : 0, DMA : 0>
// CHECK:   AIE.connect<South : 1, DMA : 1>
// CHECK: }
// CHECK: %switchbox_9_2 = AIE.switchbox(%tile_9_2) {
// CHECK:   AIE.connect<South : 0, DMA : 0>
// CHECK: }
// CHECK: %switchbox_10_2 = AIE.switchbox(%tile_10_2) {
// CHECK:   AIE.connect<South : 0, DMA : 0>
// CHECK: }
// CHECK: %switchbox_11_3 = AIE.switchbox(%tile_11_3) {
// CHECK:   AIE.connect<South : 0, DMA : 0>
// CHECK:   AIE.connect<South : 1, DMA : 1>
// CHECK: }
// CHECK: AIE.wire(%tile_0_3 : Core, %switchbox_0_3 : Core)
// CHECK: AIE.wire(%tile_0_3 : DMA, %switchbox_0_3 : DMA)
// CHECK: AIE.wire(%tile_1_4 : Core, %switchbox_1_4 : Core)
// CHECK: AIE.wire(%tile_1_4 : DMA, %switchbox_1_4 : DMA)
// CHECK: AIE.wire(%tile_3_3 : Core, %switchbox_3_3 : Core)
// CHECK: AIE.wire(%tile_3_3 : DMA, %switchbox_3_3 : DMA)
// CHECK: AIE.wire(%tile_4_2 : Core, %switchbox_4_2 : Core)
// CHECK: AIE.wire(%tile_4_2 : DMA, %switchbox_4_2 : DMA)
// CHECK: AIE.wire(%tile_5_3 : Core, %switchbox_5_3 : Core)
// CHECK: AIE.wire(%tile_5_3 : DMA, %switchbox_5_3 : DMA)
// CHECK: AIE.wire(%switchbox_5_3 : East, %switchbox_6_3 : West)
// CHECK: AIE.wire(%tile_6_3 : Core, %switchbox_6_3 : Core)
// CHECK: AIE.wire(%tile_6_3 : DMA, %switchbox_6_3 : DMA)
// CHECK: AIE.wire(%tile_7_4 : Core, %switchbox_7_4 : Core)
// CHECK: AIE.wire(%tile_7_4 : DMA, %switchbox_7_4 : DMA)
// CHECK: AIE.wire(%tile_9_2 : Core, %switchbox_9_2 : Core)
// CHECK: AIE.wire(%tile_9_2 : DMA, %switchbox_9_2 : DMA)
// CHECK: AIE.wire(%switchbox_9_2 : East, %switchbox_10_2 : West)
// CHECK: AIE.wire(%tile_10_2 : Core, %switchbox_10_2 : Core)
// CHECK: AIE.wire(%tile_10_2 : DMA, %switchbox_10_2 : DMA)
// CHECK: AIE.wire(%tile_11_3 : Core, %switchbox_11_3 : Core)
// CHECK: AIE.wire(%tile_11_3 : DMA, %switchbox_11_3 : DMA)

module {
  AIE.device(xcvc1902) {
    %tile_0_3 = AIE.tile(0, 3)
    %tile_1_4 = AIE.tile(1, 4)
    %tile_3_3 = AIE.tile(3, 3)
    %tile_4_2 = AIE.tile(4, 2)
    %tile_5_3 = AIE.tile(5, 3)
    %tile_6_3 = AIE.tile(6, 3)
    %tile_7_4 = AIE.tile(7, 4)
    %tile_9_2 = AIE.tile(9, 2)
    %tile_10_2 = AIE.tile(10, 2)
    %tile_11_3 = AIE.tile(11, 3)

    %switchbox_0_3 = AIE.switchbox(%tile_0_3) {
      AIE.connect<South : 0, DMA : 0>
      AIE.connect<East : 0, DMA : 1>
    }
    %switchbox_1_4 = AIE.switchbox(%tile_1_4) {
      AIE.connect<East : 0, DMA : 0>
    }
    %switchbox_3_3 = AIE.switchbox(%tile_3_3) {
      AIE.connect<South : 0, DMA : 0>
    }
    %switchbox_4_2 = AIE.switchbox(%tile_4_2) {
      AIE.connect<South : 0, DMA : 0>
    }
    %switchbox_5_3 = AIE.switchbox(%tile_5_3) {
      AIE.connect<South : 0, DMA : 0>
    }
    %switchbox_6_3 = AIE.switchbox(%tile_6_3) {
      AIE.connect<South : 0, DMA : 0>
      AIE.connect<South : 1, DMA : 1>
    }
    %switchbox_7_4 = AIE.switchbox(%tile_7_4) {
      AIE.connect<South : 0, DMA : 0>
      AIE.connect<South : 1, DMA : 1>
    }
    %switchbox_9_2 = AIE.switchbox(%tile_9_2) {
      AIE.connect<South : 0, DMA : 0>
    }
    %switchbox_10_2 = AIE.switchbox(%tile_10_2) {
      AIE.connect<South : 0, DMA : 0>
    }
    %switchbox_11_3 = AIE.switchbox(%tile_11_3) {
      AIE.connect<South : 0, DMA : 0>
      AIE.connect<South : 1, DMA : 1>
    }
  }
}

// -----

// CHECK: %tile_2_5 = AIE.tile(2, 5)
// CHECK: %tile_3_1 = AIE.tile(3, 1)
// CHECK: %tile_6_6 = AIE.tile(6, 6)
// CHECK: %tile_7_3 = AIE.tile(7, 3)
// CHECK: %tile_12_5 = AIE.tile(12, 5)
// CHECK: %tile_13_3 = AIE.tile(13, 3)
// CHECK: %switchbox_2_5 = AIE.switchbox(%tile_2_5) {
// CHECK:   AIE.connect<South : 0, Core : 0>
// CHECK:   AIE.connect<DMA : 0, East : 0>
// CHECK: }
// CHECK: %switchbox_3_1 = AIE.switchbox(%tile_3_1) {
// CHECK:   AIE.connect<South : 0, DMA : 0>
// CHECK:   AIE.connect<Core : 0, North : 0>
// CHECK: }
// CHECK: %switchbox_6_6 = AIE.switchbox(%tile_6_6) {
// CHECK:   AIE.connect<East : 0, Core : 0>
// CHECK:   AIE.connect<DMA : 0, West : 0>
// CHECK: }
// CHECK: %switchbox_7_3 = AIE.switchbox(%tile_7_3) {
// CHECK:   AIE.connect<East : 0, DMA : 0>
// CHECK:   AIE.connect<Core : 0, North : 0>
// CHECK: }
// CHECK: %switchbox_12_5 = AIE.switchbox(%tile_12_5) {
// CHECK:   AIE.connect<East : 0, Core : 0>
// CHECK:   AIE.connect<DMA : 0, East : 0>
// CHECK: }
// CHECK: %switchbox_13_3 = AIE.switchbox(%tile_13_3) {
// CHECK:   AIE.connect<South : 0, DMA : 0>
// CHECK:   AIE.connect<Core : 0, North : 0>
// CHECK: }
// CHECK: AIE.wire(%tile_2_5 : Core, %switchbox_2_5 : Core)
// CHECK: AIE.wire(%tile_2_5 : DMA, %switchbox_2_5 : DMA)
// CHECK: AIE.wire(%tile_3_1 : Core, %switchbox_3_1 : Core)
// CHECK: AIE.wire(%tile_3_1 : DMA, %switchbox_3_1 : DMA)
// CHECK: AIE.wire(%tile_6_6 : Core, %switchbox_6_6 : Core)
// CHECK: AIE.wire(%tile_6_6 : DMA, %switchbox_6_6 : DMA)
// CHECK: AIE.wire(%tile_7_3 : Core, %switchbox_7_3 : Core)
// CHECK: AIE.wire(%tile_7_3 : DMA, %switchbox_7_3 : DMA)
// CHECK: AIE.wire(%tile_12_5 : Core, %switchbox_12_5 : Core)
// CHECK: AIE.wire(%tile_12_5 : DMA, %switchbox_12_5 : DMA)
// CHECK: AIE.wire(%tile_13_3 : Core, %switchbox_13_3 : Core)
// CHECK: AIE.wire(%tile_13_3 : DMA, %switchbox_13_3 : DMA)

module {
  AIE.device(xcvc1902) {
    %tile_2_5 = AIE.tile(2, 5)
    %tile_3_1 = AIE.tile(3, 1)
    %tile_6_6 = AIE.tile(6, 6)
    %tile_7_3 = AIE.tile(7, 3)
    %tile_12_5 = AIE.tile(12, 5)
    %tile_13_3 = AIE.tile(13, 3)

    %switchbox_2_5 = AIE.switchbox(%tile_2_5) {
      AIE.connect<South : 0, Core : 0>
      AIE.connect<DMA : 0, East : 0>
    }
    %switchbox_3_1 = AIE.switchbox(%tile_3_1) {
      AIE.connect<South : 0, DMA : 0>
      AIE.connect<Core : 0, North : 0>
    }
    %switchbox_6_6 = AIE.switchbox(%tile_6_6) {
      AIE.connect<East : 0, Core : 0>
      AIE.connect<DMA : 0, West : 0>
    }
    %switchbox_7_3 = AIE.switchbox(%tile_7_3) {
      AIE.connect<East : 0, DMA : 0>
      AIE.connect<Core : 0, North : 0>
    }
    %switchbox_12_5 = AIE.switchbox(%tile_12_5) {
      AIE.connect<East : 0, Core : 0>
      AIE.connect<DMA : 0, East : 0>
    }
    %switchbox_13_3 = AIE.switchbox(%tile_13_3) {
      AIE.connect<South : 0, DMA : 0>
      AIE.connect<Core : 0, North : 0>
    }
  }
}
