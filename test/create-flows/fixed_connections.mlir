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
// XFAIL: *

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
