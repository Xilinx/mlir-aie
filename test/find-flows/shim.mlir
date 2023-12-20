//===- shim.mlir -----------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-find-flows -split-input-file %s | FileCheck %s

// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 1)
// CHECK:           %[[VAL_1:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_6:.*]] = aie.shim_dma(%[[VAL_1]])
// CHECK:           aie.flow(%[[VAL_0]], Core : 0, %[[VAL_6]], DMA : 0)
module {
  aie.device(xcvc1902) {
    %t21 = aie.tile(2, 1)
    %t20 = aie.tile(2, 0)
    %c21 = aie.core(%t21)  {
      aie.end
    }
    %s21 = aie.switchbox(%t21)  {
      aie.connect<Core : 0, South : 0>
    }
    %s20 = aie.switchbox(%t20)  {
      aie.connect<North : 0, South : 2>
    }
    %mux = aie.shim_mux(%t20)  {
      aie.connect<North : 2, DMA : 0>
    }
    %dma = aie.shim_dma(%t20)  {
      aie.end
    }
    aie.wire(%s21 : South, %s20 : North)
    aie.wire(%s20 : South, %mux : North)
    aie.wire(%mux : DMA, %dma : DMA)
    aie.wire(%mux : South, %t20 : DMA)
    aie.wire(%s21 : Core, %c21 : Core)
    aie.wire(%s21 : Core, %t21 : Core)
  }
}

// -----

// CHECK:  %tile_2_1 = aie.tile(2, 1)
// CHECK:  %tile_2_0 = aie.tile(2, 0)
// CHECK:  %core_2_1 = aie.core(%tile_2_1) {
// CHECK:    aie.end
// CHECK:  }
// CHECK:  %switchbox_2_1 = aie.switchbox(%tile_2_1) {
// CHECK:    aie.connect<Core : 0, South : 0>
// CHECK:  }
// CHECK:  %switchbox_2_0 = aie.switchbox(%tile_2_0) {
// CHECK:    aie.connect<North : 0, South : 2>
// CHECK:  }
// CHECK:  %shim_mux_2_0 = aie.shim_mux(%tile_2_0) {
// CHECK:    aie.connect<North : 2, DMA : 0>
// CHECK:  }
// CHECK:  %shim_dma_2_0 = aie.shim_dma(%tile_2_0) {
// CHECK:    aie.end
// CHECK:  }
// CHECK:  aie.wire(%switchbox_2_1 : South, %switchbox_2_0 : North)
// CHECK:  aie.wire(%switchbox_2_0 : South, %shim_mux_2_0 : North)
// CHECK:  aie.wire(%shim_mux_2_0 : DMA, %shim_dma_2_0 : DMA)
// CHECK:  aie.wire(%shim_mux_2_0 : South, %tile_2_0 : DMA)
// CHECK:  aie.wire(%switchbox_2_1 : Core, %core_2_1 : Core)
// CHECK:  aie.wire(%switchbox_2_1 : Core, %tile_2_1 : Core)
// CHECK:  aie.flow(%tile_2_1, Core : 0, %shim_dma_2_0, DMA : 0)

module {
  aie.device(xcvc1902) {
    %t21 = aie.tile(2, 1)
    %t20 = aie.tile(2, 0)
    %c21 = aie.core(%t21)  {
      aie.end
    }
    %s21 = aie.switchbox(%t21)  {
      aie.connect<Core : 0, South : 0>
    }
    %s20 = aie.switchbox(%t20)  {
      aie.connect<North : 0, South : 2>
    }
    %mux = aie.shim_mux(%t20)  {
      aie.connect<North : 2, DMA : 0>
    }
    %dma = aie.shim_dma(%t20)  {
      aie.end
    }
    aie.wire(%s21 : South, %s20 : North)
    aie.wire(%s20 : South, %mux : North)
    aie.wire(%mux : DMA, %dma : DMA)
    aie.wire(%mux : South, %t20 : DMA)
    aie.wire(%s21 : Core, %c21 : Core)
    aie.wire(%s21 : Core, %t21 : Core)
  }
}