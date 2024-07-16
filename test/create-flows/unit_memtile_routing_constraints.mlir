//===- memtile_routing_constraints.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK:  %tile_2_0 = aie.tile(2, 0)
// CHECK:  %tile_2_1 = aie.tile(2, 1)
// CHECK:  %tile_2_2 = aie.tile(2, 2)
// CHECK:  %tile_2_3 = aie.tile(2, 3)
// CHECK:  %switchbox_2_1 = aie.switchbox(%tile_2_1) {
// CHECK:    aie.connect<North : 0, DMA : 0>
// CHECK:    aie.connect<North : 1, South : 1>
// CHECK:  }
// CHECK:  %switchbox_2_2 = aie.switchbox(%tile_2_2) {
// CHECK:    aie.connect<DMA : 0, South : 0>
// CHECK:    aie.connect<North : 1, South : 1>
// CHECK:  }
// CHECK:  %switchbox_2_0 = aie.switchbox(%tile_2_0) {
// CHECK:    aie.connect<North : 1, South : 2>
// CHECK:  }
// CHECK:  %shim_mux_2_0 = aie.shim_mux(%tile_2_0) {
// CHECK:    aie.connect<North : 2, DMA : 0>
// CHECK:  }
// CHECK:  %switchbox_2_3 = aie.switchbox(%tile_2_3) {
// CHECK:    aie.connect<DMA : 0, South : 1>
// CHECK:  }

module {
  aie.device(xcve2802) {
    %tile_2_0 = aie.tile(2, 0)
    %tile_2_1 = aie.tile(2, 1)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)

    aie.flow(%tile_2_2, DMA : 0, %tile_2_1, DMA : 0)
    aie.flow(%tile_2_3, DMA : 0, %tile_2_0, DMA : 0)
  }
}
