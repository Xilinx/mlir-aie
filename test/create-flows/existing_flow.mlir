//===- unit_existing_flow.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK:    %[[T02:.*]] = aie.tile(0, 2)
// CHECK:    %[[T12:.*]] = aie.tile(1, 2)
// CHECK:    %[[T22:.*]] = aie.tile(2, 2)
// CHECK:    %[[T32:.*]] = aie.tile(3, 2)
// CHECK:    %{{.*}} = aie.switchbox(%[[T02]]) {
// CHECK:      aie.connect<DMA : 0, East : 0>
// CHECK:    }
// CHECK:    %{{.*}} = aie.switchbox(%[[T12]]) {
// CHECK:      aie.connect<West : 0, East : 0>
// CHECK:      aie.connect<DMA : 0, East : [[ID1:[0-9]+]]>
// CHECK:    }
// CHECK:    %{{.*}} = aie.switchbox(%[[T22]]) {
// CHECK:      aie.connect<West : 0, East : 0>
// CHECK:      aie.connect<West : [[ID1]], DMA : 0>
// CHECK:    }
// CHECK:    %{{.*}} = aie.switchbox(%[[T32]]) {
// CHECK:      aie.connect<West : 0, DMA : 0>
// CHECK:    }

module {
  aie.device(xcvc1902) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)

    // aie.flow(%tile_0_2, DMA : 0, %tile_3_2, DMA : 0)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<DMA : 0, East : 0>
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<West : 0, East : 0>
    }
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
      aie.connect<West : 0, East : 0>
    }
    %switchbox_3_2 = aie.switchbox(%tile_3_2) {
      aie.connect<West : 0, DMA : 0>
    }

    aie.flow(%tile_1_2, DMA : 0, %tile_2_2, DMA : 0)
  }
}