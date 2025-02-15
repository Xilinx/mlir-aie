//===- test_congestion1.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s -o %t.opt
// RUN: FileCheck %s --check-prefix=CHECK1 < %t.opt
// RUN: aie-translate --aie-flows-to-json %t.opt | FileCheck %s --check-prefix=CHECK2

// CHECK1:    %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK1:    %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK1:    %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK1:    %[[TILE_0_3:.*]] = aie.tile(0, 3)
// CHECK1:    %[[TILE_0_4:.*]] = aie.tile(0, 4)
// CHECK1:    %[[TILE_0_5:.*]] = aie.tile(0, 5)
// CHECK1:    %[[TILE_1_0:.*]] = aie.tile(1, 0)
// CHECK1:    aie.packet_flow(0) {
// CHECK1:      aie.packet_source<%[[TILE_0_5]], DMA : 1>
// CHECK1:      aie.packet_dest<%[[TILE_0_1]], DMA : 4>
// CHECK1:    }
// CHECK1:    aie.flow(%[[TILE_0_1]], DMA : 0, %[[TILE_0_0]], DMA : 0)
// CHECK1:    aie.flow(%[[TILE_0_2]], DMA : 0, %[[TILE_0_1]], DMA : 0)
// CHECK1:    aie.flow(%[[TILE_0_3]], DMA : 0, %[[TILE_0_1]], DMA : 1)
// CHECK1:    aie.flow(%[[TILE_0_4]], DMA : 0, %[[TILE_0_1]], DMA : 2)
// CHECK1:    aie.flow(%[[TILE_0_5]], DMA : 0, %[[TILE_0_1]], DMA : 3)

// CHECK2: "total_path_length": 19

module {
 aie.device(npu1_2col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  %tile_0_2 = aie.tile(0, 2)
  %tile_0_3 = aie.tile(0, 3)
  %tile_0_4 = aie.tile(0, 4)
  %tile_0_5 = aie.tile(0, 5)
  %tile_1_0 = aie.tile(1, 0)
  aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 0)
  aie.flow(%tile_0_3, DMA : 0, %tile_0_1, DMA : 1)
  aie.flow(%tile_0_4, DMA : 0, %tile_0_1, DMA : 2)
  aie.flow(%tile_0_5, DMA : 0, %tile_0_1, DMA : 3)
  aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)
  aie.packet_flow(0x0) {
    aie.packet_source<%tile_0_5, DMA : 1>
    aie.packet_dest<%tile_0_1, DMA : 4>
  }
 }
}
