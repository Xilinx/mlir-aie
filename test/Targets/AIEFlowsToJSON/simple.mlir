//===- simple.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-flows-to-json %s | FileCheck %s
// CHECK{LITERAL}: "route0": [ [[0, 0], ["North"]], [[0, 1], ["DMA"]], [] ],
// CHECK{LITERAL}: "route1": [ [[0, 2], ["South"]], [[0, 1], ["South"]], [[0, 0], ["South"]], [] ],

module @aie_module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 3, %0)
      aie.packet_rules(DMA : 1) {
        aie.rule(31, 1, %0)
      }
    }
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, DMA : 1>
      aie.packet_dest<%tile_0_0, DMA : 1>
    }
    %switchbox_0_0 = aie.switchbox(%tile_0_0) {
      aie.connect<South : 3, North : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 3, %0)
      aie.packet_rules(North : 3) {
        aie.rule(31, 1, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%tile_0_0) {
      aie.connect<North : 3, DMA : 1>
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_0_1 = aie.switchbox(%tile_0_1) {
      aie.connect<South : 0, DMA : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 3, %0)
      aie.packet_rules(North : 3) {
        aie.rule(31, 1, %0)
      }
    }
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
  }
}