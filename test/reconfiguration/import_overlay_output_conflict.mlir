// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: not aie-opt --aie-import-overlay --aie-create-pathfinder-flows %s -o /dev/null 2>&1 | FileCheck %s
// CHECK: error: Unable to find a legal routing

module {
  aie.device(npu2_1col) @application {
    %tile = aie.tile(0, 2)
    aie.packet_flow(0) {
      aie.packet_source<%tile, DMA : 0>
      aie.packet_dest<%tile, TileControl : 0>
    }
  } {overlay = @transport}
  aie.device(npu2_1col) @transport {
    %tile = aie.tile(0, 2)
    aie.switchbox(%tile) {
      %a = aie.amsel<5>(3) {aie.is_ctrl_overlay}
      aie.masterset(TileControl : 0, %a) {aie.is_ctrl_overlay}
      aie.packet_rules(South : 0) {
        aie.rule(31, 15, %a) {aie.is_ctrl_overlay}
      }
    }
  } {aie.is_ctrl_overlay}
}
