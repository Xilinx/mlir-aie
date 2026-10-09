//===- ctrl_pkt_overlay_reserved_amsel_fails.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Alone, as in @ctrl_pkt_overlay, the control overlay takes amsel<5> (2) at
// (0, 0), but the design's own switchbox there already uses it. A
// control-packet reload installs the overlay first, so it cannot take another
// (#3837).

// RUN: not aie-opt --pass-pipeline="builtin.module(aie-generate-column-control-overlay{route-shim-to-tile-ctrl=true emit-standalone-overlay=true},aie.device(aie-create-pathfinder-flows))" %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: Unable to find a legal routing: the control overlay keeps the packet rules it takes alone, as in @ctrl_pkt_overlay, since a control-packet reload (has_ctrl_pkt_overlay) installs them first, but at tile (0, 0), packet flow 26 into DMA:0 takes amsel<5> (2) alone, and the design's own switchbox takes it.

module {
  aie.device(npu2) @design {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %sb = aie.switchbox(%t00) {
      %a = aie.amsel<5> (2)
      %m = aie.masterset(East : 0, %a)
      aie.packet_rules(North : 0) {
        aie.rule(31, 9, %a)
      }
    }
    aie.flow(%t01, DMA : 0, %t02, DMA : 0)
  }
}
