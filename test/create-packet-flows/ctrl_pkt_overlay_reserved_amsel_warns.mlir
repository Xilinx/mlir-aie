//===- ctrl_pkt_overlay_reserved_amsel_warns.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Alone, as in @ctrl_pkt_overlay, the control overlay takes amsel<5> (2) at
// (0, 0), but the design's own switchbox there already uses it, so the overlay
// takes another amsel in the design and a control-packet reload would not keep
// it (#3837).

// RUN: aie-opt --pass-pipeline="builtin.module(aie-generate-column-control-overlay{route-shim-to-tile-ctrl=true emit-standalone-overlay=true},aie.device(aie-create-pathfinder-flows))" %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: warning: the prioritized flows (the control overlay) take other packet rules at (0, 0) South:3 than they take alone, as in @ctrl_pkt_overlay, so a control-packet reload would not keep them.

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
