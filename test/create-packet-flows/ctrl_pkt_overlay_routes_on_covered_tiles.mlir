//===- ctrl_pkt_overlay_routes_on_covered_tiles.mlir -----------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A control-packet reload configures only the switchboxes of tiles the control
// overlay routes control packets to. It covers every row of each column from
// the leftmost to the rightmost one the design occupies, so the flow from
// (0, 3) to (2, 3) may cross column 1 at (1, 3), and the configuration's
// control packets route it there (#3837).

// RUN: aie-opt --pass-pipeline="builtin.module(aie-generate-column-control-overlay{route-shim-to-tile-ctrl=true emit-standalone-overlay=true},aie.device(aie-create-pathfinder-flows))" %s -o %t
// RUN: sed -n '/@design {/,/^  }/p' %t | FileCheck %s
// RUN: aie-opt --pass-pipeline="builtin.module(aie-materialize-runtime-sequences,aie-expand-load-pdi{ctrl-pkt=true})" %t | FileCheck %s --check-prefix=RELOAD

// CHECK:      %[[T13:.+]] = aie.tile(1, 3)
// CHECK:      aie.switchbox(%[[T13]]) {
// CHECK-NEXT:   aie.connect<West : {{[0-9]+}}, East : {{[0-9]+}}>
// CHECK:        aie.masterset(TileControl : 0, %{{.+}}) {aie.is_ctrl_pkt_overlay, keep_pkt_header = true}

// RELOAD: aiex.npu.load_pdi {device_ref = @ctrl_pkt_overlay

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%arg0 : memref<16xi32>) {
      aiex.configure @design {
      }
    }
  }
  aie.device(npu2) @design {
    %t03 = aie.tile(0, 3)
    %t11 = aie.tile(1, 1)
    %t23 = aie.tile(2, 3)
    aie.flow(%t03, DMA : 0, %t23, DMA : 0)
  }
}
