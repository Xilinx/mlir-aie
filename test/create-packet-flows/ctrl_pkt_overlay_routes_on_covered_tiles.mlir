//===- ctrl_pkt_overlay_routes_on_covered_tiles.mlir -----------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A control-packet reload configures only the switchboxes of tiles the control
// overlay routes control packets to. In column 1 it covers rows 0 and 1 only,
// so the flow from (0, 3) to (2, 3) crosses column 1 there, not at (1, 3),
// whose switchbox no reload could configure (#3837).

// RUN: aie-opt --pass-pipeline="builtin.module(aie-generate-column-control-overlay{route-shim-to-tile-ctrl=true emit-standalone-overlay=true},aie.device(aie-create-pathfinder-flows))" %s -o %t
// RUN: sed -n '/@design {/,/^  }/p' %t | FileCheck %s
// RUN: aie-opt --pass-pipeline="builtin.module(aie-materialize-runtime-sequences,aie-expand-load-pdi{ctrl-pkt=true})" %t | FileCheck %s --check-prefix=RELOAD

// CHECK:     %[[T03:.+]] = aie.tile(0, 3)
// CHECK-NOT: aie.tile(1, 2)
// CHECK-NOT: aie.tile(1, 3)
// CHECK:     aie.switchbox(%[[T03]]) {
// CHECK-NEXT:  aie.connect<DMA : 0, South : 0>

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
