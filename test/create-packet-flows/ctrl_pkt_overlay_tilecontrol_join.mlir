// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-generate-column-control-overlay="configuration=true adaptive=false" --aie-import-overlay --aie-annotate-shim-sharing --aie-create-pathfinder-flows %s | FileCheck %s
// RUN: aie-opt --aie-generate-column-control-overlay="configuration=true adaptive=true" --aie-import-overlay --aie-annotate-shim-sharing --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL: aie.device(npu2) @design
// CHECK: aie.switchbox(%shim_noc_tile_0_0)
// CHECK: aie.masterset(TileControl : 0, %[[TC:[0-9]+]]) {aie.is_ctrl_pkt_overlay
// CHECK-NOT: aie.masterset(TileControl
// CHECK: aie.packet_rules(East : {{[0-9]+}}) {
// CHECK-NEXT: aie.rule(31, 29, %[[TC]])
// CHECK-NEXT: }

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%arg0 : memref<16xi32>) {
      aiex.configure @design {
      }
    }
  }
  aie.device(npu2) @design {
    %t00 = aie.tile(0, 0)
    %t10 = aie.tile(1, 0)
    aie.packet_flow(29) {
      aie.packet_source<%t10, DMA : 1>
      aie.packet_dest<%t00, TileControl : 0>
    } {keep_pkt_header = true}
  }
}
