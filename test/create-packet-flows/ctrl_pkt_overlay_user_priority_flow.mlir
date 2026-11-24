// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-generate-column-control-overlay="configuration=true adaptive=false" --aie-import-overlay --aie-annotate-shim-sharing --aie-create-pathfinder-flows %s -o %t
// RUN: aie-opt --aie-import-overlay %t | diff %t -
// RUN: FileCheck %s < %t

// CHECK-LABEL: aie.device({{.*}}) @design
// CHECK: aie.switchbox
// CHECK: aie.is_ctrl_pkt_overlay
// CHECK: overlay = @ctrl_pkt_overlay
// CHECK-LABEL: aie.device({{.*}}) @ctrl_pkt_overlay
// CHECK-NOT: aie.packet_flow
// CHECK: aie.masterset(TileControl : 0,
// CHECK-SAME: aie.is_ctrl_pkt_overlay

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%arg0 : memref<16xi32>) {
      aiex.configure @design {
      }
    }
  }
  aie.device(npu2) @design {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.packet_flow(5) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t02, DMA : 0>
    } {priority_route = true}
  }
}
