// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-create-control-overlay='adaptive=false' --aie-create-pathfinder-flows %s -o %t
// RUN: aie-opt --aie-import-overlay %t | diff %t -
// RUN: FileCheck %s < %t

// CHECK-LABEL: aie.device({{.*}}) @design
// CHECK: aie.switchbox
// CHECK: aie.is_ctrl_overlay
// CHECK: overlay = @ctrl_pkt_overlay
// CHECK-LABEL: aie.device({{.*}}) @ctrl_pkt_overlay
// CHECK-NOT: aie.packet_flow
// CHECK: aie.masterset(TileControl : 0,
// CHECK-SAME: aie.is_ctrl_overlay

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%arg0 : memref<16xi32>) {
      aiex.configure @design {
      }
    }
  }
  aie.device(npu2) @design {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t10 = aie.tile(1, 0)
    %t11 = aie.tile(1, 1)
    %t12 = aie.tile(1, 2)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    aie.flow(%t10, DMA : 1, %t01, DMA : 0)
    aie.flow(%t01, DMA : 0, %t10, DMA : 0)
    aie.flow(%t01, DMA : 1, %t02, DMA : 0)
    aie.flow(%t02, DMA : 0, %t01, DMA : 1)
    aie.flow(%t01, DMA : 2, %t03, DMA : 0)
    aie.flow(%t03, DMA : 0, %t01, DMA : 2)
    aie.flow(%t01, DMA : 3, %t04, DMA : 0)
    aie.flow(%t04, DMA : 0, %t01, DMA : 3)
    aie.flow(%t01, DMA : 4, %t05, DMA : 0)
    aie.flow(%t05, DMA : 0, %t01, DMA : 4)
    aie.flow(%t01, DMA : 5, %t02, DMA : 1)
    aie.packet_flow(15) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t00, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(3) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t11, DMA : 0>
    }
  }
}
