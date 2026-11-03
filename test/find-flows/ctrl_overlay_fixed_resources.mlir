// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-find-flows %s | FileCheck %s --check-prefixes=CHECK,FIND
// RUN: aie-opt --aie-find-flows="emit-vias=true" %s | FileCheck %s --check-prefixes=CHECK,FIND
// RUN: aie-opt --aie-find-flows --aie-create-pathfinder-flows %s | FileCheck %s --check-prefixes=CHECK,ROUTE

// CHECK-LABEL: aie.device(npu2_1col) @fixed
// CHECK: aie.switchbox(%[[TILE:tile_0_2]]) {
// CHECK-NEXT: aie.connect<DMA : 0, TileControl : 0> {aie.is_ctrl_pkt_overlay}
// CHECK-NEXT: %{{.*}} = aie.amsel<0> (0) {aie.is_ctrl_pkt_overlay}
// FIND-NEXT: }
// FIND-NEXT: aie.flow(%[[TILE]], DMA : 1, %[[TILE]], Core : 0)
// FIND-NEXT: }
// ROUTE-NEXT: aie.connect<DMA : 1, Core : 0>
// ROUTE-NEXT: }
// ROUTE-NOT: aie.flow

module {
  aie.device(npu2_1col) @fixed {
    %tile = aie.tile(0, 2)
    aie.switchbox(%tile) {
      aie.connect<DMA : 0, TileControl : 0> {aie.is_ctrl_pkt_overlay}
      %reserved = aie.amsel<0> (0) {aie.is_ctrl_pkt_overlay}
      %unused = aie.amsel<1> (0)
      aie.connect<DMA : 1, Core : 0>
    }
  }
}
