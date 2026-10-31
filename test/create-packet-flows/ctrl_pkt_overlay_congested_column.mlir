// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-create-control-overlay --aie-create-pathfinder-flows %s -o %t
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
  aie.device(npu2) @design {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %mem_tile_1_1 = aie.tile(1, 1)
    %tile_1_2 = aie.tile(1, 2)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_0_3, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %mem_tile_0_1, DMA : 2)
    aie.flow(%mem_tile_0_1, DMA : 3, %tile_0_4, DMA : 0)
    aie.flow(%tile_0_4, DMA : 0, %mem_tile_0_1, DMA : 3)
    aie.flow(%mem_tile_0_1, DMA : 4, %tile_0_5, DMA : 0)
    aie.flow(%tile_0_5, DMA : 0, %mem_tile_0_1, DMA : 4)
    aie.flow(%mem_tile_0_1, DMA : 5, %tile_0_2, DMA : 1)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_3, DMA : 1)
  }
}
