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
  aie.device(npu2_1col) @design {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.packet_flow(3) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, DMA : 0>
    }
  }
}
