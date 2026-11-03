// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-generate-column-control-overlay="configuration=true adaptive=false" --aie-import-overlay --aie-annotate-shim-sharing --aie-create-pathfinder-flows %s -o %t
// RUN: FileCheck %s < %t
// RUN: aie-opt --aie-import-overlay %t | diff %t -
// RUN: aie-opt --aie-find-flows --aie-create-pathfinder-flows %t | FileCheck %s

// The application must not take an unused msel of an active overlay arbiter.
// CHECK-LABEL: aie.device(npu2_1col) @design
// CHECK: aie.switchbox(%shim_noc_tile_0_0)
// CHECK-DAG: aie.amsel<5> (0) {aie.is_ctrl_pkt_overlay}
// CHECK-DAG: aie.amsel<5> (1) {aie.is_ctrl_pkt_overlay}
// CHECK-DAG: aie.amsel<5> (2) {aie.is_ctrl_pkt_overlay}
// CHECK-DAG: aie.amsel<5> (3) {aie.is_ctrl_pkt_overlay}
// CHECK: aie.amsel<4> (3)
// CHECK-LABEL: aie.device(npu2_1col) @ctrl_pkt_overlay
// CHECK: aie.switchbox(%shim_noc_tile_0_0)
// CHECK-DAG: aie.amsel<5> (0) {aie.is_ctrl_pkt_overlay}
// CHECK-DAG: aie.amsel<5> (1) {aie.is_ctrl_pkt_overlay}
// CHECK-DAG: aie.amsel<5> (2) {aie.is_ctrl_pkt_overlay}
// CHECK-DAG: aie.amsel<5> (3) {aie.is_ctrl_pkt_overlay}

module {
  aie.device(npu2_1col) @design {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.packet_flow(5) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%core, DMA : 0>
    } {priority_route = true}
  }
}
