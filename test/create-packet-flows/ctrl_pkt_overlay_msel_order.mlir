//===- ctrl_pkt_overlay_msel_order.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Flow 3 leaves the shim by DMA : 0 alongside the column control overlay's
// requests, as in npu-xrt/ctrl_packet_reconfig. That changes the order the
// overlay's master sets reach the arbiter plan, but their msels still match
// the standalone overlay's, so loading the design by control packets doesn't
// move them (#3837).

// RUN: aie-opt --pass-pipeline="builtin.module(aie-generate-column-control-overlay{route-shim-to-tile-ctrl=true emit-standalone-overlay=true},aie.device(aie-create-pathfinder-flows))" %s -o %t
// RUN: sed -n '/@design {/,/^  }/p' %t | FileCheck %s
// RUN: sed -n '/@ctrl_pkt_overlay {/,/^  }/p' %t | FileCheck %s

// CHECK-LABEL: aie.switchbox(%shim_noc_tile_0_0) {
// CHECK-DAG:     %[[A51:.+]] = aie.amsel<5> (1)
// CHECK-DAG:     %[[A52:.+]] = aie.amsel<5> (2)
// CHECK-DAG:     %[[A53:.+]] = aie.amsel<5> (3)
// CHECK-DAG:     aie.masterset(South : 0, %[[A53]]) {aie.is_ctrl_pkt_overlay, keep_pkt_header = true}
// CHECK-DAG:     aie.masterset(North : 4, %[[A52]]) {aie.is_ctrl_pkt_overlay}
// CHECK-DAG:     aie.masterset(TileControl : 0, %[[A51]]) {aie.is_ctrl_pkt_overlay, keep_pkt_header = true}

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
