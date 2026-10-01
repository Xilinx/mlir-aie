//===- column_control_overlay_all_tiles_unconnected.mlir ------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A shim rule on North that matches the compute tile's response ID and feeds
// the marked South : 0 master is not a response route unless the compute
// tile's TileControl : 0 actually reaches it. Here nothing is connected above
// the shim, so the compute tile still gets its response flow.

// RUN: aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tct=all-tiles" | FileCheck %s

// CHECK-DAG: %[[COMPUTE0:.*]] = aie.tile(0, 2)
// CHECK-DAG: %[[SHIM0:.*]] = aie.tile(0, 0)
// CHECK:      aie.packet_source<%[[COMPUTE0]], TileControl : 0>
// CHECK-NEXT: aie.packet_dest<%[[SHIM0]], South : 0>

aie.device(npu1_2col) {
  %shim0 = aie.tile(0, 0)
  %compute0 = aie.tile(0, 2)
  %shim1 = aie.tile(1, 0)
  %compute1 = aie.tile(1, 2)
  %sb = aie.switchbox(%shim0) {
    %ctrl = aie.amsel<5> (0)
    %m = aie.masterset(South : 0, %ctrl) {is_ctrl_pkt_overlay}
    aie.packet_rules(TileControl : 0) {
      aie.rule(31, 15, %ctrl)
    }
    aie.packet_rules(North : 1) {
      aie.rule(0, 0, %ctrl)
    }
  }
}
