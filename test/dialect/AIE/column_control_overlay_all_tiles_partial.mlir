//===- column_control_overlay_all_tiles_partial.mlir ----------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// With route-shim-to-tct=all-tiles, a routed shim response only covers the
// shim itself. The compute tile in the same column still gets its response.

// RUN: aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tct=all-tiles" | FileCheck %s

// CHECK-DAG: %[[SHIM0:.*]] = aie.tile(0, 0)
// CHECK-DAG: %[[COMPUTE0:.*]] = aie.tile(0, 2)
// CHECK-DAG: %[[SHIM1:.*]] = aie.tile(1, 0)
// CHECK-DAG: %[[COMPUTE1:.*]] = aie.tile(1, 2)
// CHECK-NOT: aie.packet_source<%[[SHIM0]], TileControl : 0>
// CHECK:      aie.packet_source<%[[COMPUTE0]], TileControl : 0>
// CHECK-NEXT: aie.packet_dest<%[[SHIM0]], South : 0>
// CHECK-NOT: aie.packet_source<%[[SHIM0]], TileControl : 0>
// CHECK:      aie.packet_source<%[[SHIM1]], TileControl : 0>
// CHECK-NEXT: aie.packet_dest<%[[SHIM1]], South : 0>
// CHECK:      aie.packet_source<%[[COMPUTE1]], TileControl : 0>
// CHECK-NEXT: aie.packet_dest<%[[SHIM1]], South : 0>
// CHECK-NOT: aie.packet_source<%[[SHIM0]], TileControl : 0>

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
  }
}
