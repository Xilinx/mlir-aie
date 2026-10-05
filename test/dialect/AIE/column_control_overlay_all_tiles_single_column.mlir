//===- column_control_overlay_all_tiles_single_column.mlir ----*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Every column's shim response is routed, but the compute tile's is not. With
// route-shim-to-tct=all-tiles the device is not fully overlaid, so the compute
// tile still gets its response flow. With shim-only the overlay is complete
// and nothing is added.

// RUN: aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tct=all-tiles" | FileCheck %s --check-prefix=ALL
// RUN: aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tct=shim-only" | FileCheck %s --check-prefix=SHIM

// ALL-DAG: %[[SHIM0:.*]] = aie.tile(0, 0)
// ALL-DAG: %[[COMPUTE0:.*]] = aie.tile(0, 2)
// ALL-NOT: aie.packet_source<%[[SHIM0]], TileControl : 0>
// ALL:      aie.packet_source<%[[COMPUTE0]], TileControl : 0>
// ALL-NEXT: aie.packet_dest<%[[SHIM0]], South : 0>
// ALL-NOT: aie.packet_source<%[[SHIM0]], TileControl : 0>

// SHIM-NOT: aie.packet_flow

aie.device(npu1_1col) {
  %shim0 = aie.tile(0, 0)
  %compute0 = aie.tile(0, 2)
  %sb = aie.switchbox(%shim0) {
    %ctrl = aie.amsel<5> (0)
    %m = aie.masterset(South : 0, %ctrl) {is_ctrl_pkt_overlay}
    aie.packet_rules(TileControl : 0) {
      aie.rule(31, 15, %ctrl)
    }
  }
}
