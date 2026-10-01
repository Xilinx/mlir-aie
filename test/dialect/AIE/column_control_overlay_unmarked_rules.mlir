//===- column_control_overlay_unmarked_rules.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Pathfinder marks a slave's packet_rules only from the first group it routes.
// When a non-overlay flow from TileControl : 0 is routed first, the rules carry
// no marker even though they hold the response route to a marked master. That
// route still counts, so no second response flow is generated.

// RUN: aie-opt %s -aie-generate-column-control-overlay | FileCheck %s

// CHECK:     aie.packet_rules(TileControl : 0) {
// CHECK-NOT: aie.packet_flow

aie.device(npu1_1col) {
  %shim = aie.tile(0, 0)
  %compute = aie.tile(0, 2)
  %sb = aie.switchbox(%shim) {
    %user = aie.amsel<0> (0)
    %ctrl = aie.amsel<5> (0)
    %m0 = aie.masterset(North : 0, %user)
    %m1 = aie.masterset(South : 0, %ctrl) {is_ctrl_pkt_overlay}
    aie.packet_rules(TileControl : 0) {
      aie.rule(31, 2, %user)
      aie.rule(31, 15, %ctrl)
    }
  }
}
