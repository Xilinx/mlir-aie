//===- column_control_overlay_all_tiles_routed.mlir -----------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Responses routed from the compute tiles enter the shim from North, and still
// count as routed, so a second run adds nothing.

// RUN: aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tct=all-tiles" -aie-create-pathfinder-flows | aie-opt -aie-generate-column-control-overlay="route-shim-to-tct=all-tiles" | FileCheck %s

// CHECK:     aie.packet_rules(North
// CHECK-NOT: aie.packet_flow

aie.device(npu1_2col) {
  %shim0 = aie.tile(0, 0)
  %compute0 = aie.tile(0, 2)
  %shim1 = aie.tile(1, 0)
  %compute1 = aie.tile(1, 2)
}
