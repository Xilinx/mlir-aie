//===- ctrl_overlay_shared_rule.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The control overlay stays materialized. The router merges flow 9 into an
// overlay rule at the shim, and that rule still claims id 9, so flow 9 stays
// materialized as well. The shim mux connect both use is kept, and routing
// the result again adds nothing.

// RUN: aie-opt --aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s
// RUN: aie-opt --aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" --aie-create-pathfinder-flows --aie-find-flows --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK-NOT:  aie.packet_flow
// CHECK:      aie.shim_mux
// CHECK-NEXT:   aie.connect<DMA : 0, North : 3>
// CHECK-NEXT: }
// CHECK:      aie.switchbox(%{{.*}}tile_0_2)
// CHECK:        aie.masterset(DMA : 1,
// CHECK:        aie.rule(31, 9,
// CHECK-NOT:  aie.packet_flow

module {
  aie.device(npu1_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    aie.packet_flow(9) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t02, DMA : 1>
    }
  }
}
