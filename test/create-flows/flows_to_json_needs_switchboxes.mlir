//===- flows_to_json_needs_switchboxes.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s | not aie-translate --aie-flows-to-json 2>&1 | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows=remove-lifted=false %s | aie-translate --aie-flows-to-json | FileCheck %s --check-prefix=JSON

// By default aie-find-flows removes the switchboxes it lifts flows from, and
// the JSON traces each flow through them.

// CHECK: error: no switchbox at the flow's source to trace it through; keep the routed switchboxes with --aie-find-flows=remove-lifted=false

// JSON: "route0": [ {{\[\[}}0, 0], ["North"]], {{\[\[}}0, 1], ["North"]], {{\[\[}}0, 2], ["DMA"]], [] ],
// JSON: "route1": [ {{\[\[}}0, 2], ["South"]], {{\[\[}}0, 1], ["South"]], {{\[\[}}0, 0], ["South"]], [] ],

module {
  aie.device(npu2) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.flow(%t00, DMA : 0, %t02, DMA : 0)
    aie.flow(%t02, DMA : 0, %t00, DMA : 0)
  }
}
