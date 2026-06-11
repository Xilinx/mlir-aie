//===- loc_preservation.mlir -----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 AMD Inc.
//
//===----------------------------------------------------------------------===//

// Verifies --aie-create-pathfinder-flows propagates the source FlowOp's
// location to the synthesized ConnectOps inside switchboxes, and that
// per-tile WireOps inherit the corresponding tile's location.

// RUN: aie-opt --aie-create-pathfinder-flows --mlir-print-debuginfo %s | FileCheck %s

#flow_loc = loc("user_design.py":42:4)
#tile_loc = loc("user_design.py":50:8)

module {
  aie.device(xcvc1902) {
    %0 = aie.tile(2, 3) loc(#tile_loc)
    %1 = aie.tile(3, 2) loc(#tile_loc)
    aie.flow(%0, Core : 1, %1, DMA : 0) loc(#flow_loc)
  }
}

// Synthesized aie.connect ops inside switchboxes carry the FlowOp's loc.
// CHECK-DAG: aie.connect<{{.*}}> loc(#[[FLOWLOC:loc[0-9]*]])
// Synthesized aie.wire ops between switchboxes/tiles carry the tile's loc.
// CHECK-DAG: aie.wire({{.*}}) loc(#[[TILELOC:loc[0-9]*]])
// CHECK-DAG: #[[FLOWLOC]] = loc("user_design.py":42:4)
// CHECK-DAG: #[[TILELOC]] = loc("user_design.py":50:8)
