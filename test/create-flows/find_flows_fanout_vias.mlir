//===- find_flows_fanout_vias.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// With emit-vias, --aie-find-flows splits a broadcast at its fan-out node into
// linear sections: a trunk flow to the fan-out's input, the fan-out switchbox
// left explicit, and a branch flow from each output.  The shared trunk is not
// duplicated, so the result splits without colliding.

// RUN: aie-opt --aie-create-pathfinder-flows %s | aie-opt --aie-find-flows=emit-vias=true | FileCheck %s

// Circuit broadcast: (0,2) -> {(0,4), (0,5)}, fanning out at (0,4).
// CHECK: %[[T02:.*]] = aie.tile(0, 2)
// CHECK: %[[T04:.*]] = aie.tile(0, 4)
// CHECK: %[[T05:.*]] = aie.tile(0, 5)
// CHECK: %[[T03:.*]] = aie.tile(0, 3)
// The fan-out node stays explicit (one input driving two outputs).
// CHECK: aie.switchbox(%[[T04]]) {
// CHECK:   aie.connect<South : [[IN:[0-9]+]], DMA : 0>
// CHECK:   aie.connect<South : [[IN]], North : [[OUT:[0-9]+]]>
// CHECK: }
// Trunk ends at the fan-out input; branch starts at the fan-out output.
// CHECK: aie.flow(%[[T02]], DMA : 0, %[[T04]], South : [[IN]]) via (%[[T02]] : DMA : 0 -> North : {{[0-9]+}}, %[[T03]] : South : {{[0-9]+}} -> North : {{[0-9]+}})
// CHECK: aie.flow(%[[T04]], North : [[OUT]], %[[T05]], DMA : 0) via (%[[T05]] : South : {{[0-9]+}} -> DMA : 0)
module {
  aie.device(xcvc1902) {
    %t02 = aie.tile(0, 2)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    aie.flow(%t02, DMA : 0, %t04, DMA : 0)
    aie.flow(%t02, DMA : 0, %t05, DMA : 0)
  }
}

