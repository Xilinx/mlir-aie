//===- find_flows_fanout_vias.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// With emit-vias, --aie-find-flows splits a broadcast at its fan-out node into
// linear sections: a trunk flow to the fan-out's input and one branch flow per
// output. The shared trunk is not duplicated, so the result splits without
// colliding.

// RUN: aie-opt --aie-create-pathfinder-flows %s | aie-opt --aie-find-flows=emit-vias=true | FileCheck %s

// Circuit broadcast: (0,2) -> {(0,4), (0,5)}, fanning out at (0,4).
// CHECK: %[[T02:.*]] = aie.tile(0, 2)
// CHECK: %[[T04:.*]] = aie.tile(0, 4)
// CHECK: %[[T05:.*]] = aie.tile(0, 5)
// CHECK: %[[T06:.*]] = aie.tile(0, 6)
// CHECK: %[[T03:.*]] = aie.tile(0, 3)
// The trunk ends at the fan-out input. Each branch pins its fan-out connection.
// CHECK: aie.flow(%[[T02]], DMA : 0, %[[T04]], South : [[IN:[0-9]+]]) via (%[[T02]] : DMA : 0 -> North : {{[0-9]+}}, %[[T03]] : South : {{[0-9]+}} -> North : {{[0-9]+}})
// CHECK: aie.flow(%[[T04]], South : [[IN]], %[[T04]], DMA : 0) via (%[[T04]] : South : [[IN]] -> DMA : 0)
// CHECK: aie.flow(%[[T04]], South : [[IN]], %[[T05]], South : [[MID:[0-9]+]]) via (%[[T04]] : South : [[IN]] -> North : [[MID]])
// CHECK: aie.flow(%[[T05]], South : [[MID]], %[[T05]], DMA : 0) via (%[[T05]] : South : [[MID]] -> DMA : 0)
// CHECK: aie.flow(%[[T05]], South : [[MID]], %[[T06]], DMA : 0) via (%[[T05]] : South : [[MID]] -> North : [[OUT:[0-9]+]], %[[T06]] : South : [[OUT]] -> DMA : 0)
module {
  aie.device(xcvc1902) {
    %t02 = aie.tile(0, 2)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    %t06 = aie.tile(0, 6)
    aie.flow(%t02, DMA : 0, %t04, DMA : 0)
    aie.flow(%t02, DMA : 0, %t05, DMA : 0)
    aie.flow(%t02, DMA : 0, %t06, DMA : 0)
  }
}
