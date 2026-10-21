//===- overused_link_names_flows.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s
// RUN: not aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 | FileCheck %s

// Five circuit flows need the four links from (0,2) down to the memtile, and
// the error names the boundary they cross and the flows that cross it.

// CHECK: error: Unable to find a legal routing: the circuit flows from (0, 2) Core:0, (0, 2) DMA:0, (0, 2) DMA:1, (0, 3) DMA:0 and 1 more cross from row 2 to row 1, which takes 5 channels across, and it has 4 channels.

module {
  aie.device(npu1_1col) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.flow(%t02, DMA : 0, %t01, DMA : 0)
    aie.flow(%t02, DMA : 1, %t01, DMA : 1)
    aie.flow(%t02, Core : 0, %t01, DMA : 2)
    aie.flow(%t03, DMA : 0, %t01, DMA : 3)
    aie.flow(%t03, DMA : 1, %t01, DMA : 4)
  }
}
