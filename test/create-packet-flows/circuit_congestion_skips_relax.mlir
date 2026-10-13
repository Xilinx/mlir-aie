//===- circuit_congestion_skips_relax.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: asserts
// RUN: not aie-opt --aie-create-pathfinder-flows -debug-only=aie-pathfinder %s 2>&1 | FileCheck %s

// Five circuit flows need the four links from (0,2) down to the memtile. The
// packet flows take other links, so loosening how they are routed cannot help,
// and the router reports the failure without searching again.

// CHECK: No packet stream crosses an overused link
// CHECK-NOT: Relax:
// CHECK: error: Unable to find a legal routing: the flows from (0, 2) Core:0, (0, 2) DMA:0, (0, 2) DMA:1, (0, 3) DMA:0 and 1 more need the links from tile (0, 2) to (0, 1)

module {
  aie.device(npu1_1col) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    aie.flow(%t02, DMA : 0, %t01, DMA : 0)
    aie.flow(%t02, DMA : 1, %t01, DMA : 1)
    aie.flow(%t02, Core : 0, %t01, DMA : 2)
    aie.flow(%t03, DMA : 0, %t01, DMA : 3)
    aie.flow(%t03, DMA : 1, %t01, DMA : 4)
    aie.packet_flow(1) {
      aie.packet_source<%t05, DMA : 0>
      aie.packet_dest<%t04, DMA : 0>
    }
    aie.packet_flow(2) {
      aie.packet_source<%t05, DMA : 1>
      aie.packet_dest<%t03, DMA : 1>
    }
  }
}
