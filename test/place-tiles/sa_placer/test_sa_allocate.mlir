//===- test_sa_allocate.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Verify SA placer generates objectfifo.allocate when a core tile's buffers
// exceed 64KB local memory and must spill to a neighbor's memory.

// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=42' %s | FileCheck %s

// Core B (0,3) receives data from A (0,2) and C (0,4).
// B's consumer buffers exceed 64KB. One fifo's buffer should be
// redirected to a neighbor with spare capacity via objectfifo.allocate.
//
// A (0,2): produces fifo1 (depth 1, 32KB on A). Light load on A: 32+4=36KB
// B (0,3): consumes fifo1 (depth 2, 64KB) + consumes fifo2 (depth 1, 8KB) + overhead = 76KB > 64KB
// C (0,4): produces fifo2 (depth 1, 8KB on C). Light load on C: 8+4=12KB
//
// B can redirect fifo2's buffer to C (north neighbor, 12KB + 8KB = 20KB << 64KB)
//
// CHECK-LABEL: @neighbor_spill
// CHECK: aie.objectfifo.allocate
// CHECK-NOT: aie.logical_tile
module @neighbor_spill {
  aie.device(npu2) {
    %coreA = aie.logical_tile<CoreTile>(0, 2)
    %coreB = aie.logical_tile<CoreTile>(0, 3)
    %coreC = aie.logical_tile<CoreTile>(0, 4)

    // A → B: producer depth=1 (32KB on A), consumer depth=2 (64KB on B)
    aie.objectfifo @fifo1(%coreA, {%coreB}, [1 : i32, 2 : i32]) : !aie.objectfifo<memref<8192xi32>>

    // C → B: depth=1, 8KB on each side
    aie.objectfifo @fifo2(%coreC, {%coreB}, 1 : i32) : !aie.objectfifo<memref<2048xi32>>

    aie.core(%coreA) { aie.end }
    aie.core(%coreB) { aie.end }
    aie.core(%coreC) { aie.end }
    aie.end
  }
}
