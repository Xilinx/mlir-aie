//===- test_sa_final_check.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Cascade adjacency is exact, so an unsatisfied cascade fails the pass. Memory
// use is an estimate checked exactly by later passes, so going over it warns
// and the placement is kept.

// RUN: not aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=42' %s 2>&1 | FileCheck %s

// (7, 2) has no core to its east or south, so the cascade cannot be placed.
// CHECK: error: SA placer failed to find a legal placement (cascade penalty={{[1-9][0-9]*}}, delegate penalty=0)
module @cascade_unplaceable {
  aie.device(npu2) {
    %src = aie.logical_tile<CoreTile>(7, 2)
    %dst = aie.logical_tile<CoreTile>(?, ?)
    aie.cascade_flow(%src, %dst)
    aie.core(%src) { aie.end }
    aie.core(%dst) { aie.end }
  }
}

// -----

// CHECK: warning: SA placer's resource estimate is over capacity
// CHECK-LABEL: module @core_memory_over_estimate
// CHECK: aie.tile(
// CHECK-NOT: aie.logical_tile
module @core_memory_over_estimate {
  aie.device(npu2) {
    %c = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.buffer(%c) : memref<20000xi32>
    aie.core(%c) { aie.end }
  }
}
