//===- test_place_shared_mem_infeasible.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The default placer enforces shared-memory adjacency too, not just
// sa_placer: a core cannot share memory with five others, so placement
// fails here instead of succeeding and only failing later in
// --aie-objectfifo-split with a diagnostic that doesn't point at placement.

// RUN: not aie-opt --aie-place-tiles %s 2>&1 | FileCheck %s

// CHECK: error: no available compute tiles for placement (shared-memory transport unsatisfiable)

module @shared_mem_five {
  aie.device(npu2) {
    %t0 = aie.logical_tile<CoreTile>(?, ?)
    %t1 = aie.logical_tile<CoreTile>(?, ?)
    %t2 = aie.logical_tile<CoreTile>(?, ?)
    %t3 = aie.logical_tile<CoreTile>(?, ?)
    %t4 = aie.logical_tile<CoreTile>(?, ?)
    %t5 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @near1(%t0, {%t1}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @near2(%t0, {%t2}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @near3(%t0, {%t3}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @near4(%t0, {%t4}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @near5(%t0, {%t5}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<8xi32>>
    aie.core(%t0) { aie.end }
    aie.core(%t1) { aie.end }
    aie.core(%t2) { aie.end }
    aie.core(%t3) { aie.end }
    aie.core(%t4) { aie.end }
    aie.core(%t5) { aie.end }
  }
}
