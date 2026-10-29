//===- test_sa_chained_links_infeasible.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-place-tiles='placer=sa_placer sa-seed=1' %s 2>&1 | FileCheck %s

// Exact separation constraints must fail here, not produce invalid placed IR.
// CHECK: error: SA placer failed to separate chained ObjectFifo links
module @pinned_chain {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %src = aie.logical_tile<MemTile>(0, 1)
    %dst = aie.logical_tile<MemTile>(0, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo @in (%shim, {%src}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @mid (%src, {%dst}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out (%dst, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@mid] ([] [])
    aie.objectfifo.link [@mid] -> [@out] ([] [])
  }
}
