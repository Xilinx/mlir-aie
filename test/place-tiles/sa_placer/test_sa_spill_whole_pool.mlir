//===- test_sa_spill_whole_pool.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A pool aie-objectfifo-allocate walks whole is one buffer that spills only
// whole. Each mem tile holds 300000 bytes of its own; the link's 30-deep pool
// of 8 KB objects needs 30 BD ids an end, more than a channel has, so the
// allocator walks it whole: 240 KB, which neither tile has left (224288 bytes
// each), though as 30 objects it would split across the two. The placer's
// estimate counts it whole and warns.

// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=1' %s -o /dev/null 2>&1 | FileCheck %s
// CHECK: warning: SA placer's resource estimate is over capacity

module {
  aie.device(npu2_2col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %m0 = aie.logical_tile<MemTile>(0, 1)
    %m1 = aie.logical_tile<MemTile>(1, 1)
    %t = aie.logical_tile<CoreTile>(?, ?)
    %b0 = aie.buffer(%m0) : memref<75000xi32>
    %b1 = aie.buffer(%m1) : memref<75000xi32>
    aie.objectfifo @in(%s, {%m0}, [2 : i32, 30 : i32]) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo @held(%m0, {%t}, 2 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo.link [@in] -> [@held] ([] [])
    aie.core(%t) {
      %o = aie.objectfifo.acquire @held(Consume, 1) : memref<2048xi32>
      aie.objectfifo.release @held(Consume, 1)
      aie.end
    }
  }
}
