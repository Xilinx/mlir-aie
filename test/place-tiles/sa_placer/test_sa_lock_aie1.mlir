//===- test_sa_lock_aie1.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" --mlir-pass-statistics %s -o /dev/null 2>&1 | FileCheck %s
// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" %s | not aie-opt --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate 2>&1 | FileCheck %s --check-prefix=ALLOC

// On AIE1 a pool takes one lock an object: @f and @g need 18 locks and the
// core's tile, the only one placed, has 16. The SA charges the 9 it cannot
// place, as allocation fails.
// CHECK: 2 sa-lock-groups
// CHECK-NEXT: 180 sa-lock-penalty
// CHECK-NEXT: 0 sa-lock-stopped
// ALLOC: could not place locks within tile lock capacity

module @aie1_deep_pools {
  aie.device(xcvc1902) {
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @f(%c, {%c}, 9 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @g(%c, {%c}, 9 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %0 = aie.objectfifo.acquire @f(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f(Produce, 1)
      %1 = aie.objectfifo.acquire @f(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f(Consume, 1)
      %2 = aie.objectfifo.acquire @g(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @g(Produce, 1)
      %3 = aie.objectfifo.acquire @g(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @g(Consume, 1)
      aie.end
    }
  }
}
