//===- test_sa_lock_constants.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" --mlir-pass-statistics %s -o %t 2> %t.err
// RUN: FileCheck %s --input-file=%t.err
// RUN: aie-opt --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate %t | FileCheck %s --check-prefix=ALLOC

// The SA asks PoolLocks' rule of the pools the split makes. @w's producer
// pool starts full of its one initial object and nothing fills it again,
// so it holds constants and takes no locks; the consumer's pool does.
// CHECK: 1 sa-lock-groups
// ALLOC-COUNT-1: __prodlock{{[0-9]+}} = aie.lock
// ALLOC-NOT: __prodlock{{[0-9]+}} = aie.lock

module @constant_pool {
  aie.device(npu2) {
    %m = aie.logical_tile<MemTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @w(%m, {%c}, 1 : i32) : !aie.objectfifo<memref<4xi32>> = [dense<[1, 2, 3, 4]> : memref<4xi32>]
    aie.core(%c) {
      %0 = aie.objectfifo.acquire @w(Consume, 1) : memref<4xi32>
      aie.objectfifo.release @w(Consume, 1)
      aie.end
    }
  }
}
