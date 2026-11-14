//===- test_sa_lock_links.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" --mlir-pass-statistics %s -o %t 2> %t.err
// RUN: FileCheck %s --input-file=%t.err
// RUN: aie-opt --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate %t | FileCheck %s --check-prefix=ALLOC

// @x0 and @x1 are each one link's output and the join's input. Each link's
// pool takes over the ends on its shared tile, whichever link a fifo names
// first: three link pools of 1, 1 and 2 segments and @y's consumer pool.
// CHECK: 5 sa-lock-groups
// ALLOC-COUNT-5: __prodlock{{[0-9]+}} = aie.lock
// ALLOC-NOT: __prodlock{{[0-9]+}} = aie.lock

module @link_into_join {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %m0 = aie.logical_tile<MemTile>(?, ?)
    %m1 = aie.logical_tile<MemTile>(?, ?)
    %m2 = aie.logical_tile<MemTile>(?, ?)
    %core = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%shim, {%m0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%shim, {%m1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @x0(%m0, {%m2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @x1(%m1, {%m2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @y(%m2, {%core}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@x0, @x1] -> [@y] ([0, 16] [])
    aie.objectfifo.link [@a] -> [@x0] ([] [0])
    aie.objectfifo.link [@b] -> [@x1] ([] [0])
    aie.core(%core) {
      %0 = aie.objectfifo.acquire @y(Consume, 1) : memref<32xi32>
      aie.objectfifo.release @y(Consume, 1)
      aie.end
    }
  }
}
