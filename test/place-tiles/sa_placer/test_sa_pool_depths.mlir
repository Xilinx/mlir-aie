//===- test_sa_pool_depths.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The SA placer charges each end the pool aie-objectfifo-split gives it, so a
// core that only overflows with the split's depths is reported over capacity.

// RUN: aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=1' --aie-objectfifo-split %s 2> %t.err | FileCheck %s
// RUN: FileCheck %s --check-prefix=ERR --input-file=%t.err

// init_values fill every object, whatever the producer acquires.
// ERR: :[[# @LINE + 4]]:3: warning: SA placer's resource estimate is over capacity
// CHECK-LABEL: module @init_values
// CHECK: aie.objectfifo.pool @f__pool(%tile_7_2) {depth = 3 : i32
module @init_values {
  aie.device(npu2) {
    %c = aie.logical_tile<CoreTile>(7, 2)
    %m = aie.logical_tile<MemTile>(7, 1)
    aie.objectfifo @f (%c, {%m}, 3 : i32) : !aie.objectfifo<memref<6000xi32>> = [dense<0> : memref<6000xi32>, dense<1> : memref<6000xi32>, dense<2> : memref<6000xi32>]
    aie.core(%c) {
      %0 = aie.objectfifo.acquire @f(Produce, 1) : memref<6000xi32>
      aie.objectfifo.release @f(Produce, 1)
      aie.end
    }
  }
}

// -----

// A per-end depth is taken at its word, whatever the consumer acquires.
// ERR: :[[# @LINE + 4]]:3: warning: SA placer's resource estimate is over capacity
// CHECK-LABEL: module @per_end_depth
// CHECK: aie.objectfifo.pool @f__cons__pool(%tile_7_2) {depth = 3 : i32
module @per_end_depth {
  aie.device(npu2) {
    %m = aie.logical_tile<MemTile>(7, 1)
    %c = aie.logical_tile<CoreTile>(7, 2)
    aie.objectfifo @f (%m, {%c}, [2 : i32, 3 : i32]) : !aie.objectfifo<memref<6000xi32>>
    aie.core(%c) {
      %0 = aie.objectfifo.acquire @f(Consume, 1) : memref<6000xi32>
      aie.objectfifo.release @f(Consume, 1)
      aie.end
    }
  }
}

// -----

// An end wired to a core stream port keeps no pool.
// ERR-NOT: warning
// CHECK-LABEL: module @stream_end
// CHECK-NOT: @f__pool
// CHECK: aie.objectfifo.pool @f__cons__pool(%mem_tile_7_1)
// CHECK-NOT: @f__pool
module @stream_end {
  aie.device(npu2) {
    %c = aie.logical_tile<CoreTile>(7, 2)
    %m = aie.logical_tile<MemTile>(7, 1)
    %big = aie.buffer(%c) : memref<13000xi32>
    aie.objectfifo @f (%c, {%m}, 1 : i32) {prod_port = #aie.end_port<Core : 0>} : !aie.objectfifo<memref<4000xi32>>
  }
}
