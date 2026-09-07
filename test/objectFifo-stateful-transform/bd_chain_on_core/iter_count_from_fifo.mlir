//===- bd_chain_on_core/iter_count_from_fifo.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectfifo-split %s | FileCheck %s

// Both ends carry `depth * repeat_count * iter_count` objects. The drainer
// replays each one within a single pass, so it makes `repeat_count` times
// fewer passes than the filler opposite it. Neither tile is a MemTile.

module @iter_count_from_fifo {
  aie.device(npu1) {
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)

    aie.objectfifo @of (%tile02, {%tile03}, 2 : i32)
        {iter_count = 5 : i32, repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
  }
}

// CHECK: aie.objectfifo.pool @of_pool(%tile_0_2) {{{.*}}repeatCount = 3 : i32
// CHECK-NOT: iterCount
// CHECK: aie.objectfifo.dma_endpoint @of_prod_dma(%tile_0_2) drains @of_pool {{{.*}}iterCount = 5 : i32
// CHECK: aie.objectfifo.pool @of_cons_pool(%tile_0_3)
// CHECK-NOT: repeatCount
// CHECK: aie.objectfifo.dma_endpoint @of_cons_dma(%tile_0_3) fills @of_cons_pool {{{.*}}iterCount = 15 : i32
