//===- per_tile_alloc_respects_buffer_alloc.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-buffer-addresses %s | FileCheck %s

module @test {
  aie.device(xcvc1902) {

    // Every buffer here is pinned, and some of the pins overlap ranges the
    // allocator would otherwise choose. Pins win: the pass places them exactly
    // where asked and allocates nothing around them.

    %t1 = aie.tile(0, 1)
    //CHECK: address = 2048
    %buf0 = aie.buffer(%t1) { address = 2048 : i32 } : memref<1024xi8>
    //CHECK: address = 3072
    %buf2 = aie.buffer(%t1) { address = 3072 : i32 } : memref<1024xi32>
    //CHECK: address = 12288
    %b3 = aie.buffer(%t1) { address = 12288 : i32 } : memref<1024xi32>
    //CHECK: address = 20000
    %b34 = aie.buffer(%t1) { address = 20000 : i32 } : memref<1024xi32>

    aie.core(%t1) {
      aie.end
    }{ stack_size = 2048 : i32}
  }
}
