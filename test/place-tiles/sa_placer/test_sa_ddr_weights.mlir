//===- test_sa_ddr_weights.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: asserts

// The DDR spread term weighs each shim end by its share of a sequence's
// bytes: two ends in flight together on their own columns, moving 1024 and
// 3072 words, still load the columns unevenly. When an end's bytes are not
// static, every end of its sequence weighs alike, and the same placement is
// even.

// RUN: aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=1' -debug-only=aie-sa-placer %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: Final {{.*}} ddr=0,
// CHECK: Final {{.*}} ddr=2,

module @ddr_dynamic {
  aie.device(npu2) {
    %s0 = aie.logical_tile<ShimNOCTile>(0, 0)
    %s1 = aie.logical_tile<ShimNOCTile>(1, 0)
    %k0 = aie.logical_tile<CoreTile>(0, 2)
    %k1 = aie.logical_tile<CoreTile>(1, 2)
    aie.objectfifo @w0(%s0, {%k0}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @w1(%s1, {%k1}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.runtime_sequence @run(%a : memref<4096xi32>, %b : memref<4096xi32>, %n : i32) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %trips = arith.index_cast %n : i32 to index
      scf.for %i = %c0 to %trips step %c1 {
        aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {metadata = @w0, id = 0 : i64} : memref<4096xi32>
      }
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, 3072][0, 0, 0, 1]) {metadata = @w1, id = 1 : i64} : memref<4096xi32>
    }
  }
}

// -----

module @ddr_static {
  aie.device(npu2) {
    %s0 = aie.logical_tile<ShimNOCTile>(0, 0)
    %s1 = aie.logical_tile<ShimNOCTile>(1, 0)
    %k0 = aie.logical_tile<CoreTile>(0, 2)
    %k1 = aie.logical_tile<CoreTile>(1, 2)
    aie.objectfifo @w0(%s0, {%k0}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @w1(%s1, {%k1}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.runtime_sequence @run(%a : memref<4096xi32>, %b : memref<4096xi32>, %n : i32) {
      %unused = arith.index_cast %n : i32 to index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %trips = arith.constant 1 : index
      scf.for %i = %c0 to %trips step %c1 {
        aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {metadata = @w0, id = 0 : i64} : memref<4096xi32>
      }
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, 3072][0, 0, 0, 1]) {metadata = @w1, id = 1 : i64} : memref<4096xi32>
    }
  }
}
