//===- test_sa_ddr_spread.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Four weight streams in flight together, each to a core of column 0: the
// SA spreads their shim ends over more shim columns than two, where nearness
// to column 0 alone packs them. Reversed and renamed, the design places
// alike. Two ends sharing explicitly are never in flight together and keep
// to column 0.

// RUN: aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=1' %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=2' %s | FileCheck %s

// CHECK-LABEL: module @ddr_spread
// CHECK-DAG: aie.tile(0, 0)
// CHECK-DAG: aie.tile(1, 0)
// CHECK-DAG: aie.tile(2, 0)
// CHECK-LABEL: module @ddr_spread_permuted
// CHECK-DAG: aie.tile(0, 0)
// CHECK-DAG: aie.tile(1, 0)
// CHECK-DAG: aie.tile(2, 0)
// CHECK-LABEL: module @ddr_apart
// CHECK: aie.tile(0, 0)
// CHECK-NOT: aie.tile({{[1-7]}}, 0)

module @ddr_spread {
  aie.device(npu2) {
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s3 = aie.logical_tile<ShimNOCTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(0, 2)
    %c1 = aie.logical_tile<CoreTile>(0, 3)
    %c2 = aie.logical_tile<CoreTile>(0, 4)
    %c3 = aie.logical_tile<CoreTile>(0, 5)
    aie.objectfifo @w0(%s0, {%c0}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @w1(%s1, {%c1}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @w2(%s2, {%c2}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @w3(%s3, {%c3}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.runtime_sequence @run(%a : memref<4096xi32>, %b : memref<4096xi32>, %c : memref<4096xi32>, %d : memref<4096xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1]) {metadata = @w0, id = 0 : i64} : memref<4096xi32>
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1]) {metadata = @w1, id = 1 : i64} : memref<4096xi32>
      aiex.npu.dma_memcpy_nd(%c[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1]) {metadata = @w2, id = 2 : i64} : memref<4096xi32>
      aiex.npu.dma_memcpy_nd(%d[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1]) {metadata = @w3, id = 3 : i64} : memref<4096xi32>
    }
  }
}

// -----

module @ddr_spread_permuted {
  aie.device(npu2) {
    %k3 = aie.logical_tile<CoreTile>(0, 5)
    %k2 = aie.logical_tile<CoreTile>(0, 4)
    %k1 = aie.logical_tile<CoreTile>(0, 3)
    %k0 = aie.logical_tile<CoreTile>(0, 2)
    %t3 = aie.logical_tile<ShimNOCTile>(?, ?)
    %t2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %t1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %t0 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @x3(%t3, {%k3}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @x2(%t2, {%k2}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @x1(%t1, {%k1}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @x0(%t0, {%k0}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.runtime_sequence @seq(%d : memref<4096xi32>, %c : memref<4096xi32>, %b : memref<4096xi32>, %a : memref<4096xi32>) {
      aiex.npu.dma_memcpy_nd(%d[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1]) {metadata = @x3, id = 3 : i64} : memref<4096xi32>
      aiex.npu.dma_memcpy_nd(%c[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1]) {metadata = @x2, id = 2 : i64} : memref<4096xi32>
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1]) {metadata = @x1, id = 1 : i64} : memref<4096xi32>
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1]) {metadata = @x0, id = 0 : i64} : memref<4096xi32>
    }
  }
}

// -----

module @ddr_apart {
  aie.device(npu2) {
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %k0 = aie.logical_tile<CoreTile>(0, 2)
    %k1 = aie.logical_tile<CoreTile>(0, 3)
    aie.objectfifo @w0(%s0, {%k0}, 2 : i32) {aie.shareable_with = [@w1]} : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @w1(%s1, {%k1}, 2 : i32) {aie.shareable_with = [@w0]} : !aie.objectfifo<memref<1024xi32>>
    aie.runtime_sequence @run(%a : memref<4096xi32>, %b : memref<4096xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1]) {metadata = @w0, id = 0 : i64, issue_token = true} : memref<4096xi32>
      aiex.npu.dma_wait {symbol = @w0}
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1]) {metadata = @w1, id = 1 : i64, issue_token = true} : memref<4096xi32>
      aiex.npu.dma_wait {symbol = @w1}
    }
  }
}
