// RUN: aie-opt --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --aie-objectfifo-allocate %s -o %t1.mlir
// RUN: aie-opt --aie-objectfifo-allocate %t1.mlir -o %t2.mlir
// RUN: diff %t1.mlir %t2.mlir

// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Buffers for each object, a counting lock pair for each segment, a DMA channel
// for each endpoint, and the flow between them.

module @resources {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo.pool @prod_pool(%tile12) {
      depth = 2 : i32, segments = [#aie.objectfifo_segment<offset = 0, size = 16>]
    } : memref<16xi32>
    aie.objectfifo.dma_endpoint @prod_dma(%tile12) drains @prod_pool

    aie.objectfifo.pool @cons_pool(%tile33) {
      depth = 3 : i32, segments = [#aie.objectfifo_segment<offset = 0, size = 16>]
    } : memref<16xi32>
    aie.objectfifo.dma_endpoint @cons_dma(%tile33) fills @cons_pool

    aie.objectfifo.flow from @prod_dma to [@cons_dma]
  }
}

// CHECK-LABEL: @resources
// CHECK-DAG:   %[[T12:.*]] = aie.tile(1, 2)
// CHECK-DAG:   %[[T33:.*]] = aie.tile(3, 3)
// CHECK-DAG:   aie.buffer(%[[T12]]) {sym_name = "prod_buff_0"} : memref<16xi32>
// CHECK-DAG:   aie.buffer(%[[T12]]) {sym_name = "prod_buff_1"} : memref<16xi32>
// CHECK-DAG:   aie.buffer(%[[T33]]) {sym_name = "cons_buff_0"} : memref<16xi32>
// CHECK-DAG:   aie.buffer(%[[T33]]) {sym_name = "cons_buff_1"} : memref<16xi32>
// CHECK-DAG:   aie.buffer(%[[T33]]) {sym_name = "cons_buff_2"} : memref<16xi32>

// The producer's lock counts free objects, the consumer's counts full ones.
// CHECK-DAG:   aie.lock(%[[T12]]) {init = 2 : i32, sym_name = "prod_prod_lock_0"}
// CHECK-DAG:   aie.lock(%[[T12]]) {init = 0 : i32, sym_name = "prod_cons_lock_0"}
// CHECK-DAG:   aie.lock(%[[T33]]) {init = 3 : i32, sym_name = "cons_prod_lock_0"}
// CHECK-DAG:   aie.lock(%[[T33]]) {init = 0 : i32, sym_name = "cons_cons_lock_0"}

// CHECK:   aie.objectfifo.pool @prod_pool({{.*}}) {buffers = [@prod_buff_0, @prod_buff_1], depth = 2 : i32, segments = [#aie.objectfifo_segment<offset = 0, size = 16, produceLock = @prod_prod_lock_0, consumeLock = @prod_cons_lock_0>]}
// CHECK:   aie.objectfifo.dma_endpoint @prod_dma({{.*}}) drains @prod_pool {channelIndex = 0 : i32}
// CHECK:   aie.objectfifo.pool @cons_pool({{.*}}) {buffers = [@cons_buff_0, @cons_buff_1, @cons_buff_2], depth = 3 : i32, segments = [#aie.objectfifo_segment<offset = 0, size = 16, produceLock = @cons_prod_lock_0, consumeLock = @cons_cons_lock_0>]}
// CHECK:   aie.objectfifo.dma_endpoint @cons_dma({{.*}}) fills @cons_pool {channelIndex = 0 : i32}
// CHECK:   aie.flow(%[[T12]], DMA : 0, %[[T33]], DMA : 0)
// CHECK-NOT: aie.objectfifo.flow
