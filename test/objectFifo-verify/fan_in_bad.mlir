//===- fan_in_bad.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-objectfifo-verify %s

// A fan-in's destination fills its next object with whatever packet arrives,
// so a source that moves a different amount would share an object with
// another.

module {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile32 = aie.tile(3, 2)
    %tile23 = aie.tile(2, 3)
    aie.objectfifo.pool @a_pool(%tile12) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @a(%tile12) fills @a_pool
    aie.objectfifo.dma_endpoint @a_dma(%tile12) drains @a_pool
    aie.objectfifo.pool @b_pool(%tile32) {depth = 1 : i32} : memref<32xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 32 : i32}
    }
    aie.objectfifo.core_endpoint @b(%tile32) fills @b_pool
    aie.objectfifo.dma_endpoint @b_dma(%tile32) drains @b_pool
    aie.objectfifo.pool @c_pool(%tile23) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @c_dma(%tile23) fills @c_pool
    aie.objectfifo.core_endpoint @c(%tile23) drains @c_pool
    // expected-error@+1 {{'aie.route' op fans in, so every end moves one object of the same size, but @a_dma moves 16 elements and @b_dma moves 32}}
    aie.route from [@a_dma, @b_dma] to [@c_dma] {packet = #aie.packet_info<>}
  }
}

// -----

// A transform on a source counts through its sizes: a 4 x 4 walk over a
// 32-element object moves 16, which is what the destination takes.

module {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile32 = aie.tile(3, 2)
    %tile23 = aie.tile(2, 3)
    aie.objectfifo.pool @a_pool(%tile12) {depth = 1 : i32} : memref<32xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 32 : i32}
    }
    aie.objectfifo.core_endpoint @a(%tile12) fills @a_pool
    aie.objectfifo.dma_endpoint @a_dma(%tile12) drains @a_pool {dimensions = #aie<bd_dim_layout_array_array[[<size = 4, stride = 8>, <size = 4, stride = 1>]]>}
    aie.objectfifo.pool @b_pool(%tile32) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @b(%tile32) fills @b_pool
    aie.objectfifo.dma_endpoint @b_dma(%tile32) drains @b_pool
    aie.objectfifo.pool @c_pool(%tile23) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @c_dma(%tile23) fills @c_pool
    aie.objectfifo.core_endpoint @c(%tile23) drains @c_pool
    aie.route from [@a_dma, @b_dma] to [@c_dma] {packet = #aie.packet_info<>}
  }
}
