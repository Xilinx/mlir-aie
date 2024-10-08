//===- basic_alloc_memtile_simple.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=basic-sequential" %s 2>&1 | FileCheck %s
// CHECK:   module @test {
// CHECK:   aie.device(xcve2302) {
// CHECK:     %tile_3_1 = aie.tile(3, 1)
// CHECK:     %a = aie.buffer(%tile_3_1) {address = 0 : i32, sym_name = "a"} : memref<65536xi32> 
// CHECK:     %memtile_dma_3_1 = aie.memtile_dma(%tile_3_1) {
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }
module @test {
 aie.device(xcve2302) {
  %0 = aie.tile(3, 1)
  %b1 = aie.buffer(%0) { sym_name = "a" } : memref<65536xi32>
  aie.memtile_dma(%0) {
    aie.end
  }
 }
}
