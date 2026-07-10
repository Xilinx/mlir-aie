//===- bank_aware_alloc_memtile_simple.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s 2>&1 | FileCheck %s
// CHECK:       %a = aie.buffer(%{{.*}}_3_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "a"} : memref<16384xi32>

module @test {
  aie.device(xcve2302) {
    %0 = aie.tile(3, 1)
    %b1 = aie.buffer(%0) { sym_name = "a" } : memref<16384xi32>
    aie.memtile_dma(%0) {
      aie.end
    }
  }
}
