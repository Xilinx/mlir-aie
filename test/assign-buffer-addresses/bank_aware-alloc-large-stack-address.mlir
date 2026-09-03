//===- bank_aware-alloc-large-stack-address.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck %s

// Buffers placed above a large stack start at the first free address in the
// bank that covers it, and pack behind one another: placement maximizes the
// free run it leaves behind.
// CHECK: %a = aie.buffer(%tile_3_3) {address = 18432 : i32, mem_bank = 1 : i32, sym_name = "a"} : memref<1024xi8>
// CHECK: %b = aie.buffer(%tile_3_3) {address = 19456 : i32, mem_bank = 1 : i32, sym_name = "b"} : memref<1024xi8>
module @test {
  aie.device(npu2) {
    %0 = aie.tile(3, 3)
    %b1 = aie.buffer(%0) {  sym_name = "a" } : memref<1024xi8>
    %b2 = aie.buffer(%0) {  sym_name = "b" } : memref<1024xi8>
    aie.core(%0) {
      aie.end
    }{ stack_size = 18432 : i32 }
  }
}
