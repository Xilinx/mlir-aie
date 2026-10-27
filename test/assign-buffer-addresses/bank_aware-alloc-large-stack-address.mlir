//===- bank_aware-alloc-large-stack-address.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses %s | FileCheck %s

// A large stack reservation pushes the first buffer into the bank that covers
// the top of the stack. Round-robin then moves to the next bank.
// CHECK: aie.buffer @a(%tile_3_3) {address = 18432 : i32, mem_bank = 1 : i32} : memref<1024xi8>
// CHECK: aie.buffer @b(%tile_3_3) {address = 32768 : i32, mem_bank = 2 : i32} : memref<1024xi8>
module @test {
  aie.device(npu2) {
    %0 = aie.tile(3, 3)
    aie.buffer @a(%0) : memref<1024xi8>
    aie.buffer @b(%0) : memref<1024xi8>
    aie.core(%0) {
      aie.end
    }{ stack_size = 18432 : i32 }
  }
}
