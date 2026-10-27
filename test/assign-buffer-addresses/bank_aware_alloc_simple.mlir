//===- bank_aware_alloc_simple.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aie-prepare-buffers preserves declaration names; addresses come from the
// pass after it.
// RUN: aie-opt --split-input-file --aie-prepare-buffers --aie-assign-buffer-addresses %s | FileCheck %s

// No core reserves data memory here, so nothing constrains the free run and
// round-robin gives each buffer its own bank.
// CHECK: aie.buffer @a(%tile_3_3) {address = 16384 : i32, mem_bank = 2 : i32} : memref<16xi8>
// CHECK: aie.buffer @b(%tile_3_3) {address = 1024 : i32, mem_bank = 0 : i32} : memref<512xi32>
// CHECK: aie.buffer @c(%tile_3_3) {address = 8192 : i32, mem_bank = 1 : i32} : memref<16xi16>
// CHECK: aie.buffer @buffer_4(%tile_4_4) {address = 1024 : i32, mem_bank = 0 : i32} : memref<500xi32>
// The 2048-byte stack runs right up to "a1", pinned at 2048, so there is no
// hole below the pin and "b2"/"b3" pack above it. "b2" opts out of alignment
// and starts immediately after "a1"; "b3" rounds up to the 32-byte load/store
// bus width. (The stack size is spelled `stack_size`; the `stackSize` this
// test used to carry is not a registered attribute and was silently ignored,
// leaving the default 1024.)
// CHECK: aie.buffer @a1(%tile_3_3) {address = 2048 : i32, mem_bank = 0 : i32} : memref<4xi8>
// CHECK: aie.buffer @b2(%tile_3_3) {address = 2052 : i32, aligned = false, mem_bank = 0 : i32} : memref<12xi8>
// CHECK: aie.buffer @b3(%tile_3_3) {address = 2080 : i32, mem_bank = 0 : i32} : memref<4xi8>
module @test {
  aie.device(xcvc1902) {
    %0 = aie.tile(3, 3)
    aie.buffer @a(%0) : memref<16xi8>
    aie.buffer @b(%0) : memref<512xi32>
    aie.buffer @c(%0) : memref<16xi16>
    %3 = aie.tile(4, 4)
    aie.buffer @buffer_4(%3) : memref<500xi32>
    aie.core(%0) {
      aie.end
    }
    aie.core(%3) {
      aie.end
    }
  }
}

// -----

module @test_align{
  aie.device(npu2) {
    %0 = aie.tile(3, 3)
    aie.buffer @a1(%0) { address = 2048 : i32} : memref<4xi8>
    aie.buffer @b2(%0) { mem_bank = 0 : i32, aligned=false} : memref<12xi8>
    aie.buffer @b3(%0) { mem_bank = 0 : i32} : memref<4xi8>
    aie.core(%0) {
      aie.end
    }{stack_size = 2048 : i32}

  }

}
