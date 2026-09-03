//===- bank_aware_prealloc_order.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Whether a buffer can be placed must not depend on the order the allocator
// visits buffers in. A per-bank watermark makes it depend on that order: a
// fixed-address buffer high in a bank moves the bank's cursor to its end, and a
// later buffer requesting the same bank is then rejected, even though the space
// below the pin is free and large enough.
//
// Both modules below describe the same layout and differ only in the order the
// two buffers appear. Both place "wants_bank1" in the hole at [16384, 24576),
// below the buffer pinned at 24576.

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck %s

// CHECK-LABEL: module @pinned_declared_first
// CHECK: %wants_bank1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "wants_bank1"} : memref<4096xbf16>
module @pinned_declared_first {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %pinned_hi = aie.buffer(%tile_0_2) {address = 24576 : i32, sym_name = "pinned_hi"} : memref<4096xbf16>
    %wants_bank1 = aie.buffer(%tile_0_2) {mem_bank = 1 : i32, sym_name = "wants_bank1"} : memref<4096xbf16>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}

// -----

// CHECK-LABEL: module @pinned_declared_last
// CHECK: %wants_bank1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "wants_bank1"} : memref<4096xbf16>
module @pinned_declared_last {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %wants_bank1 = aie.buffer(%tile_0_2) {mem_bank = 1 : i32, sym_name = "wants_bank1"} : memref<4096xbf16>
    %pinned_hi = aie.buffer(%tile_0_2) {address = 24576 : i32, sym_name = "pinned_hi"} : memref<4096xbf16>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}
