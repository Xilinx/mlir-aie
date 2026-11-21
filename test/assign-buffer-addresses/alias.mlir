//===- alias.mlir ----------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses %s | aie-translate --aie-generate-ldscript --tilecol=0 --tilerow=2 | FileCheck %s --check-prefix=LD

// An alias takes no memory of its own: the allocator places the buffer it
// names, then gives it that buffer's address plus its offset. The linker
// script names it at that address without moving past the buffer it lies in.

// CHECK-LABEL: module @placed_with_its_buffer
// CHECK:       aie.buffer(%{{.*}}) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "pool"} : memref<96xbf16>
// CHECK:       aie.buffer(%{{.*}}) {address = 1088 : i32, alias_of = @pool, alias_offset = 64 : i32, sym_name = "pool_at_32"} : memref<32xbf16>
// CHECK:       aie.buffer(%{{.*}}) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "next"} : memref<16xi32>
// LD:          . = 0x70400;
// LD-NEXT:     pool = .;
// LD-NEXT:     . += 0xC0;
// LD-NEXT:     pool_at_32 = 0x70440;
// LD-NEXT:     . = 0x74000;
// LD-NEXT:     next = .;
module @placed_with_its_buffer {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    %pool = aie.buffer(%t) {sym_name = "pool"} : memref<96xbf16>
    %mid = aie.buffer(%t) {sym_name = "pool_at_32", alias_of = @pool, alias_offset = 64 : i32} : memref<32xbf16>
    %next = aie.buffer(%t) {sym_name = "next"} : memref<16xi32>
    aie.core(%t) {
      aie.end
    }
  }
}
