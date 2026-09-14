//===- prebaked_elf_data_region.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core that carries an `elf_file` is already linked, and its .data and .bss
// sit where that link put them. Nothing reads those addresses back out of the
// ELF, so a buffer at a fixed address is what declares them to the allocator.

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck %s

// "prebaked_data" covers [8192, 12288), so "scratch" goes elsewhere and the
// allocator keeps every buffer clear of the bytes the earlier link chose.
// CHECK-LABEL: module @pin_declares_prebaked_memory
// CHECK: %prebaked_data = aie.buffer(%tile_0_3) {address = 8192 : i32, mem_bank = 0 : i32, sym_name = "prebaked_data"} : memref<4096xi8>
// CHECK: %scratch = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "scratch"} : memref<8192xi8>
module @pin_declares_prebaked_memory {
  aie.device(npu2) {
    %tile_0_3 = aie.tile(0, 3)
    %prebaked_data = aie.buffer(%tile_0_3) {sym_name = "prebaked_data", address = 8192 : i32} : memref<4096xi8>
    %scratch = aie.buffer(%tile_0_3) {sym_name = "scratch"} : memref<8192xi8>
    %core_0_3 = aie.core(%tile_0_3) {
      aie.end
    } {elf_file = "prebaked_0_3.elf", stack_size = 1024 : i32}
  }
}
