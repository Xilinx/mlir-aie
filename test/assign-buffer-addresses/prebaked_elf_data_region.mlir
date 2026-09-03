//===- prebaked_elf_data_region.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core that carries an `elf_file` is already linked, and its .data and .bss
// sit where that link put them. Nothing reads those addresses back out of the
// ELF, so a buffer at a fixed address is what declares them to the allocator.
//
// data_origin and data_length are outputs: the pass replaces the values below,
// so they cannot serve as the declaration.

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck %s

// "prebaked_data" covers [8192, 12288), so "scratch" goes elsewhere and the
// core's own region starts above both buffers.
// CHECK-LABEL: module @pin_declares_prebaked_memory
// CHECK: %prebaked_data = aie.buffer(%tile_0_3) {address = 8192 : i32, mem_bank = 0 : i32, sym_name = "prebaked_data"} : memref<4096xi8>
// CHECK: %scratch = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "scratch"} : memref<8192xi8>
// CHECK: data_length = 40960 : i32, data_origin = 24576 : i32, elf_file = "prebaked_0_3.elf"
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

// -----

// A hand-written data region is replaced, not honoured. 0x8000/0x400 goes in,
// the run above the stack comes out.
// CHECK-LABEL: module @data_region_is_an_output
// CHECK: data_length = 64512 : i32
// CHECK-SAME: data_origin = 1024 : i32
module @data_region_is_an_output {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, data_origin = 32768 : i32, data_length = 1024 : i32}
  }
}
