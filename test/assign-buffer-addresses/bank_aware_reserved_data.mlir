//===- bank_aware_reserved_data.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core's own compiled sections (.data, .rodata, .bss) share data memory with
// the buffers placed here, and the generated linker script grants the core
// compiler one region.
//
// A core that declares no `data_size` gets the largest gap the buffers leave. A
// core that declares one gets a `core_data` buffer of that size, which the
// allocator places among the rest. The three cases below are: no reservation, a
// reservation large enough to govern the layout, and a small reservation, which
// leaves the buffers free to spread across banks.

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck %s

// With no reservation the three buffers pack head to tail, which leaves 52224
// contiguous bytes above them. A spread over banks 0-2 would leave one bank's
// worth, 28672.
// CHECK-LABEL: module @no_reservation_packs
// CHECK-NOT: core_data
// CHECK: %a = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "a"} : memref<4096xi8>
// CHECK: %b = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "b"} : memref<4096xi8>
// CHECK: %c = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "c"} : memref<4096xi8>
module @no_reservation_packs {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %a = aie.buffer(%tile_0_2) {sym_name = "a"} : memref<4096xi8>
    %b = aie.buffer(%tile_0_2) {sym_name = "b"} : memref<4096xi8>
    %c = aie.buffer(%tile_0_2) {sym_name = "c"} : memref<4096xi8>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}

// -----

// The reservation is the largest extent on the tile, so placement takes it
// first and puts it at the bottom of memory. The three buffers go above it.
// CHECK-LABEL: module @reservation_carves_out_a_block
// CHECK: %a = aie.buffer(%tile_0_2) {address = 41024 : i32, mem_bank = 2 : i32, sym_name = "a"} : memref<4096xi8>
// CHECK: %b = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "b"} : memref<4096xi8>
// CHECK: %c = aie.buffer(%tile_0_2) {address = 53248 : i32, mem_bank = 3 : i32, sym_name = "c"} : memref<4096xi8>
// CHECK: %core_data_0_2 = aie.buffer(%tile_0_2) {address = 1024 : i32, core_data, mem_bank = 0 : i32, sym_name = "core_data_0_2"} : memref<40000xi8>
// The grant equals the request: the buffers take everything above the reserved
// block.
module @reservation_carves_out_a_block {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %a = aie.buffer(%tile_0_2) {sym_name = "a"} : memref<4096xi8>
    %b = aie.buffer(%tile_0_2) {sym_name = "b"} : memref<4096xi8>
    %c = aie.buffer(%tile_0_2) {sym_name = "c"} : memref<4096xi8>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, data_size = 40000 : i32}
  }
}

// -----

// The reservation holds an address of its own, so the largest free run only has
// to cover the buffers left to place. Beyond that bound the contiguity key
// ranks every candidate alike, and round-robin spreads the buffers one per
// bank.
// CHECK-LABEL: module @a_small_reservation_frees_the_buffers_to_spread
// CHECK: %a = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "a"} : memref<4096xi8>
// CHECK: %b = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "b"} : memref<4096xi8>
// CHECK: %c = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "c"} : memref<4096xi8>
// CHECK: %core_data_0_2 = aie.buffer(%tile_0_2) {address = 1024 : i32, core_data, mem_bank = 0 : i32, sym_name = "core_data_0_2"} : memref<8192xi8>
module @a_small_reservation_frees_the_buffers_to_spread {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %a = aie.buffer(%tile_0_2) {sym_name = "a"} : memref<4096xi8>
    %b = aie.buffer(%tile_0_2) {sym_name = "b"} : memref<4096xi8>
    %c = aie.buffer(%tile_0_2) {sym_name = "c"} : memref<4096xi8>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, data_size = 8192 : i32}
  }
}
