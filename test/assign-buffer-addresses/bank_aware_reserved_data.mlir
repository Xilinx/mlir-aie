//===- bank_aware_reserved_data.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core's own compiled sections (.data/.rodata/.bss) share data memory with
// the buffers placed here, and the generated linker script hands the core
// compiler exactly one region: the largest gap left between the stack and the
// buffers. Spreading buffers over banks chops that space up, so `reserved_data
// _size` tells the allocator how much contiguous room the core still needs.

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck %s

// Without a reservation the three buffers spread round-robin over banks 0-2,
// which is what limits DMA bank contention. The largest surviving gap is one
// bank, 16384 bytes.
// CHECK-LABEL: module @no_reservation_spreads
// CHECK: %a = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "a"} : memref<4096xi8>
// CHECK: %b = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "b"} : memref<4096xi8>
// CHECK: %c = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "c"} : memref<4096xi8>
module @no_reservation_spreads {
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

// Asking for 40000 contiguous bytes cannot be met while spreading, so the same
// three buffers pack from the bottom of bank 0 instead, leaving 52224 bytes in
// one run. Bank spreading is a performance preference; leaving the core room
// to link is a correctness constraint, so the constraint wins.
// CHECK-LABEL: module @reservation_packs_tightly
// CHECK: %a = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "a"} : memref<4096xi8>
// CHECK: %b = aie.buffer(%tile_0_2) {address = 5120 : i32, mem_bank = 0 : i32, sym_name = "b"} : memref<4096xi8>
// CHECK: %c = aie.buffer(%tile_0_2) {address = 9216 : i32, mem_bank = 0 : i32, sym_name = "c"} : memref<4096xi8>
module @reservation_packs_tightly {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %a = aie.buffer(%tile_0_2) {sym_name = "a"} : memref<4096xi8>
    %b = aie.buffer(%tile_0_2) {sym_name = "b"} : memref<4096xi8>
    %c = aie.buffer(%tile_0_2) {sym_name = "c"} : memref<4096xi8>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, reserved_data_size = 40000 : i32}
  }
}

// -----

// A reservation small enough to be met while still spreading leaves the
// spread placement alone: the constraint only costs bank parallelism when it
// actually has to.
// CHECK-LABEL: module @small_reservation_keeps_spread
// CHECK: %b = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "b"} : memref<4096xi8>
module @small_reservation_keeps_spread {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %a = aie.buffer(%tile_0_2) {sym_name = "a"} : memref<4096xi8>
    %b = aie.buffer(%tile_0_2) {sym_name = "b"} : memref<4096xi8>
    %c = aie.buffer(%tile_0_2) {sym_name = "c"} : memref<4096xi8>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, reserved_data_size = 8192 : i32}
  }
}
