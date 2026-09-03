//===- bank_aware_reserved_data.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core's own compiled sections (.data, .rodata, .bss) share data memory with
// the buffers placed here, and the generated linker script grants the core
// compiler one region: the largest gap left behind.
//
// Placement scores that gap directly, so the room exists without a request.
// `reserved_data_size` sets a floor on top of that. The three cases below are:
// no floor, a floor large enough to dictate the layout, and a floor small
// enough to change nothing.

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck %s

// With no floor the three buffers pack head to tail, which leaves 52224
// contiguous bytes above them. A spread over banks 0-2 would leave one bank's
// worth, 28672.
// CHECK-LABEL: module @no_reservation_packs
// CHECK: %a = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "a"} : memref<4096xi8>
// CHECK: %b = aie.buffer(%tile_0_2) {address = 5120 : i32, mem_bank = 0 : i32, sym_name = "b"} : memref<4096xi8>
// CHECK: %c = aie.buffer(%tile_0_2) {address = 9216 : i32, mem_bank = 0 : i32, sym_name = "c"} : memref<4096xi8>
// CHECK: data_length = 52224 : i32, data_origin = 13312 : i32
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

// A floor large enough to dictate the layout is carved out of the bottom of
// memory before the unconstrained buffers are placed, and they go above it.
// The grant equals the floor, because no space remains above the buffers.
// CHECK-LABEL: module @reservation_carves_out_a_block
// CHECK: %a = aie.buffer(%tile_0_2) {address = 41024 : i32, mem_bank = 2 : i32, sym_name = "a"} : memref<4096xi8>
// CHECK: %b = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "b"} : memref<4096xi8>
// CHECK: %c = aie.buffer(%tile_0_2) {address = 53248 : i32, mem_bank = 3 : i32, sym_name = "c"} : memref<4096xi8>
// The grant equals the request: the buffers take everything above the reserved
// block.
// CHECK: data_length = 40000 : i32, data_origin = 1024 : i32
module @reservation_carves_out_a_block {
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

// A floor the layout already clears costs nothing: `a` sits flush above the
// reserved block, and the grant is the 40960 bytes that survive placement, not
// the 8192 requested.
// CHECK-LABEL: module @small_reservation_is_already_met
// CHECK: %a = aie.buffer(%tile_0_2) {address = 9216 : i32, mem_bank = 0 : i32, sym_name = "a"} : memref<4096xi8>
// CHECK: %b = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "b"} : memref<4096xi8>
// CHECK: %c = aie.buffer(%tile_0_2) {address = 20480 : i32, mem_bank = 1 : i32, sym_name = "c"} : memref<4096xi8>
// CHECK: data_length = 40960 : i32, data_origin = 24576 : i32
module @small_reservation_is_already_met {
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
