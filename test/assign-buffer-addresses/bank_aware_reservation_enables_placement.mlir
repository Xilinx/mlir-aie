//===- bank_aware_reservation_enables_placement.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The allocator packs a core's static data only when the core declares
// data_size. Both modules hold the same five buffers on a tile of four
// 16384-byte banks.
//
// Without the declaration the allocator spreads the buffers one per bank and
// the largest run left over is 8192 bytes, so the link of a core needing 16384
// bytes fails. The declaration turns those bytes into a buffer the allocator
// places, which packs two buffers per bank and clears bank 3.

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses %s | FileCheck %s

// CHECK-LABEL: module @no_reservation
// CHECK: aie.buffer @b0(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32} : memref<8192xi8>
// CHECK: aie.buffer @b1(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32} : memref<8192xi8>
// CHECK: aie.buffer @b2(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32} : memref<8192xi8>
// CHECK: aie.buffer @b3(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32} : memref<8192xi8>
// CHECK: aie.buffer @b4(%tile_0_2) {address = 24576 : i32, mem_bank = 1 : i32} : memref<8192xi8>
module @no_reservation {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.buffer @b0(%t) : memref<8192xi8>
    aie.buffer @b1(%t) : memref<8192xi8>
    aie.buffer @b2(%t) : memref<8192xi8>
    aie.buffer @b3(%t) : memref<8192xi8>
    aie.buffer @b4(%t) : memref<8192xi8>
    %core = aie.core(%t) { aie.end } {stack_size = 1024 : i32}
  }
}

// -----

// CHECK-LABEL: module @reservation
// CHECK: aie.buffer @b0(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32} : memref<8192xi8>
// CHECK: aie.buffer @b1(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32} : memref<8192xi8>
// CHECK: aie.buffer @b2(%tile_0_2) {address = 24576 : i32, mem_bank = 1 : i32} : memref<8192xi8>
// CHECK: aie.buffer @b3(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32} : memref<8192xi8>
// CHECK: aie.buffer @b4(%tile_0_2) {address = 40960 : i32, mem_bank = 2 : i32} : memref<8192xi8>
// CHECK: aie.buffer @core_data_0_2(%tile_0_2) {address = 49152 : i32, core_data, mem_bank = 3 : i32} : memref<16384xi8>
module @reservation {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.buffer @b0(%t) : memref<8192xi8>
    aie.buffer @b1(%t) : memref<8192xi8>
    aie.buffer @b2(%t) : memref<8192xi8>
    aie.buffer @b3(%t) : memref<8192xi8>
    aie.buffer @b4(%t) : memref<8192xi8>
    %core = aie.core(%t) { aie.end } {stack_size = 1024 : i32, data_size = 16384 : i32}
  }
}
