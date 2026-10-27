//===- bank_aware_alloc_spans_banks.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Buffer "a" is 16384 bytes, larger than one 8192-byte bank on this device.
// Bank-aware allocation spreads traffic over banks; it does not bound how large
// a buffer may be. "a" therefore spans banks 0 and 1. It lands bank-aligned
// (address 0), the placement touching the fewest banks for its size, and the
// smaller buffers still spread round-robin over the remaining banks.

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s
// CHECK-NOT: warning
// CHECK:   module @test {
// CHECK:     aie.device(xcvc1902) {
// CHECK:       %tile_1_2 = aie.tile(1, 2)
// CHECK:       aie.buffer @act_3_4_buff_0(%tile_1_2) {address = 16384 : i32, mem_bank = 2 : i32} : memref<8xi32>
// CHECK:       aie.buffer @act_3_4_buff_1(%tile_1_2) {address = 24576 : i32, mem_bank = 3 : i32} : memref<8xi32>
// CHECK:       aie.buffer @act_3_4_buff_2(%tile_1_2) {address = 16416 : i32, mem_bank = 2 : i32} : memref<8xi32>
// CHECK:       aie.buffer @act_3_4_buff_3(%tile_1_2) {address = 24608 : i32, mem_bank = 3 : i32} : memref<8xi32>
// CHECK:       aie.lock @act_3_4_lock_0(%tile_1_2) {init = 0 : i32}
// CHECK:       aie.lock @act_3_4_lock_1(%tile_1_2) {init = 0 : i32}
// CHECK:       aie.lock @act_3_4_lock_2(%tile_1_2) {init = 0 : i32}
// CHECK:       aie.lock @act_3_4_lock_3(%tile_1_2) {init = 0 : i32}
// CHECK:       aie.buffer @a(%tile_1_2) {address = 0 : i32, mem_bank = 0 : i32} : memref<4096xi32>
// CHECK:       aie.buffer @b(%tile_1_2) {address = 16448 : i32, mem_bank = 2 : i32} : memref<16xi16>
// CHECK:       %tile_1_3 = aie.tile(1, 3)
// CHECK:     }
// CHECK:   }

module @test {
 aie.device(xcvc1902) {
  %tile12 = aie.tile(1, 2)
  aie.buffer @a(%tile12) : memref<4096xi32>  //16384 bytes
  aie.buffer @b(%tile12) : memref<16xi16> //32 bytes
  %tile13 = aie.tile(1, 3)
  aie.objectfifo @act_3_4(%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<8xi32>> //4x1 bytes
 }
}
