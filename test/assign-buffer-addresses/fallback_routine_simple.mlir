//===- fallback_routine_simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s
// CHECK:   error: Failed to allocate buffer: "a" with size: 16384 bytes.
// CHECK:   %1 = aie.buffer(%tile12) { sym_name = "a" } : memref<4096xi32>  //16384 bytes
// CHECK:          ^
// CHECK:   error: 'aie.tile' op All requested buffers don't fit in the available memory: Bank aware

// CHECK:   %tile12 = aie.tile(1, 2)
// CHECK:             ^
// CHECK: note: Current configuration of buffers in bank(s) : MemoryMap:
// CHECK: (no stack allocated)
// CHECK:         bank : 0        0x0-0x1FFF
// CHECK:         bank : 1        0x2000-0x3FFF
// CHECK:         bank : 2        0x4000-0x5FFF
// CHECK:         bank : 3        0x6000-0x7FFF

// CHECK:   module @test {
// CHECK:     aie.device(xcvc1902) {
// CHECK:       memref.global "public" @act_3_4 : memref<8xi32>
// CHECK:       %tile_1_2 = aie.tile(1, 2)
// CHECK:       %a = aie.buffer(%tile_1_2) {address = 0 : i32, sym_name = "a"} : memref<4096xi32> 
// CHECK:       %b = aie.buffer(%tile_1_2) {address = 16384 : i32, sym_name = "b"} : memref<16xi16> 
// CHECK:       %tile_1_3 = aie.tile(1, 3)
// CHECK:       %act_3_4_buff_0 = aie.buffer(%tile_1_2) {address = 16416 : i32, sym_name = "act_3_4_buff_0"} : memref<8xi32> 
// CHECK:       %act_3_4_buff_1 = aie.buffer(%tile_1_2) {address = 16448 : i32, sym_name = "act_3_4_buff_1"} : memref<8xi32> 
// CHECK:       %act_3_4_buff_2 = aie.buffer(%tile_1_2) {address = 16480 : i32, sym_name = "act_3_4_buff_2"} : memref<8xi32> 
// CHECK:       %act_3_4_buff_3 = aie.buffer(%tile_1_2) {address = 16512 : i32, sym_name = "act_3_4_buff_3"} : memref<8xi32> 
// CHECK:       %act_3_4_lock_0 = aie.lock(%tile_1_2, 0) {init = 0 : i32, sym_name = "act_3_4_lock_0"}
// CHECK:       %act_3_4_lock_1 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "act_3_4_lock_1"}
// CHECK:       %act_3_4_lock_2 = aie.lock(%tile_1_2, 2) {init = 0 : i32, sym_name = "act_3_4_lock_2"}
// CHECK:       %act_3_4_lock_3 = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "act_3_4_lock_3"}
// CHECK:     }
// CHECK:   }

module @test {
 aie.device(xcvc1902) {
  %tile12 = aie.tile(1, 2)
  %1 = aie.buffer(%tile12) { sym_name = "a" } : memref<4096xi32>  //16384 bytes
  %b1 = aie.buffer(%tile12) { sym_name = "b" } : memref<16xi16> //32 bytes
  %tile13 = aie.tile(1, 3)
  aie.objectfifo @act_3_4(%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<8xi32>> //4x1 bytes
 }
}
