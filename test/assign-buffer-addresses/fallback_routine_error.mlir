//===- fallback_routine_error.mlir ---------------------------------------------*- MLIR -*-===//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s
// CHECK:   warning: Failed to allocate buffer: "act_3_4_buff_2" with size: 2048 bytes.
// CHECK: note: see current operation: %act_3_4_buff_2 = aie.buffer(%tile_1_2) {sym_name = "act_3_4_buff_2"} : memref<512xi32>
// CHECK: warning:  Not all requested buffers fit in the available memory.
// CHECK:   %tile12 = aie.tile(1, 2)
// CHECK: warning: Bank-aware allocation failed, trying basic sequential allocation.
// CHECK: error: 'aie.tile' op allocated buffers exceeded available memory
// CHECK: (no stack allocated)
// CHECK:   %tile12 = aie.tile(1, 2)
// CHECK: note: see current operation: %0 = "aie.tile"() <{col = 1 : i32, row = 2 : i32}> : () -> index 
// CHECK: MemoryMap: 
// CHECK:   b : 0x0-0x1FFF (8192 bytes) 
// CHECK:   c : 0x2000-0x3FFF (8192 bytes) 
// CHECK:   a : 0x4000-0x4FFF (4096 bytes) 
// CHECK:   d : 0x5000-0x5FFF (4096 bytes) 
// CHECK: error: 'aie.tile' op Basic sequential allocation also failed.

module @test {
 aie.device(xcvc1902) {
  %tile12 = aie.tile(1, 2)
  %1 = aie.buffer(%tile12) { sym_name = "a" } : memref<1024xi32>  //8192 bytes
  %2 = aie.buffer(%tile12) { sym_name = "b" } : memref<2048xi32>  //8192 bytes
  %3 = aie.buffer(%tile12) { sym_name = "c" } : memref<2048xi32>  //8192 bytes
  %4 = aie.buffer(%tile12) { sym_name = "d" } : memref<1024xi32>  //4096 bytes
  %5 = aie.buffer(%tile12) { sym_name = "e" } : memref<1024xi32>  //4096 bytes
  %6 = aie.buffer(%tile12) { sym_name = "f" } : memref<256xi16>   //32 bytes
  %tile13 = aie.tile(1, 3)
  aie.objectfifo @act_3_4(%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<512xi32>> //4x1024 bytes
 }
}