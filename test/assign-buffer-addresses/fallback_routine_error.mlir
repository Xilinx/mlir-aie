//===- fallback_routine_error.mlir ---------------------------------------------*- MLIR -*-===//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s
// CHECK:   error: Failed to allocate buffer: "act_3_4_buff_2" with size: 2048 bytes.
// CHECK: note: see current operation: %10 = "aie.buffer"(%0) <{sym_name = "act_3_4_buff_2"}> : (index) -> memref<512xi32>
// CHECK: error: 'aie.tile' op All requested buffers don't fit in the available memory: Bank aware

// CHECK:   %tile12 = aie.tile(1, 2)
// CHECK:             ^

// CHECK: error: 'aie.tile' op allocated buffers exceeded available memory: Sequential
// CHECK: (no stack allocated)

// CHECK:   %tile12 = aie.tile(1, 2)
// CHECK:             ^
// CHECK: note: see current operation: %0 = "aie.tile"() <{col = 1 : i32, row = 2 : i32}> : () -> index

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