//===- fallback_routine_error.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s
// CHECK:   error: Failed to allocate buffer: "f" with size: 512 bytes.
// CHECK: note: see current operation: %6 = "aie.buffer"(%0) <{sym_name = "f"}> : (index) -> memref<256xi16>
// CHECK: error: 'aie.tile' op All requested buffers don't fit in the available memory: Bank aware

// CHECK:   %tile12 = aie.tile(1, 2)
// CHECK:             ^
// CHECK: note: see current operation: %0 = "aie.tile"() <{col = 1 : i32, row = 2 : i32}> : () -> index
// CHECK: note: Current configuration of buffers in bank(s) : MemoryMap:
// CHECK: (no stack allocated)
// CHECK:         bank : 0        0x0-0x1FFF
// CHECK:                 b       : 0x0-0x1FFF    (8192 bytes)
// CHECK:         bank : 1        0x2000-0x3FFF
// CHECK:                 c       : 0x2000-0x3FFF         (8192 bytes)
// CHECK:         bank : 2        0x4000-0x5FFF
// CHECK:                 a       : 0x4000-0x4FFF         (4096 bytes)
// CHECK:                 e       : 0x5000-0x5FFF         (4096 bytes)
// CHECK:         bank : 3        0x6000-0x7FFF
// CHECK:                 d       : 0x6000-0x6FFF         (4096 bytes)
// CHECK:                 act_3_4_buff_0  : 0x7000-0x73FF         (1024 bytes)
// CHECK:                 act_3_4_buff_1  : 0x7400-0x77FF         (1024 bytes)
// CHECK:                 act_3_4_buff_2  : 0x7800-0x7BFF         (1024 bytes)
// CHECK:                 act_3_4_buff_3  : 0x7C00-0x7FFF         (1024 bytes)

// CHECK: error: 'aie.tile' op allocated buffers exceeded available memory: Sequential
// CHECK: (no stack allocated)

// CHECK:   %tile12 = aie.tile(1, 2)
// CHECK:             ^
// CHECK: note: see current operation: %0 = "aie.tile"() <{col = 1 : i32, row = 2 : i32}> : () -> index
// CHECK: note: MemoryMap:
// CHECK:         b       : 0x0-0x1FFF    (8192 bytes)
// CHECK:         c       : 0x2000-0x3FFF         (8192 bytes)
// CHECK:         a       : 0x4000-0x4FFF         (4096 bytes)
// CHECK:         d       : 0x5000-0x5FFF         (4096 bytes)
// CHECK:         e       : 0x6000-0x6FFF         (4096 bytes)
// CHECK:         act_3_4_buff_0  : 0x7000-0x73FF         (1024 bytes)
// CHECK:         act_3_4_buff_1  : 0x7400-0x77FF         (1024 bytes)
// CHECK:         act_3_4_buff_2  : 0x7800-0x7BFF         (1024 bytes)
// CHECK:         act_3_4_buff_3  : 0x7C00-0x7FFF         (1024 bytes)
// CHECK:         f       : 0x8000-0x81FF         (512 bytes)

module @test {
 aie.device(xcvc1902) {
  %tile12 = aie.tile(1, 2)
  %1 = aie.buffer(%tile12) { sym_name = "a" } : memref<1024xi32>  //4096 bytes
  %2 = aie.buffer(%tile12) { sym_name = "b" } : memref<2048xi32>  //8192 bytes
  %3 = aie.buffer(%tile12) { sym_name = "c" } : memref<2048xi32>  //8192 bytes
  %4 = aie.buffer(%tile12) { sym_name = "d" } : memref<1024xi32>  //4096 bytes
  %5 = aie.buffer(%tile12) { sym_name = "e" } : memref<1024xi32>  //4096 bytes
  %6 = aie.buffer(%tile12) { sym_name = "f" } : memref<256xi16>   //32 bytes
  %tile13 = aie.tile(1, 3)
  aie.objectfifo @act_3_4(%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<256xi32>> //4x1024 bytes
 }
}