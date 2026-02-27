//===- fallback_routine_simple.mlir ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// check that 'aiecc.py -v' prints pass diagnotics

// RUN: %python aiecc.py --no-compile -v %s 2>&1 | FileCheck %s

// CHECK: warning: Failed to allocate buffer: "a" with size: 16384 bytes.
// CHECK: warning: Not all requested buffers fit in the available memory.
// CHECK: note: Current configuration of buffers in bank(s) : MemoryMap:
// CHECK: (no stack allocated)
// CHECK:         bank : 0        0x0-0x1FFF
// CHECK:         bank : 1        0x2000-0x3FFF
// CHECK:         bank : 2        0x4000-0x5FFF
// CHECK:         bank : 3        0x6000-0x7FFF
// CHECK: warning: Bank-aware allocation failed, trying basic sequential allocation.

module @test {
 aie.device(xcvc1902) {
  %tile12 = aie.tile(1, 2)
  %1 = aie.buffer(%tile12) { sym_name = "a" } : memref<4096xi32>  //16384 bytes
  %b1 = aie.buffer(%tile12) { sym_name = "b" } : memref<16xi16> //32 bytes
  %tile13 = aie.tile(1, 3)
  aie.objectfifo @act_3_4(%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<8xi32>> //4x1 bytes
 }
}
