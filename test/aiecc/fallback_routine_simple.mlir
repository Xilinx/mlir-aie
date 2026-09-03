//===- fallback_routine_simple.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// check that 'aiecc -v' prints pass diagnostics

// The front-end (place/allocate) only runs if some artifact roots it; request
// input_with_addresses so the buffer-allocation diagnostics below are emitted
// without invoking any core compiler.
// RUN: not %aiecc -v --skip-objectFifo-verify --get-input-with-addresses %s 2>&1 | FileCheck %s

// Buffer "a" is 16384 bytes and requests bank 1, which is only 8192 bytes on
// this device, so the tile cannot honour the pin. The pass reports an error and
// does not retry: basic sequential allocation ignores mem_bank and would place
// "a" in a bank the design never requested.
// CHECK: error: 'aie.buffer' op requires 16384 bytes, which cannot fit in bank 1 (8192 bytes total)
// CHECK: error: 'aie.tile' op Bank-aware allocation failed.

module @test {
 aie.device(xcvc1902) {
  %tile12 = aie.tile(1, 2)
  %1 = aie.buffer(%tile12) { sym_name = "a", mem_bank = 1 : i32 } : memref<4096xi32>  //16384 bytes
  %b1 = aie.buffer(%tile12) { sym_name = "b" } : memref<16xi16> //32 bytes
  %tile13 = aie.tile(1, 3)
  aie.objectfifo @act_3_4(%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<8xi32>> //4x1 bytes
 }
}
