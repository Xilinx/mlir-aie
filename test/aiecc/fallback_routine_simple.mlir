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
// RUN: %aiecc -v --get-input-with-addresses %s 2>&1 | FileCheck %s

// Buffer "a" is 16384 bytes and asks for bank 1, which is only 8192 bytes on
// this device. An explicit mem_bank is a hard constraint, so bank-aware
// allocation cannot satisfy it and falls back to the basic sequential scheme,
// which ignores mem_bank. That fallback is what makes the pass emit the
// diagnostics this test is looking for.
//
// "a" previously carried no mem_bank and merely being larger than one bank was
// enough to defeat bank-aware allocation. That is no longer so: a buffer that
// fits in no single bank now straddles bank boundaries rather than failing.
// CHECK: error: 'aie.buffer' op would override existing mem_bank
// CHECK: warning: Bank-aware allocation failed, trying basic sequential allocation.

module @test {
 aie.device(xcvc1902) {
  %tile12 = aie.tile(1, 2)
  %1 = aie.buffer(%tile12) { sym_name = "a", mem_bank = 1 : i32 } : memref<4096xi32>  //16384 bytes
  %b1 = aie.buffer(%tile12) { sym_name = "b" } : memref<16xi16> //32 bytes
  %tile13 = aie.tile(1, 3)
  aie.objectfifo @act_3_4(%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<8xi32>> //4x1 bytes
 }
}
