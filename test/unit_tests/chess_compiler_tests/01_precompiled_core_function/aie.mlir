//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aiesimulator, valid_xchess_license, !hsa
// RUN: xchesscc_wrapper aie -c %S/kernel.cc
// RUN: %PYTHON aiecc.py --aiesim --xbridge --xchesscc %s %test_lib_flags %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf
// RUN: aie.mlir.prj/aiesim.sh | FileCheck %s

// CHECK: test start.
// CHECK: PASS!

module @test_chesss_01_precompiled_core_function {
aie.device(xcvc1902) {

  %tile13 = aie.tile(1, 3)

  %buf13_0 = aie.buffer(%tile13) { sym_name = "a" } : memref<256xi32>
  %buf13_1 = aie.buffer(%tile13) { sym_name = "b" } : memref<256xi32>

  %lock13_3 = aie.lock(%tile13, 3) { sym_name = "input_lock" }
  %lock13_5 = aie.lock(%tile13, 5) { sym_name = "output_lock" }

  func.func private @func(%A: memref<256xi32>, %B: memref<256xi32>) -> () attributes {link_with = "kernel.o"}

  %core13 = aie.core(%tile13) {
    %c1_ul0 = arith.constant 1 : i32
    aie.use_lock(%lock13_3, "Acquire", %c1_ul0) // acquire for read(e.g. input ping)
    %c0_ul1 = arith.constant 0 : i32
    aie.use_lock(%lock13_5, "Acquire", %c0_ul1) // acquire for write
    func.call @func(%buf13_0, %buf13_1) : (memref<256xi32>, memref<256xi32>) -> ()
    %c0_ul2 = arith.constant 0 : i32
    aie.use_lock(%lock13_3, "Release", %c0_ul2) // release for write
    %c1_ul3 = arith.constant 1 : i32
    aie.use_lock(%lock13_5, "Release", %c1_ul3) // release for read
    aie.end
  }

}
}
