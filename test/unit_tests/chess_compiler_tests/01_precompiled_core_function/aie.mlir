//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aiesimulator, valid_xchess_license
// RUN: xchesscc_wrapper aie -c %S/kernel.cc
// RUN: %aiecc --get-aiesim --xbridge --xchesscc %s %test_lib_flags -o test.elf -- %S/test.cpp
// RUN: %run_on_board ./test.elf
// RUN: ./aie.mlir.prj/aiesim.sh | FileCheck %s

// CHECK: test start.
// CHECK: PASS!

module @test_chesss_01_precompiled_core_function {
aie.device(xcvc1902) {

  %tile13 = aie.tile(1, 3)

  aie.buffer @a(%tile13) : memref<256xi32>
  aie.buffer @b(%tile13) : memref<256xi32>

  aie.lock @input_lock(%tile13, 3)
  aie.lock @output_lock(%tile13, 5)

  func.func private @func(%A: memref<256xi32>, %B: memref<256xi32>) -> () attributes {link_with = "kernel.o"}

  %core13 = aie.core(%tile13) {
    %c1_ul0 = arith.constant 1 : i32
    aie.use_lock(@input_lock, "Acquire", %c1_ul0) // acquire for read(e.g. input ping)
    %c0_ul1 = arith.constant 0 : i32
    aie.use_lock(@output_lock, "Acquire", %c0_ul1) // acquire for write
    %buf13_0 = aie.get_buffer @a : memref<256xi32>
    %buf13_1 = aie.get_buffer @b : memref<256xi32>
    func.call @func(%buf13_0, %buf13_1) : (memref<256xi32>, memref<256xi32>) -> ()
    %c0_ul2 = arith.constant 0 : i32
    aie.use_lock(@input_lock, "Release", %c0_ul2) // release for write
    %c1_ul3 = arith.constant 1 : i32
    aie.use_lock(@output_lock, "Release", %c1_ul3) // release for read
    aie.end
  }

}
}
