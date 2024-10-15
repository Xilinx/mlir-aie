//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: aiesimulator, valid_xchess_license, !hsa
// RUN: xchesscc_wrapper aie2 -c %S/kernel.cc
// RUN: %PYTHON aiecc.py -v --aiesim --xchesscc --xbridge --no-compile-host %s %test_lib_flags %S/test.cpp
// RUN: aie.mlir.prj/aiesim.sh | FileCheck %s

// CHECK: AIE2 ISS
// CHECK: test start.
// CHECK: PASS!

module @test_chesss_01_precompiled_core_function {
  aie.device(xcve2802) {
    %tile13 = aie.tile(1, 3)

    %buf13_0 = aie.buffer(%tile13) { sym_name = "a" } : memref<256xi32>
    %buf13_1 = aie.buffer(%tile13) { sym_name = "b" } : memref<256xi32>

    %lock13_3 = aie.lock(%tile13, 3) { sym_name = "input_lock" }
    %lock13_5 = aie.lock(%tile13, 5) { sym_name = "output_lock" }

    func.func private @func(%A: memref<256xi32>, %B: memref<256xi32>) -> ()

    %core13 = aie.core(%tile13) {
      aie.use_lock(%lock13_3, "Acquire", 1) // acquire for read(e.g. input ping)
      aie.use_lock(%lock13_5, "Acquire", 0) // acquire for write
      func.call @func(%buf13_0, %buf13_1) : (memref<256xi32>, memref<256xi32>) -> ()
      aie.use_lock(%lock13_3, "Release", 0) // release for write
      aie.use_lock(%lock13_5, "Release", 1) // release for read
      aie.end
    } { link_with="kernel.o" }
  }
}
