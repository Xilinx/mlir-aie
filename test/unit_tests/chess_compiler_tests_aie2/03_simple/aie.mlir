//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aiesimulator, valid_xchess_license
// RUN: %aiecc --get-aiesim --xchesscc --xbridge %s %test_lib_flags -- %S/test.cpp
// RUN: sh -c './aie.mlir.prj/aiesim.sh; exit 0' | FileCheck %s

// CHECK: AIE2 ISS
// CHECK: test start.
// CHECK: PASS!

module @test04_shared_memory {
  aie.device(xcve2802) {
    %tile13 = aie.tile(1, 3)
    %tile14 = aie.tile(1, 4)

    aie.buffer @a(%tile13) : memref<256xi32>
    aie.buffer @b(%tile13) : memref<256xi32>
    aie.buffer @c(%tile14) : memref<256xi32>

    aie.lock @input_lock(%tile13, 3) // input buffer lock
    aie.lock @output_lock(%tile13, 5) // output buffer lock

    %core13 = aie.core(%tile13) {
      %c1_ul0 = arith.constant 1 : i32
      aie.use_lock(@input_lock, AcquireGreaterEqual, %c1_ul0)
      %idx1 = arith.constant 3 : index
      %buf13_0 = aie.get_buffer @a : memref<256xi32>
      %val1 = memref.load %buf13_0[%idx1] : memref<256xi32>
      %2    = arith.addi %val1, %val1 : i32
      %3 = arith.addi %2, %val1 : i32
      %4 = arith.addi %3, %val1 : i32
      %5 = arith.addi %4, %val1 : i32
      %idx2 = arith.constant 5 : index
      %buf13_1 = aie.get_buffer @b : memref<256xi32>
      memref.store %5, %buf13_1[%idx2] : memref<256xi32>
      %c1_ul1 = arith.constant 1 : i32
      aie.use_lock(@output_lock, Release, %c1_ul1)
      aie.end
    }
  }
}
