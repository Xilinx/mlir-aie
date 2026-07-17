//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aiesimulator, valid_xchess_license
// RUN: %PYTHON aiecc.py --aiesim --xchesscc --xbridge %s %test_lib_flags -- %S/test.cpp
// RUN: aie.mlir.prj/aiesim.sh | FileCheck %s

// CHECK: AIE2 ISS
// CHECK: test start.
// CHECK: PASS!

// XFAIL: *

module @test04_shared_memory {
  aie.device(xcve2802) {
    %tile13 = aie.tile(1, 3)
    %tile14 = aie.tile(1, 4)

    %buf13_0 = aie.buffer(%tile13) { sym_name = "a" } : memref<256xi32>
    %buf13_1 = aie.buffer(%tile13) { sym_name = "b" } : memref<256xi32>
    %buf14_0 = aie.buffer(%tile14) { sym_name = "c" } : memref<256xi32>

    %lock13_2 = aie.lock(%tile13, 2) { sym_name = "test_lock" } // test lock

    %lock13_3 = aie.lock(%tile13, 3) { sym_name = "input_write_lock", init = 1 : i32 } // input buffer lock
    %lock13_4 = aie.lock(%tile13, 4) { sym_name = "input_read_lock" } // input buffer lock
    %lock13_5 = aie.lock(%tile13, 5) { sym_name = "hidden_write_lock", init = 1 : i32 } // interbuffer lock
    %lock13_6 = aie.lock(%tile13, 6) { sym_name = "hidden_read_lock" } // interbuffer lock
    %lock14_7 = aie.lock(%tile14, 7) { sym_name = "output_write_lock", init = 1 : i32 } // output buffer lock
    %lock14_8 = aie.lock(%tile14, 8) { sym_name = "output_read_lock" } // output buffer lock

    %core13 = aie.core(%tile13) {
      %c1_ul0 = arith.constant 1 : i32
      aie.use_lock(%lock13_4, AcquireGreaterEqual, %c1_ul0) // acquire input for read(e.g. input ping)
      %c1_ul1 = arith.constant 1 : i32
      aie.use_lock(%lock13_5, AcquireGreaterEqual, %c1_ul1) // acquire input for write
      %idx1 = arith.constant 3 : index
      %val1 = memref.load %buf13_0[%idx1] : memref<256xi32>
      %2    = arith.addi %val1, %val1 : i32
      %3 = arith.addi %2, %val1 : i32
      %4 = arith.addi %3, %val1 : i32
      %5 = arith.addi %4, %val1 : i32
      %idx2 = arith.constant 5 : index
      memref.store %5, %buf13_1[%idx2] : memref<256xi32>
      %c1_ul2 = arith.constant 1 : i32
      aie.use_lock(%lock13_3, Release, %c1_ul2) // release input for write
      %c1_ul3 = arith.constant 1 : i32
      aie.use_lock(%lock13_6, Release, %c1_ul3) // release output for read
      aie.end
    }

    %core14 = aie.core(%tile14) {
      %c1_ul4 = arith.constant 1 : i32
      aie.use_lock(%lock13_6, AcquireGreaterEqual, %c1_ul4) // acquire input for read(e.g. input ping)
      %c1_ul5 = arith.constant 1 : i32
      aie.use_lock(%lock14_7, AcquireGreaterEqual, %c1_ul5) // acquire output for write
      %idx1 = arith.constant 5 : index
      %val1 = memref.load %buf13_1[%idx1] : memref<256xi32>
      %2    = arith.addi %val1, %val1 : i32
      %3 = arith.addi %2, %val1 : i32
      %4 = arith.addi %3, %val1 : i32
      %5 = arith.addi %4, %val1 : i32
      %idx2 = arith.constant 5 : index
      memref.store %5, %buf14_0[%idx2] : memref<256xi32>
      %c1_ul6 = arith.constant 1 : i32
      aie.use_lock(%lock13_5, Release, %c1_ul6) // release input for write
      %c1_ul7 = arith.constant 1 : i32
      aie.use_lock(%lock14_8, Release, %c1_ul7) // release output for read
      aie.end
    }
  }
}
