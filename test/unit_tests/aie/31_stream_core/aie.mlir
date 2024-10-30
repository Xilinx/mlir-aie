//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: aiesimulator, peano, !hsa

// RUN: %PYTHON aiecc.py --aiesim --no-xchesscc --xbridge %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf
// RUN: sh -c 'aie.mlir.prj/aiesim.sh; exit 0' | FileCheck %s

// CHECK: test start.
// CHECK: PASS!

module {
  %tile13 = aie.tile(1, 3)
  %tile23 = aie.tile(2, 3)1

  %buf13 = aie.buffer(%tile13) { sym_name = "a" } : memref<256xi32>
  %buf23 = aie.buffer(%tile23) { sym_name = "c" } : memref<256xi32>

  %lock13_3 = aie.lock(%tile13, 3) { sym_name = "input_lock" } // input buffer lock
  %lock23_7 = aie.lock(%tile23, 7) { sym_name = "output_lock" } // output buffer lock

  func.func private @do_mul(%A: memref<256xi32>) -> ()
  func.func private @do_mac(%A: memref<256xi32>) -> ()

  aie.flow(%tile13, Core : 0, %tile23, Core : 0)

  %core13 = aie.core(%tile13) {
    %0 = arith.constant 0 : i32
    %idx0 = arith.constant 3 : index
    aie.use_lock(%lock13_3, "Acquire", 1) // acquire for read(e.g. input ping)
    %val = memref.load %buf13[%idx0] : memref<256xi32>
    aie.put_stream(%0 : i32, %val : i32)
    aie.use_lock(%lock13_3, "Release", 0) // release for write
    aie.end
  }

  %core23 = aie.core(%tile23) {
    %0 = arith.constant 0 : i32
    %idx0 = arith.constant 3 : index
    aie.use_lock(%lock23_7, "Acquire", 0) // acquire for write
    %val = aie.get_stream(%0 : i32) : i32
    memref.store %val, %buf23[%idx0] : memref<256xi32>
    aie.use_lock(%lock23_7, "Release", 1) // release for read
    aie.end
  }

}
