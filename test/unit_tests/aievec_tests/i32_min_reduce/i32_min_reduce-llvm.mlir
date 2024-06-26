// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024, Advanced Micro Devices, Inc.

// REQUIRES: valid_xchess_license
// REQUIRES: peano
// RUN: mkdir -p %t/data; cd %t
// RUN: aie-opt %s %vector-to-llvmir% -o llvmir.mlir
// RUN: aie-translate llvmir.mlir %llvmir-to-ll% -o dut.ll
// RUN: %PEANO_INSTALL_DIR/bin/clang %clang_aie2_args -c dut.ll -o dut.o
// RUN: xchesscc_wrapper %xchesscc_aie2_args -DTO_LLVM +w work +o work -I%S -I. %S/testbench.cc dut.o
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work/a.out" >& xca_udm_dbg.stdout
// RUN: FileCheck --input-file=./xca_udm_dbg.stdout %s
// CHECK: TEST PASSED

module {
  func.func @dut(%arg0: memref<1024xi32>, %arg1: memref<i32>) {
    memref.assume_alignment %arg0, 32 : memref<1024xi32>
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<2147483647> : vector<16xi32>
    %0 = affine.for %arg2 = 0 to 1024 step 16 iter_args(%arg3 = %cst) -> (vector<16xi32>) {
      %2 = vector.transfer_read %arg0[%arg2], %c0_i32 : memref<1024xi32>, vector<16xi32>
      %3 = arith.minsi %arg3, %2 : vector<16xi32>
      affine.yield %3 : vector<16xi32>
    }
    %1 = vector.reduction <minsi>, %0 : vector<16xi32> into i32
    affine.store %1, %arg1[] : memref<i32>
    return
  }
}
