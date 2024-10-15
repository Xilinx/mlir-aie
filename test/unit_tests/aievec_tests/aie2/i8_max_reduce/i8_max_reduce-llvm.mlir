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
  func.func @dut(%arg0: memref<1024xi8>, %arg1: memref<i8>) {
    memref.assume_alignment %arg0, 32 : memref<1024xi8>
    %c0_i8 = arith.constant 0 : i8
    %cst = arith.constant dense<-128> : vector<64xi8>
    %0 = affine.for %arg2 = 0 to 1024 step 64 iter_args(%arg3 = %cst) -> (vector<64xi8>) {
      %2 = vector.transfer_read %arg0[%arg2], %c0_i8 : memref<1024xi8>, vector<64xi8>
      %3 = arith.maxsi %arg3, %2 : vector<64xi8>
      affine.yield %3 : vector<64xi8>
    }
    %1 = vector.reduction <maxsi>, %0 : vector<64xi8> into i8
    affine.store %1, %arg1[] : memref<i8>
    return
  }
}
