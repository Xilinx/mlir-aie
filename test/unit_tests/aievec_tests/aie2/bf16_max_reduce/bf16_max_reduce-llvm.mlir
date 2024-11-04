// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024, Advanced Micro Devices, Inc.

// REQUIRES: valid_xchess_license
// REQUIRES: peano, peano_and_chess
// RUN: mkdir -p %t/data; cd %t
// RUN: aie-opt %s %vector-to-llvmir% -o llvmir.mlir
// RUN: aie-translate llvmir.mlir %llvmir-to-ll% -o dut.ll
// RUN: %PEANO_INSTALL_DIR/bin/clang %clang_aie2_args -c dut.ll -o dut.o
// RUN: xchesscc_wrapper %xchesscc_aie2_args -DTO_LLVM +w work +o work -I%S -I. %S/testbench.cc dut.o
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../../profiling.tcl ./work/a.out" >& xca_udm_dbg.stdout
// RUN: FileCheck --input-file=./xca_udm_dbg.stdout %s
// CHECK: TEST PASSED

module {
  func.func @dut(%arg0: memref<1024xbf16>, %arg1: memref<bf16>) {
    memref.assume_alignment %arg0, 32 : memref<1024xbf16>
    %cst_0 = arith.constant dense<0xFF80> : vector<32xbf16>
    %0 = affine.for %arg2 = 0 to 1024 step 32 iter_args(%arg3 = %cst_0) -> (vector<32xbf16>) {
      %cst_1 = arith.constant 0.000000e+00 : bf16
      %3 = vector.transfer_read %arg0[%arg2], %cst_1 : memref<1024xbf16>, vector<32xbf16>
      %4 = arith.maximumf %arg3, %3 : vector<32xbf16>
      affine.yield %4 : vector<32xbf16>
    }
    %1 = vector.reduction <maximumf>, %0 : vector<32xbf16> into bf16
    affine.store %1, %arg1[] : memref<bf16>
    return
  }
}
