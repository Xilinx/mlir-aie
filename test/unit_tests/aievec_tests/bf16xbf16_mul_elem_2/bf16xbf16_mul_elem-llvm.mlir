// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023, Advanced Micro Devices, Inc.

// REQUIRES: valid_xchess_license
// REQUIRES: peano, peano_and_chess
// RUN: mkdir -p %t/data; cd %t
// RUN: aie-opt %s %vector-to-llvmir% -o llvmir.mlir
// RUN: aie-translate llvmir.mlir %llvmir-to-ll% -o dut.ll
// RUN: %PEANO_INSTALL_DIR/bin/clang %clang_aie2_args -c dut.ll -o dut.o
// RUN: xchesscc_wrapper %xchesscc_aie2_args -DTO_LLVM +w work +o work -I%S -I. %S/testbench.cc dut.o
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work/a.out" >& xca_udm_dbg.stdout
// RUN: FileCheck --input-file=./xca_udm_dbg.stdout %s
// CHECK: TEST PASSED

module {
  func.func @dut(%arg0: memref<1024xbf16>, %arg1: memref<1024xbf16>, %arg2: memref<1024xf32>) {
    memref.assume_alignment %arg0, 32 : memref<1024xbf16>
    memref.assume_alignment %arg1, 32 : memref<1024xbf16>
    memref.assume_alignment %arg2, 32 : memref<1024xf32>
    %cst = arith.constant 0.000000e+00 : bf16
    affine.for %arg3 = 0 to 1024 step 16 {
      %0 = vector.transfer_read %arg0[%arg3], %cst : memref<1024xbf16>, vector<16xbf16>
      %1 = arith.extf %0 : vector<16xbf16> to vector<16xf32>
      %2 = vector.transfer_read %arg1[%arg3], %cst : memref<1024xbf16>, vector<16xbf16>
      %3 = arith.extf %2 : vector<16xbf16> to vector<16xf32>
      %4 = arith.mulf %1, %3 : vector<16xf32>
      vector.transfer_write %4, %arg2[%arg3] : vector<16xf32>, memref<1024xf32>
    }
    return
  }
}
