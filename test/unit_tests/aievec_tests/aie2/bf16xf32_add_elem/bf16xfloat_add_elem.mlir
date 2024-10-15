// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023, Advanced Micro Devices, Inc.

// REQUIRES: valid_xchess_license
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" -lower-affine | aie-translate -aie2=true --aievec-to-cpp -o dut.cc
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. -c dut.cc -o dut.o
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. %S/testbench.cc work/dut.o
// RUN: mkdir -p data
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../../profiling.tcl ./work/a.out" >& xca_udm_dbg.stdout
// RUN: FileCheck --input-file=./xca_udm_dbg.stdout %s
// CHECK: TEST PASSED

module {
  func.func @dut(%arg0: memref<1024xbf16>, %arg1: memref<1024xbf16>, %arg2: memref<1024xf32>) {
    affine.for %arg3 = 0 to 1024 step 16 {
      %cst = arith.constant 0.000000e+00 : bf16
      %0 = vector.transfer_read %arg0[%arg3], %cst : memref<1024xbf16>, vector<16xbf16>
      %1 = arith.extf %0 : vector<16xbf16> to vector<16xf32>
      %cst_0 = arith.constant 0.000000e+00 : bf16
      %2 = vector.transfer_read %arg1[%arg3], %cst_0 : memref<1024xbf16>, vector<16xbf16>
      %3 = arith.extf %2 : vector<16xbf16> to vector<16xf32>
      %4 = arith.addf %1, %3 : vector<16xf32>
      vector.transfer_write %4, %arg2[%arg3] : vector<16xf32>, memref<1024xf32>
    }
    return
  }
}
