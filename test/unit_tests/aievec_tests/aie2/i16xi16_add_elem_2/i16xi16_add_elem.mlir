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
  func.func @dut(%arg0: memref<1024xi16>, %arg1: memref<1024xi16>, %arg2: memref<1024xi32>) {
    %c0_i16 = arith.constant 0 : i16
    affine.for %arg3 = 0 to 1024 step 32 {
      %0 = vector.transfer_read %arg0[%arg3], %c0_i16 : memref<1024xi16>, vector<32xi16>
      %1 = arith.extsi %0 : vector<32xi16> to vector<32xi32>
      %2 = vector.transfer_read %arg1[%arg3], %c0_i16 : memref<1024xi16>, vector<32xi16>
      %3 = arith.extsi %2 : vector<32xi16> to vector<32xi32>
      %4 = arith.addi %1, %3 : vector<32xi32>
      vector.transfer_write %4, %arg2[%arg3] : vector<32xi32>, memref<1024xi32>
    }
    return
  }
}
