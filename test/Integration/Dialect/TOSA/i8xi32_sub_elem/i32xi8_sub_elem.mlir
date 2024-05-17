// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023, Advanced Micro Devices, Inc.

// REQUIRES: valid_xchess_license
// RUN: mkdir -p %t/data
// RUN: aie-opt %s %tosa-to-linalg% | aie-opt %linalg-to-vector-v32% --convert-vector-to-aievec="aie-target=aieml" -lower-affine -o %t/aievec.mlir
// RUN: aie-translate %t/aievec.mlir -aieml=true --aievec-to-cpp -o %t/dut.cc
// RUN: cd %t; xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. -c dut.cc -o dut.cc.o
// RUN: cd %t; xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. %S/testbench.cc %t/work/dut.cc.o
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work/a.out" >& xca_udm_dbg.stdout
// RUN: FileCheck --input-file=%t/xca_udm_dbg.stdout %s
// CHECK: TEST PASSED

module {
  func.func @dut(%arg0: tensor<1024xi8>, %arg1: tensor<1024xi32>) -> (tensor<1024xi32>) {
    %0 = "tosa.const"() {value = dense<0> : tensor<1024xi32>} : () -> tensor<1024xi32>
    %1 = "tosa.cast"(%arg0) : (tensor<1024xi8>) -> tensor<1024xi32>
    %2 = "tosa.sub"(%arg1, %1) : (tensor<1024xi32>, tensor<1024xi32>)  -> (tensor<1024xi32>)
    %3 = "tosa.sub"(%0, %2) : (tensor<1024xi32>, tensor<1024xi32>)  -> (tensor<1024xi32>)
    return %3 : tensor<1024xi32>
  }
}

