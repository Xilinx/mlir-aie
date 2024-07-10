// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023, Advanced Micro Devices, Inc.

// REQUIRES: valid_xchess_license
// RUN: mkdir -p %t/data
// RUN: aie-opt %s %tosa-to-linalg% | aie-opt %linalg-to-vector-v16% --convert-vector-to-aievec="aie-target=aie2" -lower-affine -o %t/aievec.mlir
// RUN: aie-translate %t/aievec.mlir -aie2=true --aievec-to-cpp -o %t/dut.cc
// RUN: cd %t; xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. %S/testbench.cc dut.cc
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work/a.out" >& xca_udm_dbg.stdout
// RUN: FileCheck --input-file=%t/xca_udm_dbg.stdout %s
// CHECK: TEST PASSED

module {
  func.func @dut(%arg0: tensor<1024xbf16>, %arg1: tensor<1024xbf16>) -> (tensor<1024xbf16>) {
    %1 = "tosa.sub"(%arg0,%arg1) : (tensor<1024xbf16>, tensor<1024xbf16>)  -> (tensor<1024xbf16>)
    return %1 : tensor<1024xbf16>
  }
}