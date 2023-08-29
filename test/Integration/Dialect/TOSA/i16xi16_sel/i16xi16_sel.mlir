// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023, Advanced Micro Devices, Inc.

// REQUIRES: valid_xchess_license
// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg))" -o linalg.mlir
// RUN: mlir-opt linalg.mlir --linalg-fuse-elementwise-ops --eliminate-empty-tensors --empty-tensor-to-alloc-tensor --one-shot-bufferize="allow-return-allocs allow-unknown-ops bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --drop-equivalent-buffer-results --buffer-results-to-out-params --buffer-deallocation --canonicalize --cse --convert-linalg-to-affine-loops --affine-super-vectorize="virtual-vector-size=32" -o affine.mlir 
// RUN: aie-opt affine.mlir --convert-vector-to-aievec="aie-target=aieml" -lower-affine | aie-translate -aieml=true --aievec-to-cpp -o dut.cc
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. %S/testbench.cc dut.cc
// RUN: mkdir -p data
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work/a.out" >& xca_udm_dbg.stdout
// RUN: FileCheck --input-file=./xca_udm_dbg.stdout %s
// CHECK: TEST PASSED

module {
  func.func @dut(%arg0: tensor<1024xi16>, %arg1: tensor<1024xi16>) -> (tensor<1024xi16>) {
    %0 = "tosa.greater"(%arg0, %arg1) : (tensor<1024xi16>, tensor<1024xi16>) -> (tensor<1024xi1>)
    %1 = "tosa.select"(%0, %arg0, %arg1) : (tensor<1024xi1>, tensor<1024xi16>, tensor<1024xi16>) -> (tensor<1024xi16>)
    return %1 : tensor<1024xi16>
  }
}
