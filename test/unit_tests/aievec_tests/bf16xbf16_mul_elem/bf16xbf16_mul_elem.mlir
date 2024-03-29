// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023, Advanced Micro Devices, Inc.

// REQUIRES: valid_xchess_license

// To-CPP flow
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aieml" -lower-affine | aie-translate -aieml=true --aievec-to-cpp -o dut.cc
// RUN: xchesscc_wrapper %xchesscc_aie2_args +w work_cpp +o work_cpp -I%S -I. -c dut.cc -o dut.o
// RUN: xchesscc_wrapper %xchesscc_aie2_args -DTO_CPP -DDATA_DIR=data_cpp +w work_cpp +o work_cpp -I%S -I. %S/testbench.cc work_cpp/dut.o
// RUN: mkdir -p data_cpp
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work_cpp/a.out" >& xca_udm_dbg.cpp.stdout
// RUN: FileCheck --input-file=./xca_udm_dbg.cpp.stdout %s

// To-Vector-LLVM flow
// RUN: aie-opt %s %vector-to-aievec% -o aievec.mlir
// RUN: aie-opt %s %vector-to-llvmir% -o llvmir.vector.mlir
// RUN: aie-translate llvmir.vector.mlir %llvmir-to-ll% -o dut.vector.ll
// RUN: %clang %clang_aie2_args -c dut.vector.ll -o dut.vector.o
// RUN: xchesscc_wrapper %xchesscc_aie2_args -DTO_LLVM -DDATA_DIR=data_llvm +w work_llvm +o work_llvm -I%S -I. %S/testbench.cc dut.vector.o
// RUN: mkdir -p data_llvm
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work_llvm/a.out" >& xca_udm_dbg.llvm.stdout
// RUN: FileCheck --input-file=./xca_udm_dbg.llvm.stdout %s

// CHECK: TEST PASSED

module {
  func.func @dut(%arg0: memref<1024xbf16>, %arg1: memref<1024xbf16>, %arg2: memref<1024xbf16>) {
    memref.assume_alignment %arg0, 32 : memref<1024xbf16>
    memref.assume_alignment %arg1, 32 : memref<1024xbf16>
    memref.assume_alignment %arg2, 32 : memref<1024xbf16>
    %cst = arith.constant 0.000000e+00 : bf16
    affine.for %arg3 = 0 to 1024 step 16 {
      %0 = vector.transfer_read %arg0[%arg3], %cst : memref<1024xbf16>, vector<16xbf16>
      %1 = vector.transfer_read %arg1[%arg3], %cst : memref<1024xbf16>, vector<16xbf16>
      %2 = arith.mulf %0, %1 : vector<16xbf16>
      vector.transfer_write %2, %arg2[%arg3] : vector<16xbf16>, memref<1024xbf16>
    }
    return
  }
}
