// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023, Advanced Micro Devices, Inc.

// REQUIRES: valid_xchess_license

// To-CPP flow
// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=32" --convert-vector-to-aievec="aie-target=aieml" -lower-affine | aie-translate -aieml=true --aievec-to-cpp -o dut.cc
// RUN: xchesscc_wrapper %xchesscc_aie2_args +w work_cpp +o work_cpp -I%S -I. -c dut.cc -o dut.o
// RUN: xchesscc_wrapper %xchesscc_aie2_args -DTO_CPP +w work_cpp +o work_cpp -I%S -I. %S/testbench.cc work_cpp/dut.o
// RUN: mkdir -p data
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work_cpp/a.out" >& xca_udm_dbg.cpp.stdout
// RUN: FileCheck --input-file=./xca_udm_dbg.cpp.stdout %s

// To-Vector-LLVM flow
// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=32" %vector-to-llvmir% -o llvmir.vector.mlir
// RUN: aie-translate llvmir.vector.mlir %llvmir-to-ll% -o dut.vector.ll
// RUN: %clang %clang_aie2_args -c dut.vector.ll -o dut.vector.o
// RUN: xchesscc_wrapper %xchesscc_aie2_args -DTO_LLVM +w work_vector_llvm +o work_vector_llvm -I%S -I. %S/testbench.cc dut.vector.o
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work_vector_llvm/a.out" >& xca_udm_dbg.vector_llvm.stdout
// RUN: FileCheck --input-file=./xca_udm_dbg.vector_llvm.stdout %s

// To-Scalar-LLVM flow
// RUN: aie-opt %s %vector-to-llvmir% -o llvmir.scalar.mlir
// RUN: aie-translate llvmir.scalar.mlir %llvmir-to-ll% -o dut.scalar.ll
// RUN: %clang %clang_aie2_args -c dut.scalar.ll -o dut.scalar.o
// RUN: xchesscc_wrapper %xchesscc_aie2_args -DTO_LLVM +w work_scalar_llvm +o work_scalar_llvm -I%S -I. %S/testbench.cc dut.scalar.o
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work_scalar_llvm/a.out" >& xca_udm_dbg.scalar_llvm.stdout
// RUN: FileCheck --input-file=./xca_udm_dbg.scalar_llvm.stdout %s

// CHECK: TEST PASSED

module {
  func.func @dut(%arg0: memref<1024xi16>, %arg1: memref<1024xi16>, %arg2: memref<1024xi16>) {
    memref.assume_alignment %arg0, 32 : memref<1024xi16>
    memref.assume_alignment %arg1, 32 : memref<1024xi16>
    memref.assume_alignment %arg2, 32 : memref<1024xi16>
    affine.for %arg3 = 0 to 1024 {
      %0 = affine.load %arg0[%arg3] : memref<1024xi16>
      %1 = affine.load %arg1[%arg3] : memref<1024xi16>
      %2 = arith.muli %0, %1 : i16
      affine.store %2, %arg2[%arg3] : memref<1024xi16>
    }
    return
  }
}
