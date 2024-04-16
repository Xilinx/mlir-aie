// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023, Advanced Micro Devices, Inc.

// REQUIRES: valid_xchess_license
// REQUIRES: peano
// RUN: mkdir -p %t/data; cd %t
// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=32" %vector-to-llvmir% -o llvmir.mlir
// RUN: aie-translate llvmir.mlir %llvmir-to-ll% -o dut.ll
// RUN: %PEANO_INSTALL_DIR/bin/clang %clang_aie2_args -c dut.ll -o dut.o
// RUN: xchesscc_wrapper %xchesscc_aie2_args -DTO_LLVM +w work +o work -I%S -I. %S/testbench.cc dut.o
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work/a.out" >& xca_udm_dbg.stdout
// RUN: FileCheck --input-file=./xca_udm_dbg.stdout %s
// CHECK: TEST PASSED

module {
  func.func @dut(%arg0: memref<1024xi8>, %arg1: memref<1024xi8>, %arg2: memref<1024xi8>) {
    memref.assume_alignment %arg0, 32 : memref<1024xi8>
    memref.assume_alignment %arg1, 32 : memref<1024xi8>
    memref.assume_alignment %arg2, 32 : memref<1024xi8>
    %c0_i8 = arith.constant 0 : i8
    affine.for %arg3 = 0 to 1024 step 32 {
      %0 = vector.transfer_read %arg0[%arg3], %c0_i8 : memref<1024xi8>, vector<32xi8>
      %1 = vector.transfer_read %arg1[%arg3], %c0_i8 : memref<1024xi8>, vector<32xi8>
      %2 = arith.muli %0, %1 : vector<32xi8>
      vector.transfer_write %2, %arg2[%arg3] : vector<32xi8>, memref<1024xi8>
    }
    return
  }
}
