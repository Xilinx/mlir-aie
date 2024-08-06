// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023, Advanced Micro Devices, Inc.

// REQUIRES: valid_xchess_license
// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=16" --convert-vector-to-aievec="aie-target=aie2" -lower-affine | aie-translate -aie2=true --aievec-to-cpp -o dut.cc
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I%aie_runtime_lib%/AIE2 -I %aietools/include -D__AIEARCH__=20 -D__AIENGINE__ -I. -c %aie_runtime_lib%/AIE2/lut_based_ops.cpp -o lut_based_ops.o
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I%aie_runtime_lib%/AIE2 -I %aietools/include -D__AIEARCH__=20 -D__AIENGINE__ -I. -c dut.cc -o dut.o
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I%aie_runtime_lib%/AIE2 -I %aietools/include -D__AIEARCH__=20 -D__AIENGINE__ -I. %S/testbench.cc work/dut.o work/lut_based_ops.o
// RUN: mkdir -p data
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work/a.out" >& xca_udm_dbg.stdout
// RUN: FileCheck --input-file=./xca_udm_dbg.stdout %s
// CHECK: TEST PASSED

module {
  func.func @dut(%arg0: memref<1024xbf16>, %arg1: f32, %arg2: memref<1024xbf16>) {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = arith.divf %cst, %arg1 : f32
    %1 = arith.truncf %0 : f32 to bf16
    affine.for %arg3 = 0 to 1024 {
      %2 = affine.load %arg0[%arg3] : memref<1024xbf16>
      %3 = arith.mulf %1, %2 : bf16
      affine.store %3, %arg2[%arg3] : memref<1024xbf16>
    }
    return
  }
}

