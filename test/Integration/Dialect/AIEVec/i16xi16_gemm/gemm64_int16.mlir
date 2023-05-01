// REQUIRES: valid_xchess_license
// RUN: aie-opt %s -affine-loop-unroll=unroll-factor=16 -affine-scalrep -canonicalize -affine-super-vectorize=virtual-vector-size=16 -convert-vector-to-aievec -lower-affine -canonicalize | aie-translate -aievec-to-cpp -o gen_aie.cc
// RUN: xchesscc_wrapper aie -f -g +s +w work +o work -I%S -I. %S/testbench.cc %S/kernel.cc
// RUN: cp -r %S/data . && xca_udm_dbg -qf -T -P %aietools/data/versal_prod/lib -t "%S/../profiling.tcl ./work/a.out" | FileCheck %s

func.func @matmul(%arg0: memref<64x64xi16>, %arg1: memref<64x64xi16>, %arg2: memref<64x64xi16>) {
  affine.for %arg3 = 0 to 64 {
    affine.for %arg4 = 0 to 64 {
      affine.for %arg5 = 0 to 64 {
        %0 = affine.load %arg0[%arg3, %arg5] : memref<64x64xi16>
        %1 = affine.load %arg1[%arg5, %arg4] : memref<64x64xi16>
        %2 = arith.muli %0, %1 : i16
        %3 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
        %4 = arith.addi %3, %2 : i16
        affine.store %4, %arg2[%arg3, %arg4] : memref<64x64xi16>
      }
    }
  }
  return
}

// CHECK-LABEL: N: 64, M: 64, K: 64
// CHECK-LABEL: Running MATMUL...
// CHECK: Cycle count: [[CC:[0-9]+]]
// CHECK-LABEL: Finish MATMUL!
// CHECK-LABEL: Compare the results
// CHECK: PASSED, Max delta: [[MD:-?[0-9]+]], pixel intensity
