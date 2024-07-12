// REQUIRES: valid_xchess_license
// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=16 test-fastest-varying=0 vectorize-reductions=true" --convert-vector-to-aievec="aie-target=aie2" -lower-affine | aie-translate -aie2=true --aievec-to-cpp -o dut.cc
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I%aie_runtime_lib%/AIE2 -I %aietools/include -D__AIEARCH__=20 -D__AIENGINE__ -I. -c %aie_runtime_lib%/AIE2/lut_based_ops.cpp -o lut_based_ops.cpp.o 
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I%aie_runtime_lib%/AIE2 -I %aietools/include -D__AIEARCH__=20 -D__AIENGINE__ -I. -c dut.cc -o dut.cc.o
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I%aie_runtime_lib%/AIE2 -I %aietools/include -D__AIEARCH__=20 -D__AIENGINE__ -I. %S/testbench.cc work/dut.cc.o work/lut_based_ops.cpp.o
// RUN: mkdir -p data
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work/a.out" >& xca_udm_dbg.stdout
// RUN: FileCheck --input-file=./xca_udm_dbg.stdout %s
// CHECK: TEST PASSED
module {
  func.func @dut(%arg0: memref<1024xbf16>, %arg1: memref<1024xbf16>) {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    affine.for %arg2 = 0 to 1024 {
      %3 = affine.load %arg0[%arg2] : memref<1024xbf16>
      %4 = math.exp %3 : bf16
      affine.store %4, %arg0[%arg2] : memref<1024xbf16>
    }
    %0 = affine.for %arg2 = 0 to 1024 iter_args(%arg3 = %cst_0) -> (f32) {
      %3 = affine.load %arg0[%arg2] : memref<1024xbf16>
      %4 = arith.extf %3 : bf16 to f32
      %5 = arith.addf %arg3, %4 : f32
      affine.yield %5 : f32
    }
    %1 = arith.divf %cst, %0 : f32
    %2 = arith.truncf %1 : f32 to bf16
    affine.for %arg2 = 0 to 1024 {
      %3 = affine.load %arg0[%arg2] : memref<1024xbf16>
      %4 = arith.mulf %3, %2 : bf16
      affine.store %4, %arg1[%arg2] : memref<1024xbf16>
    }
    return
  }
}
