// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" -lower-affine | aie-translate -aie2=true --aievec-to-cpp -o dut.cc
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. -c dut.cc -o dut.o
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. %S/testbench.cc work/dut.o
// RUN: mkdir -p data
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../../profiling.tcl ./work/a.out" >& xca_udm_dbg.stdout
// RUN: FileCheck --input-file=./xca_udm_dbg.stdout %s
// CHECK: TEST PASSED

module {
  func.func @dut(%arg0: memref<1024xf32>, %arg1: memref<f32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<0xFF800000> : vector<16xf32>
    %0 = affine.for %arg2 = 0 to 1024 step 16 iter_args(%arg3 = %cst_0) -> (vector<16xf32>) {
      %3 = vector.transfer_read %arg0[%arg2], %cst : memref<1024xf32>, vector<16xf32>
      %4 = arith.maximumf %arg3, %3 : vector<16xf32>
      affine.yield %4 : vector<16xf32>
    }
    %1 = vector.reduction <maximumf>, %0 : vector<16xf32> into f32
    affine.store %1, %arg1[] : memref<f32>
    return
  }
}
