// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aieml" -lower-affine | aie-translate -aieml=true --aievec-to-cpp -o dut.cc
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. -c dut.cc -o dut.o
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. %S/testbench.cc work/dut.o
// RUN: mkdir -p data
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work/a.out" >& xca_udm_dbg.stdout
// RUN: FileCheck --input-file=./xca_udm_dbg.stdout %s
// CHECK: TEST PASSED

module {
  func.func @dut(%arg0: memref<1024xbf16>, %arg1: memref<bf16>) {
    %cst_0 = arith.constant dense<0x7F80> : vector<32xbf16>
    %0 = affine.for %arg2 = 0 to 1024 step 32 iter_args(%arg3 = %cst_0) -> (vector<32xbf16>) {
      %cst_1 = arith.constant 0.000000e+00 : bf16
      %3 = vector.transfer_read %arg0[%arg2], %cst_1 : memref<1024xbf16>, vector<32xbf16>
      %4 = arith.minimumf %arg3, %3 : vector<32xbf16>
      affine.yield %4 : vector<32xbf16>
    }
    %1 = vector.reduction <minf>, %0 : vector<32xbf16> into bf16
    affine.store %1, %arg1[] : memref<bf16>
    return
  }
}

