// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=32 test-fastest-varying=0 vectorize-reductions=true" --convert-vector-to-aievec="aie-target=aie2" -lower-affine | aie-translate -aie2=true --aievec-to-cpp -o dut.cc
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. -c dut.cc -o dut.o
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. %S/testbench.cc work/dut.o
// RUN: mkdir -p data
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../../profiling.tcl ./work/a.out" >& xca_udm_dbg.stdout
// RUN: FileCheck --input-file=./xca_udm_dbg.stdout %s
// CHECK: TEST PASSED

module {
func.func @dut(%arg0: memref<1024xi16>, %arg1: memref<i16>) {
    %c0_i16 = arith.constant 0 : i16
    %0 = affine.for %arg2 = 0 to 1024 iter_args(%arg3 = %c0_i16) -> (i16) {
      %1 = affine.load %arg0[%arg2] : memref<1024xi16>
      %2 = arith.addi %arg3, %1 : i16
      affine.yield %2 : i16
    }
    affine.store %0, %arg1[] : memref<i16>
    return
  }
}
