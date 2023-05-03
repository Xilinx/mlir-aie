// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aieml" -lower-affine | aie-translate -aieml=true --aievec-to-cpp -o dut.cc
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. %S/testbench.cc dut.cc
// RUN: mkdir -p data
// RUN: xme_ca_udm_dbg -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work/a.out" >& xme_ca_udm_dbg.stdout
// RUN: FileCheck --input-file=./xme_ca_udm_dbg.stdout %s
// CHECK: TEST PASSED

module {
  func.func @dut(%arg0: memref<1024xi8>, %arg1: memref<1024xi8>, %arg2: memref<1024xi32>) {
    %c0_i8 = arith.constant 0 : i8
    affine.for %arg3 = 0 to 1024 step 32 {
      %0 = vector.transfer_read %arg0[%arg3], %c0_i8 : memref<1024xi8>, vector<32xi8>
      %1 = arith.extsi %0 : vector<32xi8> to vector<32xi32>
      %2 = vector.transfer_read %arg1[%arg3], %c0_i8 : memref<1024xi8>, vector<32xi8>
      %3 = arith.extsi %2 : vector<32xi8> to vector<32xi32>
      %4 = arith.addi %1, %3 : vector<32xi32>
      vector.transfer_write %4, %arg2[%arg3] : vector<32xi32>, memref<1024xi32>
    }
    return
  }
}
